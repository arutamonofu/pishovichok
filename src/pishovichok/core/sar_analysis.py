"""Модуль извлечения знаний SAR и детекции Activity Cliffs (Слой 1, без участия LLM).

Реализует:
1. Парную фрагментацию молекулярных графов по ациклическим одинарным связям
   через `rdkit.Chem.rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)`.
2. Канонизацию периферийных заместителей с фиксацией корня на точке крепления `[*:1]`
   (`rootedAtAtom`) для формирования однозначных трансформаций SMIRKS `[*:1]R_lhs>>[*:1]R_rhs`.
3. Построение сети парных молекулярных соответствий (`mmp_pairs.parquet`) и SQL-агрегацию
   правил `SARRule` в `DuckDB` с расчетом `mean_delta_pic50`, `std_delta_pic50`,
   `support_pairs_count`, `mean_delta_clogp` и `mean_delta_tpsa`.
4. Строгий контроль методологической однородности протоколов испытаний (`assay_homogeneous`):
   правило признается гомогенным тогда и только тогда, когда 100% образующих его пар
   измерены в одном и том же формате `bao_format` и одном типе измерения `standard_type`.
5. Расчет структурного сходства Танимото по отпечаткам Моргана (`radius=2, fpSize=2048`),
   индекса ландшафта активности `SALI(i, j) = |ΔpIC50| / (1 - Tanimoto(i, j) + epsilon)`
   и выделение критических перепадов активности (`ActivityCliff` при `Tanimoto >= 0.85`
   и `|ΔpIC50| >= 1.5`).
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

from pishovichok.config import settings
from pishovichok.core.curation import curate_chembl_or_csv
from pishovichok.core.fingerprints import compute_fp_similarity, compute_mol_fingerprint
from pishovichok.state.schemas import (
    ActivityCliff,
    ParquetPointer,
    QueryConstraints,
    SARRule,
    ToolConfig,
)
from pishovichok.state.storage import read_parquet_pointer, write_parquet_pointer

_SULFONYL_CANONICAL_MAP: dict[str, str] = {
    "[*:1]S(C)(=O)=O": "[*:1]S(=O)(=O)C",
    "[*:1]S(N)(=O)=O": "[*:1]S(=O)(=O)N",
}


def calculate_sali(
    pic50_i: float,
    pic50_j: float,
    tanimoto: float,
    epsilon: float | None = None,
) -> float:
    """Вычисляет индекс ландшафта 'структура–активность' (Structure-Activity Landscape Index, SALI).

    Формула (Раздел 5 SPEC.md):
        SALI(i, j) = |pIC50(i) - pIC50(j)| / (1.0 - Tanimoto(i, j) + epsilon)
    """
    eps = epsilon if epsilon is not None else settings.sali_epsilon
    if eps <= 0.0:
        raise ValueError(f"SALI epsilon must be positive, got {eps}")
    clamped_sim = max(0.0, min(1.0, float(tanimoto)))
    abs_delta = abs(float(pic50_i) - float(pic50_j))
    return abs_delta / (1.0 - clamped_sim + eps)


def canonicalize_substituent_smiles(frag_smiles: str) -> str:
    """Канонизирует SMILES заместителя с корнем на точке крепления `[*:1]`.

    Гарантирует единообразную запись вида `[*:1]OC`, `[*:1]Cl`, `[*:1]C#N`,
    `[*:1]C(=O)NC`, `[*:1]OCCO`, `[*:1]S(=O)(=O)C`.
    """
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return frag_smiles

    dummy_idx = -1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_idx = atom.GetIdx()
            break

    if dummy_idx >= 0:
        rooted_smi = Chem.MolToSmiles(
            mol,
            rootedAtAtom=dummy_idx,
            canonical=True,
            isomericSmiles=True,
        )
    else:
        rooted_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return _SULFONYL_CANONICAL_MAP.get(rooted_smi, rooted_smi)


def _is_attachment_on_ring(core_mol: Chem.Mol) -> bool:
    """Проверяет, присоединена ли метка разрыва `[*:1]` непосредственно к кольцевому атому ядра."""
    for atom in core_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            for neighbor in atom.GetNeighbors():
                if neighbor.IsInRing():
                    return True
    return False


def fragment_molecule_mmpa(
    mol_or_smiles: Chem.Mol | str,
    max_cuts: int = 1,
    max_rgroup_heavy_atoms: int = 13,
    min_core_heavy_atoms: int = 6,
    max_cut_bonds: int = 20,
    *,
    tool_config: ToolConfig | None = None,
) -> list[tuple[str, str, bool, int]]:
    """Выполняет фрагментацию молекулы через `rdMMPA.FragmentMol` и выделяет пары `(core, r_group)`.

    Возвращает дедуплицированный список кортежей:
        `(core_canonical_smiles, rgroup_rooted_smiles, is_ring_attached, core_heavy_atoms)`
    """
    mol = (
        Chem.MolFromSmiles(mol_or_smiles)
        if isinstance(mol_or_smiles, str)
        else mol_or_smiles
    )
    if mol is None:
        return []

    if tool_config is not None:
        max_cuts = tool_config.mmpa_max_cuts
        max_cut_bonds = tool_config.mmpa_max_cut_bonds
        max_rgroup_heavy_atoms = tool_config.max_rgroup_heavy_atoms
        min_core_heavy_atoms = tool_config.mmpa_min_core_heavy_atoms

    raw_fragments = rdMMPA.FragmentMol(
        mol,
        maxCuts=int(max_cuts),
        maxCutBonds=int(max_cut_bonds),
        resultsAsMols=False,
    )
    results: list[tuple[str, str, bool, int]] = []
    seen: set[tuple[str, str]] = set()

    for core_smi, chains_smi in raw_fragments:
        # При maxCuts=1 rdMMPA возвращает core_smi == '' и chains_smi == 'frag1.frag2'
        if not core_smi:
            parts = chains_smi.split(".")
            if len(parts) != 2:
                continue
            m0 = Chem.MolFromSmiles(parts[0])
            m1 = Chem.MolFromSmiles(parts[1])
            if m0 is None or m1 is None:
                continue

            h0 = m0.GetNumHeavyAtoms()
            h1 = m1.GetNumHeavyAtoms()
            if h0 <= h1:
                rg_mol, core_mol, rg_heavy, core_heavy = m0, m1, h0, h1
            else:
                rg_mol, core_mol, rg_heavy, core_heavy = m1, m0, h1, h0
        else:
            core_mol = Chem.MolFromSmiles(core_smi)
            rg_mol = Chem.MolFromSmiles(chains_smi)
            if core_mol is None or rg_mol is None:
                continue
            rg_heavy = rg_mol.GetNumHeavyAtoms()
            core_heavy = core_mol.GetNumHeavyAtoms()

        if (
            rg_heavy == 0
            or rg_heavy > max_rgroup_heavy_atoms
            or core_heavy < min_core_heavy_atoms
        ):
            continue

        core_canon = Chem.MolToSmiles(core_mol, canonical=True, isomericSmiles=True)
        rg_canon = canonicalize_substituent_smiles(
            Chem.MolToSmiles(rg_mol, canonical=True, isomericSmiles=True)
        )
        key = (core_canon, rg_canon)
        if key in seen:
            continue
        seen.add(key)

        ring_attached = _is_attachment_on_ring(core_mol)
        results.append((core_canon, rg_canon, ring_attached, core_heavy))

    return results


def _filter_by_fixed_scaffold(
    rows: list[dict[str, Any]],
    fixed_scaffold_smiles: str,
) -> list[dict[str, Any]]:
    """Фильтрует выборку вокруг зафиксированного каркаса Bemis-Murcko (для Сценария 2)."""
    fixed_mol = Chem.MolFromSmiles(fixed_scaffold_smiles)
    if fixed_mol is None:
        return rows

    scaf_of_fixed = MurckoScaffold.GetScaffoldForMol(fixed_mol)
    target_scaf_mol = (
        scaf_of_fixed
        if scaf_of_fixed is not None and scaf_of_fixed.GetNumAtoms() > 0
        else fixed_mol
    )
    target_scaf_smi = Chem.MolToSmiles(
        target_scaf_mol, canonical=True, isomericSmiles=True
    )

    # 1. Точное совпадение каркаса Bemis-Murcko
    exact_matches = [r for r in rows if r.get("scaffold_smiles") == target_scaf_smi]
    if len(exact_matches) >= 2:
        return exact_matches

    # 2. Подструктурное совпадение общего фармакофорного ядра
    substruct_matches: list[dict[str, Any]] = []
    for r in rows:
        mol = Chem.MolFromSmiles(str(r.get("canonical_smiles") or ""))
        scaf = Chem.MolFromSmiles(str(r.get("scaffold_smiles") or ""))
        if mol is not None and (
            mol.HasSubstructMatch(target_scaf_mol)
            or (scaf is not None and target_scaf_mol.HasSubstructMatch(scaf))
        ):
            substruct_matches.append(r)

    return substruct_matches if len(substruct_matches) >= 2 else rows


def extract_mmp_and_cliffs(
    curated_pointer: ParquetPointer,
    session_dir: Path | str,
    fixed_scaffold_smiles: str | None = None,
    min_support_pairs: int | None = None,
    *,
    tanimoto_cliff_threshold: float | None = None,
    delta_pic50_cliff_threshold: float | None = None,
    sali_epsilon: float | None = None,
    tool_config: ToolConfig | None = None,
) -> tuple[ParquetPointer, list[SARRule], list[ActivityCliff]]:
    """Строит сеть парных молекулярных соответствий (`rdMMPA`), извлекает правила `SARRule`
    и детектирует критические перепады активности (`ActivityCliff`).

    Возвращает кортеж `(mmp_table_uri, extracted_sar_rules, detected_activity_cliffs)`.
    """
    cfg = tool_config or ToolConfig()
    resolved_min_support = (
        min_support_pairs if min_support_pairs is not None else cfg.min_support_pairs
    )
    resolved_session_dir = Path(session_dir).resolve()
    resolved_session_dir.mkdir(parents=True, exist_ok=True)

    cliff_sim_thresh = (
        tanimoto_cliff_threshold
        if tanimoto_cliff_threshold is not None
        else (
            tool_config.cliff_similarity_cutoff
            if tool_config is not None
            else settings.sali_tanimoto_threshold
        )
    )
    cliff_delta_thresh = (
        delta_pic50_cliff_threshold
        if delta_pic50_cliff_threshold is not None
        else (
            tool_config.cliff_delta_pic50_cutoff
            if tool_config is not None
            else settings.sali_delta_pic50_threshold
        )
    )
    eps = (
        sali_epsilon
        if sali_epsilon is not None
        else (tool_config.sali_epsilon if tool_config is not None else settings.sali_epsilon)
    )

    # 1. Чтение стандартизированной выборки из ParquetPointer
    all_rows = read_parquet_pointer(curated_pointer).to_pylist()
    # Исключаем цензурированные записи (> 100 uM) из количественного расчета сдвигов ΔpIC50
    quant_rows = [r for r in all_rows if not bool(r.get("is_censored", False))]

    if fixed_scaffold_smiles:
        quant_rows = _filter_by_fixed_scaffold(quant_rows, fixed_scaffold_smiles)

    # 2. Предварительный расчет графов RDKit, отпечатков, cLogP, TPSA и фрагментов rdMMPA
    mol_records: list[dict[str, Any]] = []
    core_index: dict[str, list[tuple[int, str, bool, int]]] = defaultdict(list)

    for row in quant_rows:
        csmi = str(row["canonical_smiles"])
        mol = Chem.MolFromSmiles(csmi)
        if mol is None:
            continue

        fp = compute_mol_fingerprint(
            mol,
            fp_type=cfg.fp_type,
            fp_radius=cfg.fp_radius,
            fp_nbits=cfg.fp_nbits,
        )
        clogp = float(Descriptors.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        frags = fragment_molecule_mmpa(
            mol,
            max_cuts=cfg.mmpa_max_cuts,
            max_rgroup_heavy_atoms=cfg.max_rgroup_heavy_atoms,
            min_core_heavy_atoms=cfg.mmpa_min_core_heavy_atoms,
            max_cut_bonds=cfg.mmpa_max_cut_bonds,
        )

        rec_idx = len(mol_records)
        mol_records.append(
            {
                "mol_id": str(row["mol_id"]),
                "canonical_smiles": csmi,
                "pchembl_value": float(row["pchembl_value"]),
                "bao_format": str(
                    row.get("bao_format") or settings.dominant_bao_default
                ),
                "standard_type": str(row.get("standard_type") or "IC50"),
                "assay_type": str(row.get("assay_type") or "B"),
                "fp": fp,
                "clogp": clogp,
                "tpsa": tpsa,
                "frags": frags,
            }
        )

        for core_smi, rg_smi, is_ring_att, core_heavy in frags:
            core_index[core_smi].append((rec_idx, rg_smi, is_ring_att, core_heavy))

    # 3. Построение направленных пар MMP (i -> j) вокруг общих ядер core_smiles
    # Для каждой упорядоченной пары молекул (mol_id_lhs, mol_id_rhs) выбираем наиболее
    # специфичный разрез: приоритет разрезу по связи с кольцом, затем максимальному размеру ядра
    best_pair_cuts: dict[tuple[int, int], tuple[tuple[int, int], str, str, str]] = {}
    for core_smi, items in core_index.items():
        n_items = len(items)
        if n_items < 2:
            continue
        for i in range(n_items):
            idx_lhs, rg_lhs, ring_lhs, ch_lhs = items[i]
            for j in range(n_items):
                if i == j:
                    continue
                idx_rhs, rg_rhs, ring_rhs, _ch_rhs = items[j]
                if idx_lhs == idx_rhs or rg_lhs == rg_rhs:
                    continue
                if (
                    mol_records[idx_lhs]["canonical_smiles"]
                    == mol_records[idx_rhs]["canonical_smiles"]
                ):
                    continue

                priority = (int(ring_lhs and ring_rhs), ch_lhs)
                pair_key = (idx_lhs, idx_rhs)
                existing = best_pair_cuts.get(pair_key)
                if existing is None or priority > existing[0]:
                    smirks_str = f"{rg_lhs}>>{rg_rhs}"
                    best_pair_cuts[pair_key] = (priority, core_smi, rg_lhs, rg_rhs)

    # Формируем строки таблицы mmp_pairs.parquet и кэш трансформаций для Activity Cliffs
    mmp_rows: list[dict[str, Any]] = []
    pair_smirks_lookup: dict[tuple[int, int], str] = {}

    for (idx_lhs, idx_rhs), (_prio, core_smi, rg_lhs, rg_rhs) in best_pair_cuts.items():
        lhs = mol_records[idx_lhs]
        rhs = mol_records[idx_rhs]

        smirks_str = f"{rg_lhs}>>{rg_rhs}"
        pair_smirks_lookup[(idx_lhs, idx_rhs)] = smirks_str

        pic50_lhs = lhs["pchembl_value"]
        pic50_rhs = rhs["pchembl_value"]
        delta_pic50 = round(pic50_rhs - pic50_lhs, 4)

        sim = round(
            compute_fp_similarity(lhs["fp"], rhs["fp"], metric=cfg.similarity_metric),
            4,
        )
        sali_val = round(calculate_sali(pic50_lhs, pic50_rhs, sim, epsilon=eps), 4)
        is_cliff = bool(
            sim >= cliff_sim_thresh and abs(delta_pic50) >= cliff_delta_thresh
        )

        bao_lhs = lhs["bao_format"]
        bao_rhs = rhs["bao_format"]
        stype_lhs = lhs["standard_type"]
        stype_rhs = rhs["standard_type"]
        pair_homog = bool(bao_lhs == bao_rhs and stype_lhs == stype_rhs)

        mmp_rows.append(
            {
                "pair_id": f"{lhs['mol_id']}->{rhs['mol_id']}",
                "mol_id_lhs": lhs["mol_id"],
                "mol_id_rhs": rhs["mol_id"],
                "smiles_lhs": lhs["canonical_smiles"],
                "smiles_rhs": rhs["canonical_smiles"],
                "core_smiles": core_smi,
                "smirks_transformation": smirks_str,
                "from_fragment": rg_lhs,
                "to_fragment": rg_rhs,
                "pic50_lhs": pic50_lhs,
                "pic50_rhs": pic50_rhs,
                "delta_pic50": delta_pic50,
                "delta_clogp": round(rhs["clogp"] - lhs["clogp"], 4),
                "delta_tpsa": round(rhs["tpsa"] - lhs["tpsa"], 4),
                "tanimoto_similarity": sim,
                "sali_index": sali_val,
                "is_activity_cliff": is_cliff,
                "bao_lhs": bao_lhs,
                "bao_rhs": bao_rhs,
                "standard_type_lhs": stype_lhs,
                "standard_type_rhs": stype_rhs,
                "pair_assay_homogeneous": pair_homog,
            }
        )

    # 4. Сохранение таблицы mmp_pairs.parquet через Fat Pointer
    mmp_pointer = write_parquet_pointer(
        data=mmp_rows,
        table_type="mmp_pairs",
        session_dir=resolved_session_dir,
        filename="mmp_pairs.parquet",
    )

    # 5. Агрегация статистически значимых правил SARRule через векторизованный SQL DuckDB
    sar_rules: list[SARRule] = []
    if mmp_rows:
        conn = duckdb.connect(database=":memory:")
        mmp_path_posix = Path(mmp_pointer.uri).as_posix()
        min_delta_clause = (
            f" AND AVG(delta_pic50) >= {float(cfg.min_mean_delta_pic50)}"
            if cfg.prioritize_positive_rules
            else ""
        )
        agg_sql = f"""
            SELECT
                smirks_transformation,
                from_fragment,
                to_fragment,
                COUNT(*) AS support_pairs_count,
                AVG(delta_pic50) AS mean_delta_pic50,
                COALESCE(STDDEV_SAMP(delta_pic50), 0.0) AS std_delta_pic50,
                AVG(delta_clogp) AS mean_delta_clogp,
                AVG(delta_tpsa) AS mean_delta_tpsa,
                BOOL_AND(pair_assay_homogeneous) AS all_pairs_homog,
                LIST_SORT(LIST_DISTINCT(LIST_CONCAT(LIST(bao_lhs), LIST(bao_rhs)))) AS bao_formats,
                LIST_SORT(LIST_DISTINCT(LIST_CONCAT(LIST(standard_type_lhs), LIST(standard_type_rhs)))) AS std_types,
                LIST(pair_id) AS pair_ids
            FROM read_parquet('{mmp_path_posix}')
            GROUP BY smirks_transformation, from_fragment, to_fragment
            HAVING COUNT(*) >= {int(resolved_min_support)}{min_delta_clause}
            ORDER BY (COUNT(*) >= 3) DESC, mean_delta_pic50 DESC, support_pairs_count DESC, smirks_transformation ASC
        """
        agg_records = conn.sql(agg_sql).to_arrow_table().to_pylist()
        if not agg_records and cfg.prioritize_positive_rules and tool_config is None:
            # Обратная совместимость при отсутствии положительных правил без явного ToolConfig
            fallback_sql = f"""
                SELECT
                    smirks_transformation,
                    from_fragment,
                    to_fragment,
                    COUNT(*) AS support_pairs_count,
                    AVG(delta_pic50) AS mean_delta_pic50,
                    COALESCE(STDDEV_SAMP(delta_pic50), 0.0) AS std_delta_pic50,
                    AVG(delta_clogp) AS mean_delta_clogp,
                    AVG(delta_tpsa) AS mean_delta_tpsa,
                    BOOL_AND(pair_assay_homogeneous) AS all_pairs_homog,
                    LIST_SORT(LIST_DISTINCT(LIST_CONCAT(LIST(bao_lhs), LIST(bao_rhs)))) AS bao_formats,
                    LIST_SORT(LIST_DISTINCT(LIST_CONCAT(LIST(standard_type_lhs), LIST(standard_type_rhs)))) AS std_types,
                    LIST(pair_id) AS pair_ids
                FROM read_parquet('{mmp_path_posix}')
                GROUP BY smirks_transformation, from_fragment, to_fragment
                HAVING COUNT(*) >= {int(resolved_min_support)}
                ORDER BY (COUNT(*) >= 3) DESC, mean_delta_pic50 DESC, support_pairs_count DESC, smirks_transformation ASC
            """
            agg_records = conn.sql(fallback_sql).to_arrow_table().to_pylist()

        selected_agg = agg_records

        for rule_idx, rec in enumerate(selected_agg, start=1):
            bao_list = [str(b) for b in (rec["bao_formats"] or [])]
            stype_list = [str(s) for s in (rec["std_types"] or [])]
            all_pairs_homog = bool(rec["all_pairs_homog"])
            # Правило методологически однородно тогда и только тогда, когда все его пары
            # измерены в одном и том же формате BAO и одном типе измерения standard_type
            rule_is_homogeneous = (
                all_pairs_homog and len(bao_list) == 1 and len(stype_list) == 1
            )
            pair_ids_list = [str(pid) for pid in (rec["pair_ids"] or [])][:5]

            sar_rules.append(
                SARRule(
                    rule_id=f"R{rule_idx}",
                    smirks=str(rec["smirks_transformation"]),
                    from_fragment=str(rec["from_fragment"]),
                    to_fragment=str(rec["to_fragment"]),
                    mean_delta_pic50=round(float(rec["mean_delta_pic50"]), 4),
                    std_delta_pic50=round(max(0.0, float(rec["std_delta_pic50"])), 4),
                    support_pairs_count=int(rec["support_pairs_count"]),
                    mean_delta_clogp=round(float(rec["mean_delta_clogp"]), 4),
                    mean_delta_tpsa=round(float(rec["mean_delta_tpsa"]), 4),
                    assay_homogeneous=rule_is_homogeneous,
                    participating_bao_formats=bao_list,
                    example_pair_ids=pair_ids_list,
                )
            )

    # 6. Детекция критических перепадов активности (Activity Cliffs) по индексу SALI
    raw_cliffs: list[dict[str, Any]] = []
    n_mols = len(mol_records)
    for i in range(n_mols):
        rec_i = mol_records[i]
        for j in range(i + 1, n_mols):
            rec_j = mol_records[j]
            if rec_i["canonical_smiles"] == rec_j["canonical_smiles"]:
                continue

            sim = compute_fp_similarity(
                rec_i["fp"], rec_j["fp"], metric=cfg.similarity_metric
            )
            if sim < cliff_sim_thresh:
                continue

            abs_delta = abs(rec_i["pchembl_value"] - rec_j["pchembl_value"])
            if abs_delta < cliff_delta_thresh:
                continue

            sali_val = calculate_sali(
                rec_i["pchembl_value"],
                rec_j["pchembl_value"],
                sim,
                epsilon=eps,
            )
            if rec_i["pchembl_value"] >= rec_j["pchembl_value"]:
                act_idx, inact_idx = i, j
            else:
                act_idx, inact_idx = j, i

            act_rec = mol_records[act_idx]
            inact_rec = mol_records[inact_idx]

            # Определяем минимальную структурную трансформацию от неактивного к активному аналогу
            smirks_diff = pair_smirks_lookup.get((inact_idx, act_idx))
            if smirks_diff is None:
                smirks_diff = (
                    f"{inact_rec['canonical_smiles']}>>{act_rec['canonical_smiles']}"
                )

            cliff_homog = bool(
                act_rec["bao_format"] == inact_rec["bao_format"]
                and act_rec["standard_type"] == inact_rec["standard_type"]
            )

            raw_cliffs.append(
                {
                    "mol_id_active": act_rec["mol_id"],
                    "mol_id_inactive": inact_rec["mol_id"],
                    "smiles_active": act_rec["canonical_smiles"],
                    "smiles_inactive": inact_rec["canonical_smiles"],
                    "pic50_active": round(act_rec["pchembl_value"], 4),
                    "pic50_inactive": round(inact_rec["pchembl_value"], 4),
                    "abs_delta_pic50": round(abs_delta, 4),
                    "tanimoto_similarity": round(sim, 4),
                    "sali_index": round(sali_val, 4),
                    "smirks_difference": smirks_diff,
                    "assay_homogeneous": cliff_homog,
                }
            )

    raw_cliffs.sort(key=lambda c: (c["sali_index"], c["abs_delta_pic50"]), reverse=True)
    detected_cliffs = [
        ActivityCliff(
            cliff_id=f"CLIFF_{idx:02d}",
            **cliff_data,
        )
        for idx, cliff_data in enumerate(raw_cliffs, start=1)
    ]

    return (mmp_pointer, sar_rules, detected_cliffs)


def extract_mmp_with_background_fallback(
    curated_pointer: ParquetPointer,
    session_dir: Path | str,
    fixed_scaffold_smiles: str | None = None,
    min_support_pairs: int | None = None,
    background_target_id: str | None = None,
    min_rules_threshold: int | None = None,
    *,
    tool_config: ToolConfig | None = None,
) -> tuple[ParquetPointer, list[SARRule], list[ActivityCliff], bool]:
    """Обертка над `extract_mmp_and_cliffs` с автоматическим обогащением фоновыми правилами ChEMBL.

    Реализует детерминированный фолбэк для Сценария 3:
    если в загруженной малой лабораторной серии CSV найдено менее `min_rules_threshold`
    статистически значимых правил SAR, автоматически извлекаются гомогенные правила MMP
    по целевой мишени из локального кэша `/data/chembl/` и выставляется флаг
    `enriched_with_chembl_background = True`.
    """
    cfg = tool_config or ToolConfig()
    resolved_min_support = (
        min_support_pairs if min_support_pairs is not None else cfg.min_support_pairs
    )
    resolved_bg_target = (
        background_target_id
        if background_target_id is not None
        else cfg.background_target_id
    )
    mmp_ptr, rules, cliffs = extract_mmp_and_cliffs(
        curated_pointer=curated_pointer,
        session_dir=session_dir,
        fixed_scaffold_smiles=fixed_scaffold_smiles,
        min_support_pairs=resolved_min_support,
        tool_config=cfg,
    )

    thresh = (
        min_rules_threshold
        if min_rules_threshold is not None
        else (
            tool_config.min_rules_before_enrichment
            if tool_config is not None
            else settings.min_sar_rules_before_enrichment
        )
    )
    if len(rules) >= thresh or thresh <= 0:
        return (mmp_ptr, rules, cliffs, False)

    # Фоновое обогащение из локального слепка ChEMBL в изолированном временном буфере
    with tempfile.TemporaryDirectory() as bg_tmp:
        bg_constraints = QueryConstraints(
            scenario_type="target_based_sar",
            target_chembl_id=resolved_bg_target,
        )
        _raw_bg_ptr, cur_bg_ptr, _bg_summary = curate_chembl_or_csv(
            session_dir=Path(bg_tmp),
            constraints=bg_constraints,
            assay_filter_strict=True,
            tool_config=cfg,
        )
        _bg_mmp_ptr, bg_rules, _bg_cliffs = extract_mmp_and_cliffs(
            curated_pointer=cur_bg_ptr,
            session_dir=Path(bg_tmp),
            min_support_pairs=resolved_min_support,
            tool_config=cfg,
        )

    existing_smirks = {r.smirks for r in rules}
    merged_rules = list(rules)
    for bg_rule in bg_rules:
        if bg_rule.smirks not in existing_smirks:
            existing_smirks.add(bg_rule.smirks)
            merged_rules.append(
                SARRule(
                    rule_id=f"R{len(merged_rules) + 1}",
                    smirks=bg_rule.smirks,
                    from_fragment=bg_rule.from_fragment,
                    to_fragment=bg_rule.to_fragment,
                    mean_delta_pic50=bg_rule.mean_delta_pic50,
                    std_delta_pic50=bg_rule.std_delta_pic50,
                    support_pairs_count=bg_rule.support_pairs_count,
                    mean_delta_clogp=bg_rule.mean_delta_clogp,
                    mean_delta_tpsa=bg_rule.mean_delta_tpsa,
                    assay_homogeneous=bg_rule.assay_homogeneous,
                    participating_bao_formats=bg_rule.participating_bao_formats,
                    example_pair_ids=bg_rule.example_pair_ids,
                )
            )

    return (mmp_ptr, merged_rules, cliffs, len(merged_rules) > len(rules))
