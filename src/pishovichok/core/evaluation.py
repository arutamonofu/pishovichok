"""Модуль расчета физико-химических дескрипторов, подструктурных фильтров и ретросинтеза BRICS (Слой 1).

Выполняется на 100% детерминированно без вызовов LLM (Разделы 3.5, 3.6 и 6.1.4 ARCHITECTURE.md):
1. Расчет дескрипторов RDKit: `ExactMolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`,
   `NumRotatableBonds`, `FractionCSP3`, `QED.qed` и числа нарушений правила пяти Липински.
2. Расчет индекса синтетической доступности Эртля (`SAscore` в диапазоне `[1.0, 10.0]`) через
   штатный модуль `RDConfig.RDContribDir / 'SA_Score'` (`sascorer.calculateScore`) с резервной
   формулой топологической сложности на базе `BertzCT`, спиро- и мостиковых атомов.
3. Проверка структурных алертов реакционной способности и пан-скрининговых интерференций по
   каталогам `FilterCatalogParams.FilterCatalogs.PAINS`, `BRENK` и `CHEMBL_Glaxo`.
4. Ретросинтетическая декомпозиция связей по алгоритму `BRICS.BRICSDecompose` с очисткой
   изотопных меток точек разрыва (`[1*]`, `[3*]`, `[16*]` -> `[H]`) и канонизацией прекурсоров.
"""

from __future__ import annotations

import contextlib
import math
import re
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem, RDConfig
from rdkit.Chem import (
    BRICS,
    QED,
    Descriptors,
    FilterCatalog,
    GraphDescriptors,
    rdMolDescriptors,
)
from rdkit.ML.Descriptors import MoleculeDescriptors

from pishovichok.state.schemas import CalcRDKitMetrics, ToolConfig

_SASCORER_MODULE: Any = None
_SASCORER_LOAD_ATTEMPTED: bool = False

# Кэш каталогов подструктурных фильтров RDKit C++ и калькуляторов дескрипторов
_CATALOG_BY_NAME: dict[str, FilterCatalog.FilterCatalog] = {}
_DESCRIPTOR_CALC_CACHE: dict[
    tuple[str, ...], MoleculeDescriptors.MolecularDescriptorCalculator
] = {}
_KNOWN_DESCRIPTOR_NAMES: frozenset[str] = frozenset(
    name for name, _fn in Descriptors._descList
)

_CATALOG_ENUM_MAP: dict[str, Any] = {
    "PAINS_A": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
    "PAINS_B": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
    "PAINS_C": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    "Brenk": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    "Glaxo": FilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo,
    "SureChEMBL": FilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL_SureChEMBL,
    "NIH": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
    "ZINC": FilterCatalog.FilterCatalogParams.FilterCatalogs.ZINC,
}

_BRICS_DUMMY_REGEX = re.compile(r"\[\d+\*\]|\[\*\]")


def _get_sascorer() -> Any:
    """Лениво загружает и кэширует модуль `sascorer` из `RDConfig.RDContribDir / 'SA_Score'`."""
    global _SASCORER_MODULE, _SASCORER_LOAD_ATTEMPTED
    if _SASCORER_LOAD_ATTEMPTED:
        return _SASCORER_MODULE

    _SASCORER_LOAD_ATTEMPTED = True
    try:
        sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
        if sa_dir.is_dir():
            sa_dir_str = str(sa_dir)
            if sa_dir_str not in sys.path:
                sys.path.append(sa_dir_str)
            import sascorer  # type: ignore[import-not-found]

            _SASCORER_MODULE = sascorer
    except Exception:  # noqa: BLE001
        _SASCORER_MODULE = None

    return _SASCORER_MODULE


def _get_filter_catalog(catalog_name: str) -> FilterCatalog.FilterCatalog | None:
    """Лениво инициализирует и кэширует каталог FilterCatalog по его имени из ToolConfig."""
    cached = _CATALOG_BY_NAME.get(catalog_name)
    if cached is not None:
        return cached
    enum_val = _CATALOG_ENUM_MAP.get(catalog_name)
    if enum_val is None:
        return None
    with contextlib.suppress(Exception):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(enum_val)
        cat = FilterCatalog.FilterCatalog(params)
        _CATALOG_BY_NAME[catalog_name] = cat
        return cat
    return None


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Вычисляет индекс синтетической доступности Ertl (`SAscore` в диапазоне `[1.0, 10.0]`).

    Использует `sascorer.calculateScore(mol)` из `RDContribDir/SA_Score`, а при его отсутствии —
    калиброванную аналитическую формулу топологической сложности (`BertzCT`, стереоцентры,
    спиро- и мостиковые атомы).
    """
    scorer = _get_sascorer()
    if scorer is not None:
        with contextlib.suppress(Exception):
            raw_score = float(scorer.calculateScore(mol))
            return round(max(1.0, min(10.0, raw_score)), 4)

    # Детерминированный аналитический фолбэк (Раздел 3.5 ARCHITECTURE.md)
    bertz = float(GraphDescriptors.BertzCT(mol))
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    fallback_sa = (
        1.4
        + (bertz / 650.0)
        + 0.35 * chiral_centers
        + 0.50 * (spiro_atoms + bridgehead_atoms)
        + 0.08 * max(0, rings - 2)
    )
    return round(max(1.0, min(10.0, fallback_sa)), 4)


def check_structural_alerts_detailed(
    mol: Chem.Mol,
    *,
    tool_config: ToolConfig | None = None,
) -> tuple[int, list[str], int, list[str], int, int, list[str]]:
    """Проверяет молекулу по активным каталогам из ToolConfig и custom_ban_smarts.

    Возвращает кортеж:
        `(pains_count, pains_names, brenk_count, brenk_names, glaxo_count, extra_count, extra_names)`
    """
    cfg = tool_config or ToolConfig()
    active_cats = list(cfg.active_filter_catalogs)

    pains_names: list[str] = []
    brenk_names: list[str] = []
    glaxo_count = 0
    extra_names: list[str] = []

    for cat_name in active_cats:
        cat = _get_filter_catalog(cat_name)
        if cat is None:
            continue
        if cat_name in ("PAINS_A", "PAINS_B", "PAINS_C"):
            for entry in cat.GetMatches(mol):
                desc = str(entry.GetDescription())
                if desc not in pains_names:
                    pains_names.append(desc)
        elif cat_name == "Brenk":
            for entry in cat.GetMatches(mol):
                desc = str(entry.GetDescription())
                if cfg.brenk_ignore_long_chain and desc == "Aliphatic_long_chain":
                    continue
                if desc not in brenk_names:
                    brenk_names.append(desc)
        else:
            matches = cat.GetMatches(mol)
            if cat_name == "Glaxo":
                glaxo_count = len(matches)
            for entry in matches:
                desc = f"{cat_name}:{entry.GetDescription()}"
                if desc not in extra_names:
                    extra_names.append(desc)

    for ban_idx, smarts_str in enumerate(cfg.custom_ban_smarts, start=1):
        if not smarts_str:
            continue
        pat_mol = Chem.MolFromSmarts(smarts_str) or Chem.MolFromSmiles(smarts_str)
        if pat_mol is not None and mol.HasSubstructMatch(pat_mol):
            label = f"CUSTOM_{ban_idx}:{smarts_str}"
            if label not in extra_names:
                extra_names.append(label)

    return (
        len(pains_names),
        pains_names,
        len(brenk_names),
        brenk_names,
        glaxo_count,
        len(extra_names),
        extra_names,
    )


def check_structural_alerts(
    mol: Chem.Mol,
    *,
    tool_config: ToolConfig | None = None,
) -> tuple[int, list[str], int, list[str], int]:
    """Проверяет молекулу по медико-химическим каталогам из ToolConfig (совместимый 5-элементный контракт)."""
    p_cnt, p_names, b_cnt, b_names, g_cnt, _e_cnt, _e_names = (
        check_structural_alerts_detailed(mol, tool_config=tool_config)
    )
    return (p_cnt, p_names, b_cnt, b_names, g_cnt)


def _strip_dummy_and_isolated_h(mol: Chem.Mol) -> Chem.Mol:
    """Заменяет связанные фиктивные атомы `[#0]` на водород и удаляет несвязанные H/фиктивные атомы."""
    rw = Chem.RWMol(mol)
    atoms_to_remove: list[int] = []
    for atom in rw.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 0:
            if atom.GetDegree() == 0:
                atoms_to_remove.append(atom.GetIdx())
            else:
                atom.SetAtomicNum(1)
                atom.SetIsotope(0)
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
        elif atomic_num == 1 and atom.GetDegree() == 0:
            atoms_to_remove.append(atom.GetIdx())

    for idx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(idx)

    clean_mol = Chem.RemoveHs(rw.GetMol())
    Chem.SanitizeMol(clean_mol)
    return clean_mol


def _clean_brics_fragment_smiles(raw_frag_smiles: str) -> str | None:
    """Очищает фрагмент BRICS от изотопных меток точек разрыва (`[1*]`, `[16*]`) и канонизирует SMILES."""
    frag_mol = Chem.MolFromSmiles(raw_frag_smiles)
    if frag_mol is not None:
        with contextlib.suppress(Exception):
            clean_mol = _strip_dummy_and_isolated_h(frag_mol)
            if clean_mol.GetNumHeavyAtoms() > 0:
                return Chem.MolToSmiles(
                    clean_mol, canonical=True, isomericSmiles=True
                )

    # Текстовый фолбэк удаления фиктивных меток [n*] без удаления несвязанного H до очистки
    parser_params = Chem.SmilesParserParams()
    parser_params.removeHs = False
    stripped = _BRICS_DUMMY_REGEX.sub("[H]", raw_frag_smiles)
    fallback_mol = Chem.MolFromSmiles(stripped, parser_params)
    if fallback_mol is not None:
        with contextlib.suppress(Exception):
            clean_fb = _strip_dummy_and_isolated_h(fallback_mol)
            if clean_fb.GetNumHeavyAtoms() > 0:
                return Chem.MolToSmiles(clean_fb, canonical=True, isomericSmiles=True)

    return None


def decompose_brics_precursors(
    mol: Chem.Mol,
    *,
    tool_config: ToolConfig | None = None,
) -> tuple[list[str], int]:
    """Выполняет ретросинтетическую декомпозицию молекулы по алгоритму BRICS.

    Управляется параметрами `brics_min_fragment_size` и `brics_only_leaf_nodes` из `ToolConfig`.
    """
    cfg = tool_config or ToolConfig()
    canon_full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        bonds_broken = len(brics_bonds)
        if bonds_broken == 0:
            return ([canon_full], 0)

        raw_frags = set(
            BRICS.BRICSDecompose(
                mol,
                minFragmentSize=int(cfg.brics_min_fragment_size),
                keepNonLeafNodes=not bool(cfg.brics_only_leaf_nodes),
                returnMols=False,
            )
        )
    except Exception:  # noqa: BLE001
        return ([canon_full], 0)

    cleaned_set: set[str] = set()
    for raw_f in raw_frags:
        clean_smi = _clean_brics_fragment_smiles(str(raw_f))
        if clean_smi:
            cleaned_set.add(clean_smi)

    if not cleaned_set:
        return ([canon_full], 0)

    min_keep = min(2, int(cfg.brics_min_fragment_size))
    substantial = [
        smi
        for smi in cleaned_set
        if (m := Chem.MolFromSmiles(smi)) is not None and m.GetNumHeavyAtoms() >= min_keep
    ]
    final_list = substantial if substantial else list(cleaned_set)
    final_list.sort(
        key=lambda s: (
            -(Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else 0),
            s,
        )
    )
    return (final_list, bonds_broken)


def count_lipinski_violations(
    mw: float,
    clogp: float,
    hbd: int,
    hba: int,
    *,
    tool_config: ToolConfig | None = None,
) -> int:
    """Подсчитывает число нарушений правила пяти Липински (Ro5, от 0 до 4) по порогам ToolConfig."""
    cfg = tool_config or ToolConfig()
    violations = 0
    if mw > cfg.lipinski_mw_cutoff:
        violations += 1
    if clogp > cfg.lipinski_logp_cutoff:
        violations += 1
    if hbd > cfg.lipinski_hbd_cutoff:
        violations += 1
    if hba > cfg.lipinski_hba_cutoff:
        violations += 1
    return violations


def compute_extra_rdkit_descriptors(
    mol: Chem.Mol,
    descriptor_names: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Вычисляет запрошенные дополнительные дескрипторы RDKit через MolecularDescriptorCalculator."""
    valid_names = tuple(
        name for name in descriptor_names if name in _KNOWN_DESCRIPTOR_NAMES
    )
    if not valid_names:
        return {}
    calc = _DESCRIPTOR_CALC_CACHE.get(valid_names)
    if calc is None:
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(list(valid_names))
        _DESCRIPTOR_CALC_CACHE[valid_names] = calc
    raw_vals = calc.CalcDescriptors(mol)
    out: dict[str, float] = {}
    for name, val in zip(valid_names, raw_vals, strict=False):
        with contextlib.suppress(Exception):
            fval = float(val)
            if math.isfinite(fval):
                out[name] = round(fval, 4)
    return out


def calculate_rdkit_metrics(
    mol_or_smiles: Chem.Mol | str,
    parent_mol_or_smiles: Chem.Mol | str | None = None,
    *,
    tool_config: ToolConfig | None = None,
) -> CalcRDKitMetrics:
    """Вычисляет полный детерминированный профиль `CalcRDKitMetrics` (Категория 2 данных кандидата)."""
    cfg = tool_config or ToolConfig()
    mol = (
        Chem.MolFromSmiles(mol_or_smiles)
        if isinstance(mol_or_smiles, str)
        else mol_or_smiles
    )
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(
            f"Cannot compute CalcRDKitMetrics for invalid molecule: {mol_or_smiles}"
        )

    parent_mol: Chem.Mol | None = None
    if parent_mol_or_smiles is not None:
        parent_mol = (
            Chem.MolFromSmiles(parent_mol_or_smiles)
            if isinstance(parent_mol_or_smiles, str)
            else parent_mol_or_smiles
        )

    mw = round(float(Descriptors.ExactMolWt(mol)), 4)
    clogp = round(float(Descriptors.MolLogP(mol)), 4)
    tpsa = round(float(Descriptors.TPSA(mol)), 4)
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))
    rot_bonds = int(Descriptors.NumRotatableBonds(mol))
    fsp3 = round(max(0.0, min(1.0, float(Descriptors.FractionCSP3(mol)))), 4)
    qed_val = round(max(0.0, min(1.0, float(QED.qed(mol)))), 4)
    sa_score = calculate_sa_score(mol)
    lipinski = count_lipinski_violations(
        mw=mw, clogp=clogp, hbd=hbd, hba=hba, tool_config=cfg
    )

    (
        pains_count,
        pains_names,
        brenk_count,
        brenk_names,
        glaxo_count,
        extra_count,
        extra_names,
    ) = check_structural_alerts_detailed(mol, tool_config=cfg)
    brics_fragments, brics_bonds_broken = decompose_brics_precursors(
        mol, tool_config=cfg
    )
    extra_descriptors = compute_extra_rdkit_descriptors(
        mol, cfg.extra_rdkit_descriptors
    )

    if parent_mol is not None:
        parent_clogp = float(Descriptors.MolLogP(parent_mol))
        parent_tpsa = float(Descriptors.TPSA(parent_mol))
        delta_clogp = round(clogp - parent_clogp, 4)
        delta_tpsa = round(tpsa - parent_tpsa, 4)
    else:
        delta_clogp = 0.0
        delta_tpsa = 0.0

    return CalcRDKitMetrics(
        mw=mw,
        clogp=clogp,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rot_bonds,
        fsp3=fsp3,
        qed=qed_val,
        sa_score=sa_score,
        lipinski_violations=lipinski,
        pains_alerts_count=pains_count,
        pains_matched_names=pains_names,
        brenk_alerts_count=brenk_count,
        brenk_matched_names=brenk_names,
        glaxo_alerts_count=glaxo_count,
        extra_alerts_count=extra_count,
        extra_alerts_matched_names=extra_names,
        extra_descriptors=extra_descriptors,
        brics_fragments=brics_fragments,
        brics_bonds_broken=brics_bonds_broken,
        delta_clogp_vs_parent=delta_clogp,
        delta_tpsa_vs_parent=delta_tpsa,
    )


def evaluate_docking_batch(
    candidates: list[Any],
    target_id: str | None,
    top_k: int = 3,
    exhaustiveness: int = 4,
    custom_pdb_id: str | None = None,
    custom_box_center: list[float] | None = None,
    workspace_dir: str | Path | None = None,
    selected_mol_ids: list[str] | None = None,
    constraints: Any | None = None,
    tool_config: Any | None = None,
    receptors_dir: Path | str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Выполняет пакетный 3D-докинг топ-K кандидатов (или выбранных candidate_id) и пересчитывает MPO."""
    from pishovichok.core.docking import run_docking
    from pishovichok.core.scoring import calculate_harrington_mpo_score
    from pishovichok.state.schemas import DockingResult
    from pishovichok.state.storage import save_candidate_pose_sdf

    if not candidates:
        return list(candidates), {
            "status": "SKIPPED",
            "docked_count": 0,
            "success_count": 0,
            "pdb_id": custom_pdb_id,
            "source": None,
            "best_score_kcal": None,
            "message": "Список кандидатов пуст — докинг не выполнялся.",
        }

    if selected_mol_ids:
        selected_set = {str(cid) for cid in selected_mol_ids}
        target_indices = [
            idx
            for idx, cand in enumerate(candidates)
            if str(getattr(cand, "candidate_id", "")) in selected_set
        ]
    else:
        limit = max(1, int(top_k))
        target_indices = list(range(min(limit, len(candidates))))

    updated_list = list(candidates)
    success_count = 0
    best_score: float | None = None
    last_pdb_id: str | None = custom_pdb_id
    last_source: str | None = None
    last_message: str | None = None
    overall_status = "SKIPPED"

    for pos, idx in enumerate(target_indices):
        cand = updated_list[idx]
        dock_res = run_docking(
            ligand_smiles=cand.canonical_smiles,
            target_id=target_id,
            exhaustiveness=exhaustiveness,
            custom_pdb_id=custom_pdb_id,
            custom_box_center=custom_box_center,
            receptors_dir=receptors_dir,
        )

        pose_path = dock_res.get("pose_sdf_path")
        pose_block = dock_res.get("pose_sdf_block")
        if workspace_dir is not None and pose_block:
            pose_path = save_candidate_pose_sdf(
                workspace_dir=workspace_dir,
                mol_id=cand.candidate_id,
                sdf_content=pose_block,
            )

        res_status = str(dock_res.get("status", "FAILED"))
        if res_status not in {"PENDING", "SUCCESS", "SKIPPED", "FAILED"}:
            res_status = "FAILED"

        docking_model = DockingResult(
            status=res_status,  # type: ignore[arg-type]
            pdb_id=dock_res.get("pdb_id"),
            source=dock_res.get("source"),
            docking_score_kcal=dock_res.get("docking_score_kcal"),
            backend=dock_res.get("backend"),
            interactions=dock_res.get("interactions") or {},
            pose_sdf_path=pose_path,
            pose_sdf_block=pose_block,
            receptor_pdb_path=dock_res.get("receptor_pdb_path"),
            box_center=dock_res.get("box_center"),
            box_size=dock_res.get("box_size"),
            key_residues=list(dock_res.get("key_residues") or []),
            message=dock_res.get("message"),
        )

        new_mpo = cand.mpo_harrington_score
        if (
            res_status == "SUCCESS"
            and docking_model.docking_score_kcal is not None
            and constraints is not None
        ):
            has_alerts = (
                cand.calc_rdkit.pains_alerts_count > 0
                or cand.calc_rdkit.brenk_alerts_count > 0
                or cand.calc_rdkit.extra_alerts_count > 0
            )
            if cand.alert_penalty_mode == "veto" and has_alerts:
                new_mpo = 0.0
            else:
                new_mpo = calculate_harrington_mpo_score(
                    calc_rdkit=cand.calc_rdkit,
                    pred_admet=cand.pred_admet,
                    parent_exp_pic50=cand.parent_exp_pic50,
                    constraints=constraints,
                    docking_score_kcal=docking_model.docking_score_kcal,
                    tool_config=tool_config,
                )

        updated_list[idx] = cand.model_copy(
            update={"docking": docking_model, "mpo_harrington_score": new_mpo}
        )

        last_pdb_id = docking_model.pdb_id or last_pdb_id
        last_source = docking_model.source or last_source
        last_message = docking_model.message or last_message

        if res_status == "SUCCESS":
            success_count += 1
            overall_status = "SUCCESS"
            dg = docking_model.docking_score_kcal
            if dg is not None and (best_score is None or dg < best_score):
                best_score = dg
        elif res_status == "SKIPPED" and pos == 0:
            # Если структура рецептора отсутствует вообще, пропускаем остальные без повторных сетевых запросов
            for rem_idx in target_indices[1:]:
                rem_cand = updated_list[rem_idx]
                updated_list[rem_idx] = rem_cand.model_copy(
                    update={"docking": docking_model}
                )
            overall_status = "SKIPPED"
            break
        elif overall_status != "SUCCESS":
            overall_status = res_status

    return updated_list, {
        "status": overall_status,
        "docked_count": len(target_indices),
        "success_count": success_count,
        "pdb_id": last_pdb_id,
        "source": last_source,
        "best_score_kcal": best_score,
        "message": last_message,
    }
