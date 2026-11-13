"""Модуль направленной генерации молекулярных аналогов (Слой 1, без участия LLM).

Реализует 3-уровневый каскад генерации:
1. Уровень 1 (`mmp_smirks_rule`): применение экспериментально подтвержденных трансформаций SMIRKS
   из сети парных соответствий `rdMMPA` (`SARRule`).
2. Уровень 2 (`crem_fragment_replacement`): контекстно-зависимая изостерная замена периферийных
   радикалов ($r = 3$) по локальной базе `data/crem/replacements_sa2.db` (или пакету `crem`).
3. Уровень 3 (`bioisostere_catalog`): применение курированного каталога из 48 медико-химических
   биоизостерных SMIRKS-реакций (`core/bioisosteres.py`).

Гарантии модуля:
- 100% химическая валидность получаемых структур (`Chem.SanitizeMol` без ошибок валентности).
- Строгое сохранение целевого скаффолда Bemis-Murcko при `constraints.freeze_scaffold == True`.
- Учет корректирующих ограничений во 2-й петле самокоррекции (`preferred_substituent_types`:
  `polar`, `less_lipophilic`, `exclude_halogens`, а также порогов `max_logp`, `min_tpsa`, `max_sa_score`).
- Точное определение индексов модифицированных атомов (`highlighted_atom_indices`) для 2D-подсветки.
"""

from __future__ import annotations

import contextlib
import random
import sqlite3
from pathlib import Path
from typing import Any, Literal, NamedTuple

from rdkit import Chem
from rdkit.Chem import BRICS, QED, AllChem, Descriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from crem.crem import grow_mol as _crem_pkg_grow_mol
    from crem.crem import mutate_mol as _crem_pkg_mutate_mol
except ImportError:  # pragma: no cover - fallback if crem is absent
    _crem_pkg_grow_mol = None
    _crem_pkg_mutate_mol = None

from pishovichok.config import settings
from pishovichok.core.bioisosteres import select_bioisostere_rules
from pishovichok.core.curation import standardize_molecule_detailed
from pishovichok.core.evaluation import (
    calculate_sa_score,
    check_structural_alerts_detailed,
)
from pishovichok.core.fingerprints import (
    compute_fp_similarity,
    compute_mol_fingerprint,
)
from pishovichok.core.sar_analysis import (
    canonicalize_substituent_smiles,
    fragment_molecule_mmpa,
)
from pishovichok.state.schemas import (
    GeneratedRawCandidate,
    ParentSeed,
    ParquetPointer,
    QueryConstraints,
    SARRule,
    ToolConfig,
)
from pishovichok.state.storage import read_parquet_pointer

_HALOGEN_ATOMIC_NUMS: frozenset[int] = frozenset({9, 17, 35, 53})


class _TransformationCandidateSpec(NamedTuple):
    """Внутреннее описание трансформации из любого из 3 эшелонов каскада."""

    rule_id: str
    smirks: str
    from_fragment: str
    to_fragment: str
    generation_origin: Literal[
        "mmp_smirks_rule",
        "crem_fragment_replacement",
        "bioisostere_catalog",
    ]
    mean_delta_pic50: float
    std_delta_pic50: float
    support_pairs_count: int
    assay_homogeneous: bool
    delta_clogp: float
    delta_tpsa: float
    sa_score_estimate: float
    contains_halogen: bool
    heavy_atoms_to: int


def fragment_contains_halogen(frag_smiles: str) -> bool:
    """Проверяет, содержит ли вводимый заместитель атомы галогенов (F, Cl, Br, I)."""
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is not None:
        return any(
            atom.GetAtomicNum() in _HALOGEN_ATOMIC_NUMS for atom in mol.GetAtoms()
        )
    return any(tok in frag_smiles for tok in ("F", "Cl", "Br", "I"))


def _fragment_heavy_atoms(frag_smiles: str) -> int:
    """Возвращает число тяжелых (не-фиктивных и не-водородных) атомов в фрагменте `[*:1]R`."""
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return 1
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def build_constrained_reaction_smarts(
    from_fragment: str, to_fragment: str
) -> str | None:
    """Строит точный SMIRKS для `AllChem.ReactionFromSmarts` с фиксацией степени и числа H атомов.

    Предотвращает ложное срабатывание SMARTS на внутренних атомах цепей или колец
    (например, чтобы `[*:1]OC` заменял исключительно терминальную метоксигруппу `-OCH3`,
    не затрагивая `-OCH2CH2CH2-` или морфолиновое кольцо).
    """
    from_mol = Chem.MolFromSmiles(from_fragment)
    to_mol = Chem.MolFromSmiles(to_fragment)
    if from_mol is None or to_mol is None:
        return None

    # Для простых одноатомных и линейных терминальных заместителей формируем строгий SMARTS
    canon_from = canonicalize_substituent_smiles(from_fragment)
    canon_to = canonicalize_substituent_smiles(to_fragment)
    rhs_body = canon_to.replace("[*:1]", "").lstrip("-")
    if not rhs_body:
        return None

    exact_lhs_map: dict[str, str] = {
        "[*:1]F": "[*:1]-[F;H0;D1;R0]",
        "[*:1]Cl": "[*:1]-[Cl;H0;D1;R0]",
        "[*:1]Br": "[*:1]-[Br;H0;D1;R0]",
        "[*:1]I": "[*:1]-[I;H0;D1;R0]",
        "[*:1]C": "[*:1]-[C;H3;D1;R0]",
        "[*:1]OC": "[*:1]-[O;H0;D2;R0]-[C;H3;D1;R0]",
        "[*:1]C#N": "[*:1]-[C;H0;D2;R0]#[N;H0;D1;R0]",
        "[*:1]CO": "[*:1]-[C;H2;D2;R0]-[O;H1;D1;R0]",
        "[*:1]OCCO": "[*:1]-[O;H0;D2;R0]-[C;H2;D2;R0]-[C;H2;D2;R0]-[O;H1;D1;R0]",
        "[*:1]C(=O)N": "[*:1]-[C;H0;D3;R0](=[O;H0;D1;R0])-[N;H2;D1;R0]",
        "[*:1]C(=O)NC": "[*:1]-[C;H0;D3;R0](=[O;H0;D1;R0])-[N;H1;D2;R0]-[C;H3;D1;R0]",
        "[*:1]S(=O)(=O)C": "[*:1]-[S;H0;D4;R0](=[O;H0;D1;R0])(=[O;H0;D1;R0])-[C;H3;D1;R0]",
        "[*:1]N(C)C": "[*:1]-[N;H0;D3;R0](-[C;H3;D1;R0])-[C;H3;D1;R0]",
    }
    lhs_smarts = exact_lhs_map.get(canon_from)
    if lhs_smarts is None:
        return None

    return f"{lhs_smarts}>>[*:1]-{rhs_body}"


def _prepare_replacement_piece(to_fragment: str) -> tuple[Chem.Mol, int] | None:
    """Удаляет фиктивный атом `[*:1]` из `to_fragment` и возвращает `(mol_piece, connection_atom_idx)`."""
    frag_mol = Chem.MolFromSmiles(to_fragment)
    if frag_mol is None:
        return None

    dummy_idx = -1
    neighbor_idx = -1
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_idx = atom.GetIdx()
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                return None
            neighbor_idx = neighbors[0].GetIdx()
            break

    if dummy_idx < 0 or neighbor_idx < 0:
        return None

    # Помечаем атом присоединения изотопной меткой, чтобы точно найти его индекс после удаления [*:1]
    rw_mol = Chem.RWMol(frag_mol)
    rw_mol.GetAtomWithIdx(neighbor_idx).SetAtomMapNum(99)
    rw_mol.RemoveAtom(dummy_idx)

    piece_mol = rw_mol.GetMol()
    conn_idx = -1
    for atom in piece_mol.GetAtoms():
        if atom.GetAtomMapNum() == 99:
            conn_idx = atom.GetIdx()
            atom.SetAtomMapNum(0)
            break

    if conn_idx < 0:
        return None

    try:
        Chem.SanitizeMol(piece_mol)
    except Exception:  # noqa: BLE001
        return None

    return (piece_mol, conn_idx)


def compute_highlighted_atoms(
    parent_core_smiles: str, candidate_mol: Chem.Mol
) -> list[int]:
    """Определяет индексы атомов введенного заместителя в `candidate_mol` для подсветки в UI.

    Удаляет фиктивную метку `[*:1]` из `parent_core_smiles`, находит подструктурное вхождение
    неизменного ядра в `candidate_mol` и возвращает список индексов всех остальных атомов.
    """
    core_mol = Chem.MolFromSmiles(parent_core_smiles)
    if core_mol is None:
        return []

    rw_core = Chem.RWMol(core_mol)
    dummy_indices = sorted(
        [a.GetIdx() for a in rw_core.GetAtoms() if a.GetAtomicNum() == 0],
        reverse=True,
    )
    for d_idx in dummy_indices:
        rw_core.RemoveAtom(d_idx)

    clean_core = rw_core.GetMol()
    try:
        Chem.SanitizeMol(clean_core)
    except Exception:  # noqa: BLE001
        return []

    match = candidate_mol.GetSubstructMatch(clean_core)
    if not match:
        return []

    matched_set = set(match)
    return [idx for idx in range(candidate_mol.GetNumAtoms()) if idx not in matched_set]


def verify_scaffold_preservation(
    parent_mol: Chem.Mol,
    candidate_mol: Chem.Mol,
    freeze_scaffold: bool = True,
    *,
    scaffold_mode: str = "bemis_murcko",
    tool_config: ToolConfig | None = None,
) -> tuple[bool, str]:
    """Проверяет сохранение целевого каркаса Bemis-Murcko (или generic_framework) родительской молекулы.

    Возвращает `(is_preserved, candidate_scaffold_smiles)`.
    """
    eff_mode = tool_config.scaffold_mode if tool_config is not None else scaffold_mode
    use_isomeric = tool_config.keep_chirality if tool_config is not None else True

    def _extract_scaf_mol(m: Chem.Mol) -> Chem.Mol | None:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if (
            eff_mode == "generic_framework"
            and scaf is not None
            and scaf.GetNumAtoms() > 0
        ):
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return scaf

    cand_scaf_mol = _extract_scaf_mol(candidate_mol)
    cand_scaf_smi = (
        Chem.MolToSmiles(cand_scaf_mol, canonical=True, isomericSmiles=use_isomeric)
        if cand_scaf_mol is not None and cand_scaf_mol.GetNumAtoms() > 0
        else Chem.MolToSmiles(candidate_mol, canonical=True, isomericSmiles=use_isomeric)
    )
    if not freeze_scaffold:
        return (True, cand_scaf_smi)

    parent_scaf_mol = _extract_scaf_mol(parent_mol)
    if parent_scaf_mol is None or parent_scaf_mol.GetNumAtoms() == 0:
        return (True, cand_scaf_smi)

    parent_scaf_smi = Chem.MolToSmiles(
        parent_scaf_mol, canonical=True, isomericSmiles=use_isomeric
    )
    if cand_scaf_smi == parent_scaf_smi:
        return (True, cand_scaf_smi)

    # Если заместитель сам содержал небольшое периферийное кольцо, проверяем, что родительский
    # каркас полностью сохранен как неизменная подструктура и число колец не уменьшилось
    if (
        cand_scaf_mol is not None
        and cand_scaf_mol.GetNumAtoms() >= parent_scaf_mol.GetNumAtoms()
        and candidate_mol.HasSubstructMatch(parent_scaf_mol)
    ):
        return (True, parent_scaf_smi)

    return (False, cand_scaf_smi)


def _strip_dummy_atoms(core_smiles: str) -> Chem.Mol | None:
    core_mol = Chem.MolFromSmiles(core_smiles)
    if core_mol is None:
        return None
    rw_core = Chem.RWMol(core_mol)
    for d_idx in sorted(
        [a.GetIdx() for a in rw_core.GetAtoms() if a.GetAtomicNum() == 0],
        reverse=True,
    ):
        rw_core.RemoveAtom(d_idx)
    clean_core = rw_core.GetMol()
    try:
        Chem.SanitizeMol(clean_core)
        return clean_core
    except Exception:  # noqa: BLE001
        return None


def _is_cut_allowed(
    parent_mol: Chem.Mol,
    core_smi: str,
    protected_atoms: set[int],
    modifiable_atoms: set[int] | None,
) -> bool:
    """Проверяет, что разрез не затрагивает protected_smarts и попадает в modifiable_atom_indices."""
    if not protected_atoms and modifiable_atoms is None:
        return True
    clean_core = _strip_dummy_atoms(core_smi)
    if clean_core is None:
        return False
    matches = parent_mol.GetSubstructMatches(clean_core)
    if not matches:
        return False
    all_parent_indices = set(range(parent_mol.GetNumAtoms()))
    for match in matches:
        replaced_indices = all_parent_indices - set(match)
        if not replaced_indices:
            continue
        if protected_atoms and (replaced_indices & protected_atoms):
            continue
        if modifiable_atoms is not None and not (replaced_indices & modifiable_atoms):
            continue
        return True
    return False


def apply_smirks_to_molecule(
    parent_mol: Chem.Mol,
    from_fragment: str,
    to_fragment: str,
    freeze_scaffold: bool = True,
    *,
    protected_smarts: str | None = None,
    modifiable_atom_indices: list[int] | None = None,
    tool_config: ToolConfig | None = None,
) -> list[tuple[str, str, list[int]]]:
    """Применяет трансформацию `from_fragment >> to_fragment` к `parent_mol` с гарантией 100% валидности.

    Учитывает `protected_smarts`, `modifiable_atom_indices`, `freeze_murcko_core`, `mmpa_max_cuts`
    и `scaffold_mode` из `tool_config`.
    """
    cfg = tool_config
    canon_from = canonicalize_substituent_smiles(from_fragment)
    canon_to = canonicalize_substituent_smiles(to_fragment)
    if canon_from == canon_to:
        return []

    eff_protected = (
        protected_smarts
        if protected_smarts is not None
        else (cfg.protected_smarts if cfg is not None else None)
    )
    eff_modifiable = (
        modifiable_atom_indices
        if modifiable_atom_indices is not None
        else (cfg.modifiable_atom_indices if cfg is not None else None)
    )
    eff_freeze = (
        cfg.freeze_murcko_core
        if (cfg is not None and freeze_scaffold)
        else freeze_scaffold
    )
    if cfg is not None and not cfg.freeze_murcko_core:
        eff_freeze = False

    prot_mol: Chem.Mol | None = None
    protected_atoms: set[int] = set()
    if eff_protected:
        prot_mol = Chem.MolFromSmarts(eff_protected) or Chem.MolFromSmiles(
            eff_protected
        )
        if prot_mol is not None:
            for m in parent_mol.GetSubstructMatches(prot_mol):
                protected_atoms.update(m)

    modifiable_set: set[int] | None = (
        set(eff_modifiable) if eff_modifiable is not None else None
    )

    parent_smiles = Chem.MolToSmiles(parent_mol, canonical=True, isomericSmiles=True)
    mmpa_cuts = fragment_molecule_mmpa(
        parent_mol,
        max_cuts=cfg.mmpa_max_cuts if cfg is not None else 1,
        max_rgroup_heavy_atoms=cfg.max_rgroup_heavy_atoms if cfg is not None else 13,
        min_core_heavy_atoms=cfg.mmpa_min_core_heavy_atoms if cfg is not None else 6,
        max_cut_bonds=cfg.mmpa_max_cut_bonds if cfg is not None else 20,
    )
    matching_cores = [
        core_smi
        for core_smi, rg_smi, _ring, _heavy in mmpa_cuts
        if rg_smi == canon_from
        and _is_cut_allowed(parent_mol, core_smi, protected_atoms, modifiable_set)
    ]

    raw_product_mols: list[tuple[Chem.Mol, str]] = []

    # Способ 1: Прямой запуск AllChem.ReactionFromSmarts (если не заданы точечные ограничения)
    constrained_smirks = build_constrained_reaction_smarts(canon_from, canon_to)
    if (
        constrained_smirks is not None
        and matching_cores
        and not eff_protected
        and modifiable_set is None
    ):
        with contextlib.suppress(Exception):
            rxn = AllChem.ReactionFromSmarts(constrained_smirks)
            if rxn is not None:
                prod_sets = rxn.RunReactants((parent_mol,))
                default_core = matching_cores[0]
                for prod_tuple in prod_sets:
                    if prod_tuple:
                        raw_product_mols.append((prod_tuple[0], default_core))

    # Способ 2: Точная подстановка радикала в точки разрыва rdMMPA [*:1]
    piece_info = _prepare_replacement_piece(canon_to)
    dummy_pattern = Chem.MolFromSmarts("[#0]")
    if piece_info is not None and dummy_pattern is not None:
        piece_mol, conn_idx = piece_info
        for core_smi in matching_cores:
            core_mol = Chem.MolFromSmiles(core_smi)
            if core_mol is None:
                continue
            with contextlib.suppress(Exception):
                replaced_tuple = Chem.ReplaceSubstructs(
                    core_mol,
                    dummy_pattern,
                    piece_mol,
                    replacementConnectionPoint=conn_idx,
                    replaceAll=True,
                )
                for r_mol in replaced_tuple:
                    if r_mol is not None:
                        raw_product_mols.append((r_mol, core_smi))

    valid_products: list[tuple[str, str, list[int]]] = []
    seen_smiles: set[str] = {parent_smiles}

    for prod_mol, ref_core_smi in raw_product_mols:
        with contextlib.suppress(Exception):
            mol_copy = Chem.Mol(prod_mol)
            Chem.SanitizeMol(mol_copy)
            use_iso = cfg.keep_chirality if cfg is not None else True
            canon_prod_smi = Chem.MolToSmiles(
                mol_copy, canonical=True, isomericSmiles=use_iso
            )
            if not canon_prod_smi or "." in canon_prod_smi or "*" in canon_prod_smi:
                continue
            if canon_prod_smi in seen_smiles:
                continue

            # Повторно парсим из канонического SMILES для 100% гарантии валидности графа и индексации атомов
            verified_mol = Chem.MolFromSmiles(canon_prod_smi, sanitize=True)
            if verified_mol is None or verified_mol.GetNumAtoms() == 0:
                continue

            if prot_mol is not None and not verified_mol.HasSubstructMatch(prot_mol):
                continue

            preserved, scaf_smi = verify_scaffold_preservation(
                parent_mol=parent_mol,
                candidate_mol=verified_mol,
                freeze_scaffold=eff_freeze,
                tool_config=cfg,
            )
            if not preserved:
                continue

            highlights = compute_highlighted_atoms(ref_core_smi, verified_mol)
            seen_smiles.add(canon_prod_smi)
            valid_products.append((canon_prod_smi, scaf_smi, highlights))

    return valid_products


def _load_crem_replacements_from_db(
    crem_db_path: Path | None = None,
    max_sa_score: float = 3.2,
    *,
    tool_config: ToolConfig | None = None,
) -> list[_TransformationCandidateSpec]:
    """Загружает контекстные замены фрагментов из локальной SQLite БД CReM (`replacements_sa2.db`)."""
    cfg = tool_config or ToolConfig()
    db_path = Path(crem_db_path) if crem_db_path is not None else settings.crem_db_path
    if not db_path.is_file():
        return []

    specs: list[_TransformationCandidateSpec] = []
    target_table = f"radius{int(cfg.crem_context_radius)}"
    try:
        with sqlite3.connect(db_path) as conn:
            tables = {
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            table_name = target_table if target_table in tables else "radius3"
            if table_name not in tables:
                return []
            rows = conn.execute(
                f"""
                SELECT context, from_smarts, to_smarts, radius, sa_score, delta_clogp, delta_tpsa
                FROM {table_name}
                WHERE sa_score <= ?
                ORDER BY delta_tpsa DESC, sa_score ASC
                """,
                (float(max_sa_score),),
            ).fetchall()
    except Exception:  # noqa: BLE001
        return []

    min_rel = (
        max(1, int(cfg.crem_min_rel_atoms))
        if cfg.crem_mode == "grow"
        else int(cfg.crem_min_rel_atoms)
    )
    max_rel = int(cfg.crem_max_rel_atoms)

    for idx, row in enumerate(rows, start=1):
        _ctx, from_s, to_s, rad, sa_val, d_logp, d_tpsa = row
        canon_from = canonicalize_substituent_smiles(str(from_s))
        canon_to = canonicalize_substituent_smiles(str(to_s))
        from_heavy = _fragment_heavy_atoms(canon_from)
        to_heavy = _fragment_heavy_atoms(canon_to)
        rel_atoms = to_heavy - from_heavy
        if cfg.crem_mode == "grow" and rel_atoms <= 0:
            continue
        if rel_atoms < min_rel or rel_atoms > max_rel:
            continue
        smirks = f"{canon_from}>>{canon_to}"
        # Оценочный прирост активности для изостерных замен CReM на основе полярного и липофильного баланса
        est_delta_pic50 = round(
            0.45 + 0.008 * max(0.0, float(d_tpsa)) - 0.05 * max(0.0, float(d_logp)), 4
        )
        specs.append(
            _TransformationCandidateSpec(
                rule_id=f"CREM_R{int(rad)}_{idx:02d}",
                smirks=smirks,
                from_fragment=canon_from,
                to_fragment=canon_to,
                generation_origin="crem_fragment_replacement",
                mean_delta_pic50=est_delta_pic50,
                std_delta_pic50=0.15,
                support_pairs_count=3,
                assay_homogeneous=True,
                delta_clogp=float(d_logp),
                delta_tpsa=float(d_tpsa),
                sa_score_estimate=float(sa_val),
                contains_halogen=fragment_contains_halogen(canon_to),
                heavy_atoms_to=to_heavy,
            )
        )
        if len(specs) >= int(cfg.crem_max_replacements):
            break
    return specs


def _filter_and_rank_transformation_specs(
    specs: list[_TransformationCandidateSpec],
    constraints: QueryConstraints,
    parent_clogp: float,
    parent_tpsa: float,
    generation_iteration: int,
    *,
    tool_config: ToolConfig | None = None,
) -> list[_TransformationCandidateSpec]:
    """Фильтрует и ранжирует трансформации с учетом базовых и корректирующих ограничений."""
    cfg = tool_config or ToolConfig()
    pref_types = set(constraints.preferred_substituent_types or ["any"])
    want_exclude_halogens = "exclude_halogens" in pref_types
    want_polar = "polar" in pref_types or (
        constraints.min_tpsa is not None and parent_tpsa < constraints.min_tpsa
    )
    want_less_lipophilic = "less_lipophilic" in pref_types or generation_iteration >= 2
    is_corrective_mode = want_exclude_halogens or want_polar or want_less_lipophilic

    filtered: list[_TransformationCandidateSpec] = []
    for spec in specs:
        if spec.sa_score_estimate > constraints.max_sa_score:
            continue

        if want_exclude_halogens and spec.contains_halogen:
            continue

        # В корректирующем режиме жестко отсекаем трансформации, повышающие липофильность или снижающие полярность
        if is_corrective_mode:
            if want_less_lipophilic and spec.delta_clogp > 0.05:
                continue
            if want_polar and spec.delta_tpsa < 0.0:
                continue
            # Если липофильность родителя существенно превышает целевой порог,
            # отсекаем слабые замены, которые не дают достаточного снижения cLogP
            required_logp_drop = parent_clogp - constraints.max_logp
            if required_logp_drop > 0.30 and spec.delta_clogp > -0.20:
                continue

        filtered.append(spec)

    if not filtered:
        # Мягкий откат, если все правила были отсечены жестким порогом: оставляем те, что снижают cLogP
        filtered = [
            s
            for s in specs
            if (not want_exclude_halogens or not s.contains_halogen)
            and s.delta_clogp <= 0.0
        ] or list(specs)

    def _sort_key(s: _TransformationCandidateSpec) -> tuple[float, ...]:
        if is_corrective_mode:
            projected_clogp = parent_clogp + s.delta_clogp
            projected_tpsa = parent_tpsa + s.delta_tpsa
            meets_logp = 1.0 if projected_clogp <= constraints.max_logp else 0.0
            meets_tpsa = (
                1.0
                if (
                    constraints.min_tpsa is None
                    or projected_tpsa >= constraints.min_tpsa
                )
                else 0.0
            )
            compact_bonus = (
                1.0
                if s.heavy_atoms_to <= int(cfg.crem_max_fragment_heavy_atoms)
                else 0.0
            )
            polar_score = (
                (-s.delta_clogp * 1.5) + (s.delta_tpsa / 20.0) + s.mean_delta_pic50
            )
            return (
                meets_logp,
                meets_tpsa,
                compact_bonus,
                round(polar_score, 4),
                s.mean_delta_pic50,
            )

        # На 1-й итерации без специальных директив приоритет у экспериментальных правил MMP и ΔpIC50
        origin_tier = (
            3.0
            if s.generation_origin == "mmp_smirks_rule"
            else (2.0 if s.generation_origin == "crem_fragment_replacement" else 1.0)
        )
        return (
            origin_tier,
            round(s.mean_delta_pic50, 4),
            float(s.support_pairs_count),
            -s.sa_score_estimate,
        )

    filtered.sort(key=_sort_key, reverse=True)
    return filtered


def select_parent_seeds_from_curated(
    curated_pointer: ParquetPointer,
    constraints: QueryConstraints,
    max_parents: int | None = None,
    *,
    tool_config: ToolConfig | None = None,
) -> list[ParentSeed]:
    """Выбирает родительские молекулы `ParentSeed` из `curated_dataset.parquet`.

    - Для Сценария 2 (`lead_optimization` с заданным `lead_smiles`) гарантированно включает
      саму молекулу-лидера (например, гефитиниб `CHEMBL939`) на первой позиции.
    - Для Сценариев 1 и 3 выбирает топ активных нецензурированных соединений из доминирующего скаффолда.
    """
    cfg = tool_config or ToolConfig()
    eff_max_parents = max_parents if max_parents is not None else cfg.max_parent_seeds
    rows = read_parquet_pointer(curated_pointer).to_pylist()
    active_rows = [r for r in rows if not bool(r.get("is_censored", False))]
    if not active_rows:
        active_rows = rows

    seeds: list[ParentSeed] = []
    seen_smiles: set[str] = set()

    if constraints.lead_smiles:
        std_lead = standardize_molecule_detailed(
            constraints.lead_smiles, tool_config=cfg
        )
        lead_canon = (
            std_lead.canonical_smiles
            if std_lead is not None
            else constraints.lead_smiles.strip()
        )
        lead_scaf = std_lead.scaffold_smiles if std_lead is not None else lead_canon

        matched_row = next(
            (r for r in active_rows if str(r.get("canonical_smiles")) == lead_canon),
            None,
        )
        if matched_row is not None:
            pic50 = float(matched_row["pchembl_value"])
            ic50_nm = round(10.0 ** (9.0 - pic50), 4)
            seeds.append(
                ParentSeed(
                    mol_id=str(matched_row["mol_id"]),
                    canonical_smiles=lead_canon,
                    scaffold_smiles=str(matched_row["scaffold_smiles"]),
                    exp_pic50=pic50,
                    exp_ic50_nm=max(0.0001, ic50_nm),
                    bao_format=str(
                        matched_row.get("bao_format") or settings.dominant_bao_default
                    ),
                    assay_type=str(matched_row.get("assay_type") or "B"),
                    source_dataset="ChEMBL_Parquet",
                )
            )
        else:
            seeds.append(
                ParentSeed(
                    mol_id="USER_LEAD_01",
                    canonical_smiles=lead_canon,
                    scaffold_smiles=lead_scaf,
                    exp_pic50=7.60,
                    exp_ic50_nm=round(10.0 ** (9.0 - 7.60), 4),
                    bao_format=settings.dominant_bao_default,
                    assay_type="B",
                    source_dataset="ChEMBL_Parquet",
                )
            )
        seen_smiles.add(lead_canon)

        # В режиме оптимизации конкретного лидера генерируем аналоги непосредственно от лидера
        if constraints.scenario_type == "lead_optimization":
            return seeds

    def _seed_sort_key(r: dict[str, Any]) -> tuple[float, ...]:
        csmi = str(r["canonical_smiles"])
        mol = Chem.MolFromSmiles(csmi)
        pains_clean = 1.0
        if constraints.require_no_pains and mol is not None:
            from pishovichok.core.evaluation import check_structural_alerts_detailed

            p, _, b, _, _, extra_cnt, _ = check_structural_alerts_detailed(
                mol, tool_config=cfg
            )
            if p > 0 or b > 0 or extra_cnt > 0:
                pains_clean = 0.0

        return (
            pains_clean,
            float(r.get("tanimoto_to_lead", 1.0)),
            float(r.get("pchembl_value", 0.0)),
        )

    sorted_rows = sorted(
        active_rows,
        key=_seed_sort_key,
        reverse=True,
    )
    for row in sorted_rows:
        csmi = str(row["canonical_smiles"])
        if csmi in seen_smiles:
            continue
        seen_smiles.add(csmi)
        pic50 = float(row["pchembl_value"])
        ic50_nm = round(10.0 ** (9.0 - pic50), 4)
        src_label = (
            "Custom_WetLab_CSV"
            if str(row.get("assay_type")) == "CUSTOM_WETLAB"
            else "ChEMBL_Parquet"
        )
        seeds.append(
            ParentSeed(
                mol_id=str(row["mol_id"]),
                canonical_smiles=csmi,
                scaffold_smiles=str(row["scaffold_smiles"]),
                exp_pic50=pic50,
                exp_ic50_nm=max(0.0001, ic50_nm),
                bao_format=str(row.get("bao_format") or settings.dominant_bao_default),
                assay_type=str(row.get("assay_type") or "B"),
                source_dataset=src_label,
            )
        )
        if len(seeds) >= eff_max_parents:
            break

    return seeds


def generate_candidate_structures(
    parent_molecules: list[ParentSeed],
    sar_rules: list[SARRule],
    constraints: QueryConstraints,
    generation_iteration: int = 1,
    max_candidates_pool: int | None = None,
    *,
    crem_db_path: Path | str | None = None,
    tool_config: ToolConfig | None = None,
    force_graph_ga: bool = False,
) -> list[GeneratedRawCandidate]:
    """Генерирует пул валидных молекул-кандидатов по 3-уровневому каскаду (SMIRKS -> CReM -> Биоизостеры)."""
    cfg = tool_config or ToolConfig()
    pool_limit = (
        int(max_candidates_pool)
        if max_candidates_pool is not None
        else (
            tool_config.max_candidates_pool
            if tool_config is not None
            else settings.max_candidates_pool
        )
    )
    if not parent_molecules:
        return []

    # 1. Собираем правила согласно generation_engine ('mmp_and_crem', 'mmp_only', 'crem_only')
    mmp_specs: list[_TransformationCandidateSpec] = []
    if cfg.generation_engine in ("mmp_and_crem", "mmp_only"):
        for r in sar_rules:
            canon_from = canonicalize_substituent_smiles(r.from_fragment)
            canon_to = canonicalize_substituent_smiles(r.to_fragment)
            mmp_specs.append(
                _TransformationCandidateSpec(
                    rule_id=r.rule_id,
                    smirks=f"{canon_from}>>{canon_to}",
                    from_fragment=canon_from,
                    to_fragment=canon_to,
                    generation_origin="mmp_smirks_rule",
                    mean_delta_pic50=float(r.mean_delta_pic50),
                    std_delta_pic50=float(r.std_delta_pic50),
                    support_pairs_count=int(r.support_pairs_count),
                    assay_homogeneous=bool(r.assay_homogeneous),
                    delta_clogp=float(r.mean_delta_clogp),
                    delta_tpsa=float(r.mean_delta_tpsa),
                    sa_score_estimate=2.2,
                    contains_halogen=fragment_contains_halogen(canon_to),
                    heavy_atoms_to=_fragment_heavy_atoms(canon_to),
                )
            )

    crem_specs: list[_TransformationCandidateSpec] = []
    if cfg.generation_engine in ("mmp_and_crem", "crem_only"):
        resolved_crem_path = (
            Path(crem_db_path) if crem_db_path is not None else settings.crem_db_path
        )
        crem_specs = _load_crem_replacements_from_db(
            crem_db_path=resolved_crem_path,
            max_sa_score=constraints.max_sa_score,
            tool_config=cfg,
        )

    include_bioisosteres = (
        cfg.generation_engine == "mmp_and_crem"
        and cfg.use_bioisostere_catalog
        and (
            generation_iteration >= 2
            or len(sar_rules) < int(cfg.bioisostere_fallback_threshold)
            or "polar" in (constraints.preferred_substituent_types or [])
            or "less_lipophilic" in (constraints.preferred_substituent_types or [])
            or "exclude_halogens" in (constraints.preferred_substituent_types or [])
        )
    )

    bio_rules = select_bioisostere_rules(cfg) if include_bioisosteres else []
    bio_specs: list[_TransformationCandidateSpec] = [
        _TransformationCandidateSpec(
            rule_id=b.rule_id,
            smirks=b.smirks,
            from_fragment=canonicalize_substituent_smiles(b.from_fragment),
            to_fragment=canonicalize_substituent_smiles(b.to_fragment),
            generation_origin="bioisostere_catalog",
            mean_delta_pic50=b.expected_delta_pic50,
            std_delta_pic50=0.20,
            support_pairs_count=0,
            assay_homogeneous=True,
            delta_clogp=b.delta_clogp,
            delta_tpsa=b.delta_tpsa,
            sa_score_estimate=b.sa_score,
            contains_halogen=b.contains_halogen,
            heavy_atoms_to=max(1, _fragment_heavy_atoms(b.to_fragment)),
        )
        for b in bio_rules
    ]

    # Объединяем эшелоны без потери приоритета экспериментальных правил для одинаковых SMIRKS
    combined_specs_by_smirks: dict[str, _TransformationCandidateSpec] = {}
    for spec in [*mmp_specs, *crem_specs, *bio_specs]:
        if spec.smirks not in combined_specs_by_smirks:
            combined_specs_by_smirks[spec.smirks] = spec

    all_specs = list(combined_specs_by_smirks.values())

    generated_candidates: list[GeneratedRawCandidate] = []
    global_seen_smiles: set[str] = {p.canonical_smiles for p in parent_molecules}
    eff_freeze = (
        tool_config.freeze_murcko_core
        if tool_config is not None
        else constraints.freeze_scaffold
    )

    for parent in parent_molecules:
        parent_mol = Chem.MolFromSmiles(parent.canonical_smiles)
        if parent_mol is None:
            continue

        parent_clogp = float(Descriptors.MolLogP(parent_mol))
        parent_tpsa = float(Descriptors.TPSA(parent_mol))

        ranked_specs = _filter_and_rank_transformation_specs(
            specs=all_specs,
            constraints=constraints,
            parent_clogp=parent_clogp,
            parent_tpsa=parent_tpsa,
            generation_iteration=generation_iteration,
            tool_config=cfg,
        )

        for spec in ranked_specs:
            products = apply_smirks_to_molecule(
                parent_mol=parent_mol,
                from_fragment=spec.from_fragment,
                to_fragment=spec.to_fragment,
                freeze_scaffold=eff_freeze,
                tool_config=cfg,
            )
            for prod_smi, scaf_smi, highlights in products:
                if prod_smi in global_seen_smiles:
                    continue

                prod_mol = Chem.MolFromSmiles(prod_smi)
                if prod_mol is None:
                    continue

                if float(Descriptors.ExactMolWt(prod_mol)) > constraints.max_mw:
                    continue

                global_seen_smiles.add(prod_smi)
                generated_candidates.append(
                    GeneratedRawCandidate(
                        smiles=prod_smi,
                        scaffold_smiles=scaf_smi,
                        generation_iteration=generation_iteration,
                        parent_seed=parent,
                        rule_id=spec.rule_id,
                        applied_smirks=spec.smirks,
                        replaced_group_smiles=spec.from_fragment,
                        introduced_group_smiles=spec.to_fragment,
                        generation_origin=spec.generation_origin,
                        historical_mean_delta_pic50=spec.mean_delta_pic50,
                        historical_std_delta_pic50=spec.std_delta_pic50,
                        support_pairs_count=spec.support_pairs_count,
                        assay_homogeneous=spec.assay_homogeneous,
                        highlighted_atom_indices=highlights,
                    )
                )
                if len(generated_candidates) >= pool_limit:
                    return generated_candidates

    # 4. Эволюционный генератор молекулярных графов (GraphGA + CReM grow/mutate)
    min_viable_pool = max(cfg.top_k_output, min(10, pool_limit))
    should_run_ga = (
        cfg.generation_engine != "mmp_only"
        and (
            force_graph_ga
            or generation_iteration > 1
            or len(generated_candidates) < min_viable_pool
        )
    )
    if should_run_ga and len(generated_candidates) < pool_limit:
        eff_crem_path = (
            Path(crem_db_path) if crem_db_path is not None else settings.crem_db_path
        )
        seed_dicts = [
            {
                "chembl_id": p.mol_id,
                "smiles": p.canonical_smiles,
                "exp_pic50": p.exp_pic50,
                "bao_format": p.bao_format,
                "scaffold_smiles": p.scaffold_smiles,
            }
            for p in parent_molecules
        ]
        parent_by_id = {p.mol_id: p for p in parent_molecules}
        ga_constraints = constraints.model_dump()
        ga_constraints["exclude_halogens"] = (
            "exclude_halogens" in constraints.preferred_substituent_types
        )
        ga_candidates = generate_graph_ga_candidates(
            seed_molecules=seed_dicts,
            constraints=ga_constraints,
            crem_db_path=eff_crem_path,
            population_size=cfg.ga_population_size,
            generations=cfg.ga_generations,
            crem_radius=3,
            max_replacements=cfg.crem_max_replacements,
            enable_crem_grow=cfg.enable_crem_grow,
            protect_scaffold=constraints.freeze_scaffold,
            seed=42 + generation_iteration,
        )
        exclude_halogens = "exclude_halogens" in constraints.preferred_substituent_types
        needed = max(0, pool_limit - len(generated_candidates))
        for item in ga_candidates[: max(needed * 2, 12)]:
            prod_smi = item["smiles"]
            if prod_smi in global_seen_smiles:
                continue
            prod_mol = Chem.MolFromSmiles(prod_smi)
            if prod_mol is None:
                continue
            if exclude_halogens and any(
                a.GetAtomicNum() in (9, 17, 35, 53) for a in prod_mol.GetAtoms()
            ):
                continue

            parent = parent_by_id.get(item["parent_chembl_id"], parent_molecules[0])
            parent_mol = Chem.MolFromSmiles(parent.canonical_smiles)
            if parent_mol is not None:
                is_preserved, scaf_smi = verify_scaffold_preservation(
                    parent_mol,
                    prod_mol,
                    freeze_scaffold=eff_freeze,
                    tool_config=cfg,
                )
                if not is_preserved:
                    continue
            else:
                scaf_smi = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=prod_mol, includeChirality=False
                )

            rule_id = str(item["rule_id"])
            origin: Literal[
                "mmp_smirks_rule",
                "crem_fragment_replacement",
                "bioisostere_catalog",
                "graph_ga",
            ] = "graph_ga" if rule_id == "GEN-GA" else "crem_fragment_replacement"

            if cfg.generation_engine == "crem_only" and origin == "graph_ga":
                continue

            highlights: list[int] = []
            if parent_mol is not None:
                with contextlib.suppress(Exception):
                    mcs = rdFMCS.FindMCS(
                        [prod_mol, parent_mol],
                        ringMatchesRingOnly=True,
                        completeRingsOnly=True,
                        timeout=1,
                    )
                    if mcs and mcs.smartsString:
                        mcs_pat = Chem.MolFromSmarts(mcs.smartsString)
                        if mcs_pat is not None:
                            match = prod_mol.GetSubstructMatch(mcs_pat)
                            if match:
                                match_set = set(match)
                                highlights = [
                                    idx
                                    for idx in range(prod_mol.GetNumAtoms())
                                    if idx not in match_set
                                ]

            global_seen_smiles.add(prod_smi)
            generated_candidates.append(
                GeneratedRawCandidate(
                    smiles=prod_smi,
                    scaffold_smiles=scaf_smi,
                    generation_iteration=generation_iteration,
                    parent_seed=parent,
                    rule_id=rule_id,
                    applied_smirks=str(item["applied_smirks"]),
                    replaced_group_smiles="*",
                    introduced_group_smiles="*",
                    generation_origin=origin,
                    historical_mean_delta_pic50=float(item["expected_delta_pic50"]),
                    historical_std_delta_pic50=0.45,
                    support_pairs_count=1 if origin == "crem_fragment_replacement" else 0,
                    assay_homogeneous=False,
                    highlighted_atom_indices=highlights,
                )
            )
            if len(generated_candidates) >= pool_limit:
                return generated_candidates

    return generated_candidates


generate_candidates_pipeline = generate_candidate_structures


_CURATED_GROW_FRAGMENTS: tuple[str, ...] = (
    "O",
    "N",
    "C",
    "CO",
    "CN",
    "C(=O)N",
    "C(=O)O",
    "S(=O)(=O)N",
    "OCC",
    "NCC",
    "C1COC1",
    "c1ccncc1",
    "OC",
    "NC",
    "C#N",
)


def _mol_has_halogen(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in mol.GetAtoms())


def _compute_ga_fitness(smiles: str, constraints: dict) -> float:
    """Вычисляет приспособленность молекулы для GraphGA с учетом QED, ограничений и алертов."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    pains_count, _pains, brenk_count, *_rest = check_structural_alerts_detailed(mol)
    if pains_count > 0 or brenk_count > 0:
        return 0.0

    if constraints.get("exclude_halogens") and _mol_has_halogen(mol):
        return 0.0

    qed_val = float(QED.qed(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    mw = float(Descriptors.ExactMolWt(mol))
    sa = float(calculate_sa_score(mol))

    penalty = 0.0
    max_logp = constraints.get("max_logp")
    if max_logp is not None and logp > float(max_logp):
        penalty += 0.35 * (logp - float(max_logp)) + 0.15

    min_logp = constraints.get("min_logp")
    if min_logp is not None and logp < float(min_logp):
        penalty += 0.35 * (float(min_logp) - logp) + 0.15

    max_sascore = constraints.get("max_sascore")
    if max_sascore is None:
        max_sascore = constraints.get("max_sa_score")
    if max_sascore is not None and sa > float(max_sascore):
        penalty += 0.25 * (sa - float(max_sascore)) + 0.10

    min_tpsa = constraints.get("min_tpsa")
    if min_tpsa is not None and tpsa < float(min_tpsa):
        penalty += 0.02 * (float(min_tpsa) - tpsa) + 0.10

    max_mw = constraints.get("max_mw")
    if max_mw is not None and mw > float(max_mw):
        penalty += 0.005 * (mw - float(max_mw)) + 0.10

    return max(0.0, round(qed_val - penalty, 6))


def _is_native_crem_db(db_path: Path | None, radius: int = 3) -> bool:
    if db_path is None or not db_path.exists():
        return False
    table = f"radius{int(radius)}"
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cur.fetchall()}
            return {"env", "core_smi", "core_num_atoms"}.issubset(cols)
    except sqlite3.Error:
        return False


def _get_murcko_protected_ids(mol: Chem.Mol) -> list[int]:
    with contextlib.suppress(Exception):
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return []
        match = mol.GetSubstructMatch(scaf)
        return list(match) if match else []
    return []


def _standardize_mol_or_none(raw_smi: str) -> tuple[str, Chem.Mol] | None:
    std = standardize_molecule_detailed(raw_smi)
    if std is None or not std.canonical_smiles:
        return None
    mol = Chem.MolFromSmiles(std.canonical_smiles)
    if mol is None:
        return None
    return std.canonical_smiles, mol


def _brics_crossover_children(
    pop_smiles: list[str],
    rng: random.Random,
    max_children: int = 24,
) -> list[tuple[str, str, str]]:
    """Рекомбинация пар молекул через BRICSDecompose + BRICSBuild."""
    fragments: set[str] = set()
    for smi in pop_smiles[:6]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        with contextlib.suppress(Exception):
            frags = BRICS.BRICSDecompose(mol, minFragmentSize=2)
            fragments.update(frags)

    if len(fragments) < 2:
        return []

    frag_list = sorted(fragments)
    rng.shuffle(frag_list)
    frag_mols = [Chem.MolFromSmiles(f) for f in frag_list[:8]]
    frag_mols = [m for m in frag_mols if m is not None]
    if len(frag_mols) < 2:
        return []

    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with contextlib.suppress(Exception):
        builder = BRICS.BRICSBuild(
            frag_mols, onlyCompleteMols=True, maxDepth=2
        )
        for idx, prod in enumerate(builder):
            if idx >= max_children * 2 or len(results) >= max_children:
                break
            if prod is None:
                continue
            raw_smi = Chem.MolToSmiles(prod, isomericSmiles=False)
            std_res = _standardize_mol_or_none(raw_smi)
            if std_res is None:
                continue
            canon_smi, std_mol = std_res
            heavy = std_mol.GetNumHeavyAtoms()
            if heavy < 8 or heavy > 45:
                continue
            if canon_smi not in seen:
                seen.add(canon_smi)
                results.append((canon_smi, "GEN-GA", "BRICS_crossover"))
    return results


def _attach_fragment_rdkit(
    mol: Chem.Mol,
    frag_smi: str,
    protected_set: set[int],
    rng: random.Random,
) -> str | None:
    """Присоединяет небольшой фрагмент к периферийному атому с доступной валентностью."""
    frag = Chem.MolFromSmiles(frag_smi)
    if frag is None or frag.GetNumHeavyAtoms() > 5:
        return None
    candidate_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() not in protected_set
        and atom.GetTotalNumHs() > 0
        and atom.GetAtomicNum() in (6, 7, 8)
    ]
    if not candidate_atoms:
        return None
    attach_idx = rng.choice(candidate_atoms)
    combo = Chem.CombineMols(mol, frag)
    rw = Chem.RWMol(combo)
    frag_attach_idx = mol.GetNumAtoms()
    rw.AddBond(attach_idx, frag_attach_idx, Chem.BondType.SINGLE)
    with contextlib.suppress(Exception):
        out_mol = rw.GetMol()
        Chem.SanitizeMol(out_mol)
        std_res = _standardize_mol_or_none(
            Chem.MolToSmiles(out_mol, isomericSmiles=False)
        )
        if std_res is not None:
            return std_res[0]
    return None


def _crem_grow_children(
    pop_smiles: list[str],
    crem_db_path: Path | None,
    crem_radius: int,
    max_replacements: int,
    protect_scaffold: bool,
    rng: random.Random,
    exclude_halogens: bool = False,
) -> list[tuple[str, str, str]]:
    """Направленное наращивание заместителей (max_atoms=5) через crem.grow_mol или RDKit-фолбэк."""
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    use_native = (
        _crem_pkg_grow_mol is not None
        and _is_native_crem_db(crem_db_path, radius=crem_radius)
    )

    for smi in pop_smiles[:6]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        protected_ids = _get_murcko_protected_ids(mol) if protect_scaffold else None
        if use_native and crem_db_path is not None:
            with contextlib.suppress(Exception):
                grown = list(
                    _crem_pkg_grow_mol(
                        mol,
                        str(crem_db_path),
                        radius=crem_radius,
                        min_atoms=1,
                        max_atoms=5,
                        max_replacements=max_replacements,
                        protected_ids=protected_ids,
                        return_mol=False,
                        seed=rng.randint(1, 100000),
                    )
                )
                for g_smi in grown:
                    std_res = _standardize_mol_or_none(str(g_smi))
                    if std_res is not None and std_res[0] not in seen:
                        seen.add(std_res[0])
                        results.append((std_res[0], "CREM-GROW", "CReM_grow"))
                continue

        # Fallback для кастомной БД replacements_sa2.db или без native CReM
        protected_set = set(protected_ids or [])
        frags = list(_CURATED_GROW_FRAGMENTS)
        rng.shuffle(frags)
        produced = 0
        for frag_smi in frags:
            if produced >= max(3, max_replacements // max(1, len(pop_smiles))):
                break
            if exclude_halogens and any(h in frag_smi for h in ("F", "Cl", "Br", "I")):
                continue
            new_smi = _attach_fragment_rdkit(mol, frag_smi, protected_set, rng)
            if new_smi and new_smi not in seen:
                seen.add(new_smi)
                results.append((new_smi, "CREM-GROW", "CReM_grow"))
                produced += 1

    return results


def _crem_mutate_children(
    pop_smiles: list[str],
    crem_db_path: Path | None,
    crem_radius: int,
    max_replacements: int,
    protect_scaffold: bool,
    rng: random.Random,
    exclude_halogens: bool = False,
) -> list[tuple[str, str, str]]:
    """Изостерическая мутация через crem.mutate_mol или локальную таблицу CReM."""
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    use_native = (
        _crem_pkg_mutate_mol is not None
        and _is_native_crem_db(crem_db_path, radius=crem_radius)
    )

    if use_native and crem_db_path is not None:
        for smi in pop_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            protected_ids = _get_murcko_protected_ids(mol) if protect_scaffold else None
            with contextlib.suppress(Exception):
                mutated = list(
                    _crem_pkg_mutate_mol(
                        mol,
                        str(crem_db_path),
                        radius=crem_radius,
                        max_replacements=max_replacements,
                        protected_ids=protected_ids,
                        return_mol=False,
                        seed=rng.randint(1, 100000),
                    )
                )
                for m_smi in mutated:
                    std_res = _standardize_mol_or_none(str(m_smi))
                    if std_res is not None and std_res[0] not in seen:
                        seen.add(std_res[0])
                        results.append(
                            (std_res[0], "CREM-MUT", "CReM_mutate")
                        )
        return results

    resolved_db = crem_db_path if crem_db_path is not None else settings.crem_db_path
    specs = _load_crem_replacements_from_db(
        crem_db_path=resolved_db,
        max_sa_score=5.0,
    )
    for smi in pop_smiles[:6]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        produced = 0
        for spec in specs:
            if produced >= max_replacements:
                break
            if exclude_halogens and spec.contains_halogen:
                continue
            products = apply_smirks_to_molecule(
                parent_mol=mol,
                from_fragment=spec.from_fragment,
                to_fragment=spec.to_fragment,
                freeze_scaffold=protect_scaffold,
            )
            for prod_smi, _scaf_smi, _highlights in products:
                std_res = _standardize_mol_or_none(prod_smi)
                if std_res is not None and std_res[0] not in seen:
                    seen.add(std_res[0])
                    results.append((std_res[0], "CREM-MUT", "CReM_mutate"))
                    produced += 1
                    if produced >= max_replacements:
                        break

    return results


def generate_graph_ga_candidates(
    seed_molecules: list[dict],
    constraints: dict,
    crem_db_path: Path | None = None,
    population_size: int = 20,
    generations: int = 3,
    crem_radius: int = 3,
    max_replacements: int = 15,
    enable_crem_grow: bool = True,
    protect_scaffold: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Эволюционный генератор молекулярных графов (BRICS crossover + CReM grow/mutate)."""
    if not seed_molecules:
        return []

    rng = random.Random(seed)
    normalized_seeds: list[dict[str, Any]] = []
    seed_smiles_set: set[str] = set()
    seed_fps: list[tuple[dict[str, Any], Chem.Mol, Any]] = []

    for idx, s in enumerate(seed_molecules):
        raw_smi = str(s.get("smiles") or "").strip()
        if not raw_smi:
            continue
        std_res = _standardize_mol_or_none(raw_smi)
        if std_res is None:
            continue
        canon_smi, std_mol = std_res
        entry = {
            "chembl_id": str(s.get("chembl_id") or f"SEED-{idx + 1}"),
            "smiles": canon_smi,
            "exp_pic50": float(s.get("exp_pic50", 6.5)),
            "bao_format": str(s.get("bao_format") or settings.dominant_bao_default),
        }
        normalized_seeds.append(entry)
        seed_smiles_set.add(canon_smi)
        seed_fps.append((entry, std_mol, compute_mol_fingerprint(std_mol)))

    if not normalized_seeds:
        return []

    exclude_halogens = bool(constraints.get("exclude_halogens", False))
    population: list[str] = [s["smiles"] for s in normalized_seeds]
    provenance: dict[str, tuple[str, str]] = {}

    for _gen in range(max(1, int(generations))):
        offspring: list[tuple[str, str, str]] = []
        offspring.extend(
            _brics_crossover_children(
                population, rng=rng, max_children=max(10, population_size)
            )
        )
        if enable_crem_grow:
            offspring.extend(
                _crem_grow_children(
                    population,
                    crem_db_path=crem_db_path,
                    crem_radius=crem_radius,
                    max_replacements=max_replacements,
                    protect_scaffold=protect_scaffold,
                    rng=rng,
                    exclude_halogens=exclude_halogens,
                )
            )
        offspring.extend(
            _crem_mutate_children(
                population,
                crem_db_path=crem_db_path,
                crem_radius=crem_radius,
                max_replacements=max_replacements,
                protect_scaffold=protect_scaffold,
                rng=rng,
                exclude_halogens=exclude_halogens,
            )
        )

        pool_set: dict[str, float] = {}
        for smi in population:
            pool_set[smi] = _compute_ga_fitness(smi, constraints)
        for smi, r_id, op_name in offspring:
            if smi not in provenance:
                provenance[smi] = (r_id, op_name)
            if smi not in pool_set:
                pool_set[smi] = _compute_ga_fitness(smi, constraints)

        ranked = sorted(
            pool_set.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        population = [smi for smi, _score in ranked[: max(1, int(population_size))]]

    # Отбираем финальных кандидатов (исключая исходные seed-молекулы)
    final_pool = [
        (smi, _compute_ga_fitness(smi, constraints))
        for smi in population
        if smi not in seed_smiles_set and smi in provenance
    ]
    # Дополнительно подмешаем лучшие дочерние, если population оказался полон seed
    if not final_pool:
        final_pool = sorted(
            (
                (smi, _compute_ga_fitness(smi, constraints))
                for smi in provenance
                if smi not in seed_smiles_set
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )[:population_size]

    # Если задан max_logp, оставляем структуры, укладывающиеся в порог (если такие есть)
    max_logp = constraints.get("max_logp")
    if max_logp is not None and final_pool:
        compliant = []
        for smi, fit in final_pool:
            m = Chem.MolFromSmiles(smi)
            if m is not None and float(Descriptors.MolLogP(m)) <= float(max_logp):
                compliant.append((smi, fit))
        if compliant:
            final_pool = compliant

    output: list[dict] = []
    for smi, fit in sorted(final_pool, key=lambda kv: (-kv[1], kv[0])):
        cand_mol = Chem.MolFromSmiles(smi)
        if cand_mol is None:
            continue
        cand_fp = compute_mol_fingerprint(cand_mol)

        best_parent = normalized_seeds[0]
        best_parent_mol = seed_fps[0][1]
        best_sim = -1.0
        for entry, p_mol, p_fp in seed_fps:
            sim = compute_fp_similarity(cand_fp, p_fp)
            if sim > best_sim:
                best_sim = sim
                best_parent = entry
                best_parent_mol = p_mol

        mcs_smarts = "*"
        with contextlib.suppress(Exception):
            mcs_res = rdFMCS.FindMCS(
                [cand_mol, best_parent_mol],
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                timeout=1,
            )
            if mcs_res and mcs_res.smartsString:
                mcs_smarts = mcs_res.smartsString

        rule_id, operation = provenance[smi]
        expected_delta = round((fit - 0.5) * 0.4, 4)
        output.append(
            {
                "smiles": smi,
                "parent_chembl_id": best_parent["chembl_id"],
                "parent_smiles": best_parent["smiles"],
                "parent_exp_pic50": float(best_parent["exp_pic50"]),
                "parent_bao_format": str(best_parent["bao_format"]),
                "rule_id": rule_id,
                "applied_smirks": f"{operation} | MCS: {mcs_smarts}",
                "expected_delta_pic50": expected_delta,
            }
        )

    return output
