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

from pishovichok.state.schemas import CalcRDKitMetrics

_SASCORER_MODULE: Any = None
_SASCORER_LOAD_ATTEMPTED: bool = False

# Предкомпилированные синглтоны каталогов подструктурных фильтров RDKit C++
_PAINS_CATALOG: FilterCatalog.FilterCatalog | None = None
_BRENK_CATALOG: FilterCatalog.FilterCatalog | None = None
_GLAXO_CATALOG: FilterCatalog.FilterCatalog | None = None

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


def _init_filter_catalogs() -> tuple[
    FilterCatalog.FilterCatalog | None,
    FilterCatalog.FilterCatalog | None,
    FilterCatalog.FilterCatalog | None,
]:
    """Инициализирует и кэширует каталоги PAINS, Brenk и Glaxo."""
    global _PAINS_CATALOG, _BRENK_CATALOG, _GLAXO_CATALOG
    if _PAINS_CATALOG is None:
        with contextlib.suppress(Exception):
            pains_params = FilterCatalog.FilterCatalogParams()
            pains_params.AddCatalog(
                FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS
            )
            _PAINS_CATALOG = FilterCatalog.FilterCatalog(pains_params)

            brenk_params = FilterCatalog.FilterCatalogParams()
            brenk_params.AddCatalog(
                FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK
            )
            _BRENK_CATALOG = FilterCatalog.FilterCatalog(brenk_params)

            glaxo_params = FilterCatalog.FilterCatalogParams()
            glaxo_params.AddCatalog(
                FilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo
            )
            _GLAXO_CATALOG = FilterCatalog.FilterCatalog(glaxo_params)

    return (_PAINS_CATALOG, _BRENK_CATALOG, _GLAXO_CATALOG)


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


def check_structural_alerts(
    mol: Chem.Mol,
) -> tuple[int, list[str], int, list[str], int]:
    """Проверяет молекулу по медико-химическим каталогам PAINS (A/B/C), Brenk и Glaxo.

    Возвращает кортеж:
        `(pains_count, pains_matched_names, brenk_count, brenk_matched_names, glaxo_count)`
    """
    pains_cat, brenk_cat, glaxo_cat = _init_filter_catalogs()

    pains_names: list[str] = []
    brenk_names: list[str] = []
    glaxo_count = 0

    if pains_cat is not None:
        for entry in pains_cat.GetMatches(mol):
            desc = str(entry.GetDescription())
            if desc not in pains_names:
                pains_names.append(desc)

    if brenk_cat is not None:
        for entry in brenk_cat.GetMatches(mol):
            desc = str(entry.GetDescription())
            # Aliphatic_long_chain в каталоге Brenk — эвристика длины алкильной цепи, а не реакционноспособный токсикофор
            if desc == "Aliphatic_long_chain":
                continue
            if desc not in brenk_names:
                brenk_names.append(desc)

    if glaxo_cat is not None:
        glaxo_count = len(glaxo_cat.GetMatches(mol))

    return (len(pains_names), pains_names, len(brenk_names), brenk_names, glaxo_count)


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


def decompose_brics_precursors(mol: Chem.Mol) -> tuple[list[str], int]:
    """Выполняет ретросинтетическую декомпозицию молекулы по алгоритму BRICS.

    Возвращает `(brics_fragments, brics_bonds_broken)`:
    - `brics_fragments`: отсортированный по убыванию размера список валидных канонических SMILES
      исходных строительных блоков (без фиктивных меток `[n*]`).
    - `brics_bonds_broken`: количество разорванных ретросинтетических связей BRICS.
    """
    canon_full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        bonds_broken = len(brics_bonds)
        if bonds_broken == 0:
            return ([canon_full], 0)

        raw_frags = set(BRICS.BRICSDecompose(mol, returnMols=False))
    except Exception:  # noqa: BLE001
        return ([canon_full], 0)

    cleaned_set: set[str] = set()
    for raw_f in raw_frags:
        clean_smi = _clean_brics_fragment_smiles(str(raw_f))
        if clean_smi:
            cleaned_set.add(clean_smi)

    if not cleaned_set:
        return ([canon_full], 0)

    # Если есть содержательные строительные блоки (>= 2 тяжелых атомов), исключаем тривиальные одноатомные осколки (H2O, NH3, CH4)
    substantial = [
        smi
        for smi in cleaned_set
        if (m := Chem.MolFromSmiles(smi)) is not None and m.GetNumHeavyAtoms() >= 2
    ]
    final_list = substantial if substantial else list(cleaned_set)
    final_list.sort(
        key=lambda s: (
            -(Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else 0),
            s,
        )
    )
    return (final_list, bonds_broken)


def count_lipinski_violations(mw: float, clogp: float, hbd: int, hba: int) -> int:
    """Подсчитывает число нарушений классического правила пяти Липински (Ro5, от 0 до 4)."""
    violations = 0
    if mw > 500.0:
        violations += 1
    if clogp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def calculate_rdkit_metrics(
    mol_or_smiles: Chem.Mol | str,
    parent_mol_or_smiles: Chem.Mol | str | None = None,
) -> CalcRDKitMetrics:
    """Вычисляет полный детерминированный профиль `CalcRDKitMetrics` (Категория 2 данных кандидата)."""
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
    lipinski = count_lipinski_violations(mw=mw, clogp=clogp, hbd=hbd, hba=hba)

    pains_count, pains_names, brenk_count, brenk_names, glaxo_count = (
        check_structural_alerts(mol)
    )
    brics_fragments, brics_bonds_broken = decompose_brics_precursors(mol)

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
        brics_fragments=brics_fragments,
        brics_bonds_broken=brics_bonds_broken,
        delta_clogp_vs_parent=delta_clogp,
        delta_tpsa_vs_parent=delta_tpsa,
    )
