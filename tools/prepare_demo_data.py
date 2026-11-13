"""Скрипт генерации эталонных локальных демонстрационных наборов данных (Слой 5).

Создает предкэшированные данные для всех 3 пользовательских сценариев из Раздела 7 SPEC.md:
1. `data/chembl/CHEMBL203.parquet` (EGFR — онкология, включает гефитиниб и ряд 4-анилинохиназолинов)
2. `data/chembl/CHEMBL5251.parquet` (BTK — аутоиммунные заболевания, пиразолопиримидины и аминопиримидины)
3. `data/chembl/CHEMBL262.parquet` (GSK-3beta — нейродегенеративные патологии, малеимиды и аминопиразолы)
4. `data/chembl/fallback_targets.parquet` (объединенный резервный слепок всех целевых мишеней)
5. `data/samples/custom_wetlab_series.csv` (пользовательская серия из 25 валидных соединений + краевые случаи:
   резкие перепады активности Activity Cliffs с Tanimoto >= 0.85 и |ΔpIC50| >= 1.5, дубликаты, соли,
   невалидные SMILES и цензурированные измерения '> 100 uM')
6. `data/crem/replacements_sa2.db` (компактная локальная SQLite-база изостерных замен CReM)
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from pishovichok.config import settings
from pishovichok.state.schemas import PARQUET_TABLE_REQUIRED_COLUMNS


def _pic50_to_ic50_nm(pic50: float) -> float:
    """Конвертирует значение pIC50 (-log10 M) в концентрацию IC50 в нМ."""
    return round(10.0 ** (9.0 - pic50), 3)


def _validate_smiles(smiles: str) -> str:
    """Проверяет химическую валидность SMILES в RDKit и возвращает исходную строку."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid RDKit SMILES in demo generator: {smiles}")
    return smiles


def _build_target_rows(
    target_chembl_id: str,
    scaffolds: list[tuple[str, str]],
    r_groups_homogeneous: list[tuple[str, float, str]],
    r_groups_heterogeneous_cell: list[tuple[str, float, str]],
    cliff_pairs: list[tuple[str, str, float, float, str]],
    extra_reference_drugs: list[tuple[str, str, float, str]],
    id_prefix: str,
) -> list[dict[str, Any]]:
    """Генерирует таблицу измерений ChEMBL по мишени с преднамеренной гетерогенностью BAO.

    Гарантирует:
    1. Наличие гомогенного пула `BAO_0000357` (single protein format, IC50) с серией MMP-правил
       (например, [*:1]OC >> [*:1]Cl, [*:1][H] >> [*:1]C#N, [*:1]Cl >> [*:1]C(=O)NC) с support >= 5.
    2. Наличие гетерогенной примеси `BAO_0000219` (cell-based format), формирующей ложно-высокий
       эффект для правила `[*:1][H] >> [*:1]S(=O)(=O)C` до запуска Петли самокоррекции №1 (`refilter_data`).
    3. Наличие пар Activity Cliffs с `Tanimoto >= 0.85` и `|ΔpIC50| >= 1.5`.
    4. Присутствие солевых форм (`.Cl`, `.CS(=O)(=O)O`), дубликатов и цензурированных записей (`> 100000 nM`).
    """
    rows: list[dict[str, Any]] = []
    row_idx = 1

    # 1. Эталонные препараты и близкие аналоги (например, Гефитиниб CHEMBL939 для EGFR)
    for chembl_id, smiles, pic50, bao in extra_reference_drugs:
        _validate_smiles(smiles)
        rows.append(
            {
                "raw_row_id": row_idx,
                "molecule_chembl_id": chembl_id,
                "raw_smiles": smiles,
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": _pic50_to_ic50_nm(pic50),
                "standard_units": "nM",
                "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_REF",
                "assay_type": "B" if bao == "BAO_0000357" else "F",
                "bao_format": bao,
                "target_chembl_id": target_chembl_id,
                "target_organism": "Homo sapiens",
            }
        )
        row_idx += 1

    # 2. Комбинаторная сетка: 6 скаффолдов x гомогенные заместители в BAO_0000357 (биохимический тест)
    for scaf_idx, (scaf_code, scaf_template) in enumerate(scaffolds, start=1):
        base_pic50 = 6.30 + 0.12 * (scaf_idx % 4)
        for rg_idx, (rg_smiles, delta_pic50, _rg_label) in enumerate(
            r_groups_homogeneous, start=1
        ):
            full_smiles = scaf_template.replace("{R}", rg_smiles)
            _validate_smiles(full_smiles)
            # Каждая 9-я запись снабжается противоионом HCl для проверки очистки солей
            raw_smiles = f"{full_smiles}.Cl" if (row_idx % 9 == 0) else full_smiles
            pic50 = round(base_pic50 + delta_pic50, 2)
            mol_id = f"CHEMBL_{id_prefix}_{scaf_code}_R{rg_idx:02d}"

            rows.append(
                {
                    "raw_row_id": row_idx,
                    "molecule_chembl_id": mol_id,
                    "raw_smiles": raw_smiles,
                    "standard_type": "IC50",
                    "standard_relation": "=",
                    "standard_value": _pic50_to_ic50_nm(pic50),
                    "standard_units": "nM",
                    "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_BIO_01",
                    "assay_type": "B",
                    "bao_format": "BAO_0000357",
                    "target_chembl_id": target_chembl_id,
                    "target_organism": "Homo sapiens",
                }
            )
            row_idx += 1

            # Добавляем повторные измерения (реплики) для первых двух заместителей каждого 2-го скаффолда
            if scaf_idx % 2 == 0 and rg_idx <= 2:
                rep_pic50 = round(pic50 + 0.08, 2)
                rows.append(
                    {
                        "raw_row_id": row_idx,
                        "molecule_chembl_id": mol_id,
                        "raw_smiles": full_smiles,
                        "standard_type": "IC50",
                        "standard_relation": "=",
                        "standard_value": _pic50_to_ic50_nm(rep_pic50),
                        "standard_units": "nM",
                        "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_BIO_02",
                        "assay_type": "B",
                        "bao_format": "BAO_0000357",
                        "target_chembl_id": target_chembl_id,
                        "target_organism": "Homo sapiens",
                    }
                )
                row_idx += 1

        # 3. Гетерогенные клеточные измерения (BAO_0000219, assay_type='F') на тех же скаффолдах!
        # Смешивание BAO_0000357 (R=H) и BAO_0000219 (R=S(=O)(=O)C и др.) порождает
        # методологически некорректные MMP-пары с assay_homogeneous=False -> триггер Петли №1!
        for het_idx, (het_rg, het_delta, _het_label) in enumerate(
            r_groups_heterogeneous_cell, start=1
        ):
            het_smiles = scaf_template.replace("{R}", het_rg)
            _validate_smiles(het_smiles)
            het_pic50 = round(base_pic50 + het_delta, 2)
            rows.append(
                {
                    "raw_row_id": row_idx,
                    "molecule_chembl_id": f"CHEMBL_{id_prefix}_{scaf_code}_CELL{het_idx:02d}",
                    "raw_smiles": het_smiles,
                    "standard_type": "IC50",
                    "standard_relation": "=",
                    "standard_value": _pic50_to_ic50_nm(het_pic50),
                    "standard_units": "nM",
                    "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_CELL_01",
                    "assay_type": "F",
                    "bao_format": "BAO_0000219",
                    "target_chembl_id": target_chembl_id,
                    "target_organism": "Homo sapiens",
                }
            )
            row_idx += 1

    # 4. Пары Activity Cliffs (высокое структурное сходство Tanimoto >= 0.85, |ΔpIC50| >= 1.5)
    for cliff_idx, (smi_act, smi_inact, pic50_act, pic50_inact, bao) in enumerate(
        cliff_pairs, start=1
    ):
        _validate_smiles(smi_act)
        _validate_smiles(smi_inact)
        rows.append(
            {
                "raw_row_id": row_idx,
                "molecule_chembl_id": f"CHEMBL_{id_prefix}_CLIFF_{cliff_idx:02d}A",
                "raw_smiles": smi_act,
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": _pic50_to_ic50_nm(pic50_act),
                "standard_units": "nM",
                "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_BIO_01",
                "assay_type": "B",
                "bao_format": bao,
                "target_chembl_id": target_chembl_id,
                "target_organism": "Homo sapiens",
            }
        )
        row_idx += 1
        rows.append(
            {
                "raw_row_id": row_idx,
                "molecule_chembl_id": f"CHEMBL_{id_prefix}_CLIFF_{cliff_idx:02d}B",
                "raw_smiles": smi_inact,
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": _pic50_to_ic50_nm(pic50_inact),
                "standard_units": "nM",
                "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_BIO_01",
                "assay_type": "B",
                "bao_format": bao,
                "target_chembl_id": target_chembl_id,
                "target_organism": "Homo sapiens",
            }
        )
        row_idx += 1

    # 5. Цензурированное неактивное измерение (> 100 uM)
    inactive_smi = scaffolds[0][1].replace("{R}", "C(C)(C)C")
    _validate_smiles(inactive_smi)
    rows.append(
        {
            "raw_row_id": row_idx,
            "molecule_chembl_id": f"CHEMBL_{id_prefix}_CENSORED_01",
            "raw_smiles": inactive_smi,
            "standard_type": "IC50",
            "standard_relation": ">",
            "standard_value": 100.0,
            "standard_units": "uM",
            "assay_chembl_id": f"CHEMBL_ASSAY_{id_prefix}_BIO_01",
            "assay_type": "B",
            "bao_format": "BAO_0000357",
            "target_chembl_id": target_chembl_id,
            "target_organism": "Homo sapiens",
        }
    )

    return rows


def generate_chembl203_egfr() -> list[dict[str, Any]]:
    """Генерирует эталонный датасет ингибиторов EGFR (CHEMBL203) для Сценария 2 и сквозного примера."""
    # 6 реальных 4-анилинохиназолиновых скаффолдов (близкая окрестность гефитиниба/эрлотиниба)
    scaffolds = [
        ("QNZ1", "COc1cc2ncnc(Nc3ccc({R})cc3)c2cc1OCCCN1CCOCC1"),
        ("QNZ2", "COc1cc2ncnc(Nc3ccc(F)c({R})c3)c2cc1OCCCN1CCOCC1"),
        ("QNZ3", "COc1cc2ncnc(Nc3ccc({R})c(F)c3)c2cc1OCCOC"),
        ("QNZ4", "COc1cc2ncnc(Nc3ccc({R})cc3)c2cc1OC"),
        ("QNZ5", "COc1cc2ncnc(Nc3cccc({R})c3)c2cc1OCCCN1CCN(C)CC1"),
        ("QNZ6", "COc1cc2ncnc(Nc3ccc({R})c(Cl)c3)c2cc1OCCN1CCCC1"),
    ]
    # Гомогенные заместители в BAO_0000357 (обеспечивают как галогеновые правила с ростом LogP,
    # так и полярные правила с ΔTPSA > 0 и снижением LogP для Петли №2!)
    r_groups_homogeneous = [
        ("OC", 0.00, "methoxy_baseline"),
        ("Cl", 0.72, "chloro_lipophilic"),
        ("F", 0.45, "fluoro"),
        ("C#N", 0.85, "cyano_polar"),
        ("C(=O)NC", 0.92, "methylamide_polar"),
        ("OCCO", 0.64, "hydroxyethoxy_polar"),
        ("C", 0.15, "methyl"),
    ]
    # Гетерогенные заместители в клеточном формате BAO_0000219 (создают ложное правило +1.40 до фильтрации)
    r_groups_heterogeneous_cell = [
        ("S(=O)(=O)C", 1.40, "methylsulfonyl_cell_assay"),
        ("S(=O)(=O)N", 1.25, "sulfonamide_cell_assay"),
    ]
    # Activity Cliffs с Tanimoto(Morgan r=2, 2048) >= 0.85 и |ΔpIC50| >= 1.5
    cliff_pairs = [
        (
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "CCc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            8.48,
            6.52,
            "BAO_0000357",
        ),
        (
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C)CC1",
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(CC)CC1",
            8.65,
            6.80,
            "BAO_0000357",
        ),
        (
            "COc1cc2ncnc(Nc3ccc(OCC4CCCC4)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "COc1cc2ncnc(Nc3ccc(OCC4CCCCC4)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            8.30,
            6.45,
            "BAO_0000357",
        ),
    ]
    extra_refs = [
        # Гефитиниб (CHEMBL939) — эталонный лидер Сценария 2
        (
            "CHEMBL939",
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            7.85,
            "BAO_0000357",
        ),
        # Эрлотиниб (CHEMBL553)
        ("CHEMBL553", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", 8.10, "BAO_0000357"),
        # Лапатиниб (CHEMBL554)
        (
            "CHEMBL554",
            "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
            7.92,
            "BAO_0000357",
        ),
    ]
    return _build_target_rows(
        target_chembl_id="CHEMBL203",
        scaffolds=scaffolds,
        r_groups_homogeneous=r_groups_homogeneous,
        r_groups_heterogeneous_cell=r_groups_heterogeneous_cell,
        cliff_pairs=cliff_pairs,
        extra_reference_drugs=extra_refs,
        id_prefix="EGFR",
    )


def generate_chembl5251_btk() -> list[dict[str, Any]]:
    """Генерирует эталонный датасет ингибиторов BTK (CHEMBL5251) для Сценария 1."""
    # Пиразоло[3,4-d]пиримидиновые и имидазопиразиновые скаффолды ингибиторов BTK
    scaffolds = [
        ("BTK1", "Nc1ncnc2c1c(-c1ccc(Oc3ccc({R})cc3)cc1)cn2C1CCN(C(=O)C=C)CC1"),
        ("BTK2", "Nc1ncnc2c1c(-c1ccc(Oc3cccc({R})c3)cc1)cn2C1CCCC1"),
        ("BTK3", "Nc1ncnc2c1c(-c1ccc(NC(=O)c3ccc({R})cc3)cc1)cn2C1CCN(C(=O)CC#N)CC1"),
        ("BTK4", "Nc1ncnc2c1c(-c1ccc(Oc3ccc({R})c(F)c3)cc1)cn2C1CCOCC1"),
        ("BTK5", "Nc1ncnc2c1c(-c1ccc(Oc3ccc({R})cc3)cc1)cn2CC1CCCN1C(=O)C=C"),
        ("BTK6", "Nc1ncnc2c1c(-c1ccc(Oc3ccc({R})cc3)c(F)c1)cn2C1CCN(C(=O)C)CC1"),
    ]
    r_groups_homogeneous = [
        ("OC", 0.00, "methoxy_baseline"),
        ("F", 0.68, "fluoro"),
        ("C#N", 0.94, "cyano_polar"),
        ("C(=O)NC", 1.05, "methylamide_polar"),
        ("OCCO", 0.76, "hydroxyethoxy_polar"),
        ("Cl", 0.55, "chloro"),
        ("C", 0.20, "methyl"),
    ]
    r_groups_heterogeneous_cell = [
        ("S(=O)(=O)C", 1.45, "methylsulfonyl_cell_assay"),
        ("S(=O)(=O)N", 1.30, "sulfonamide_cell_assay"),
    ]
    cliff_pairs = [
        (
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OCC4CCCC4)c(F)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OCC4CCCCC4)c(F)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            8.92,
            6.85,
            "BAO_0000357",
        ),
        (
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(F)c(OC)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(F)c(OCC)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            8.75,
            6.90,
            "BAO_0000357",
        ),
        (
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C#N)c(OCC4CCCC4)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C#N)c(OCC4CCCCC4)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            8.80,
            7.05,
            "BAO_0000357",
        ),
    ]
    extra_refs = [
        # Ибрутиниб (CHEMBL1873475)
        (
            "CHEMBL1873475",
            "Nc1ncnc2c1c(-c1ccc(Oc3ccccc3)cc1)cn2C1CCCN(C(=O)C=C)C1",
            8.92,
            "BAO_0000357",
        ),
        # Акалабрутиниб (CHEMBL3707348)
        (
            "CHEMBL3707348",
            "CC#CC(=O)N1CCCC1c1nc(-c2ccc(C(=O)Nc3ccccn3)cc2)c2c(N)nccn12",
            8.28,
            "BAO_0000357",
        ),
        # Занубрутиниб (CHEMBL3989921)
        (
            "CHEMBL3989921",
            "C=CC(=O)N1CCC(C2CCNc3c(C(N)=O)c(-c4ccc(Oc5ccccc5)cc4)nn23)CC1",
            8.55,
            "BAO_0000357",
        ),
    ]
    return _build_target_rows(
        target_chembl_id="CHEMBL5251",
        scaffolds=scaffolds,
        r_groups_homogeneous=r_groups_homogeneous,
        r_groups_heterogeneous_cell=r_groups_heterogeneous_cell,
        cliff_pairs=cliff_pairs,
        extra_reference_drugs=extra_refs,
        id_prefix="BTK",
    )


def generate_chembl262_gsk3b() -> list[dict[str, Any]]:
    """Генерирует эталонный датасет ингибиторов GSK-3beta (CHEMBL262)."""
    scaffolds = [
        ("GSK1", "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc({R})cc21"),
        ("GSK2", "Cn1cc(C2=C(c3cn(CCN4CCCC4)c4ccccc34)C(=O)NC2=O)c2ccc({R})cc21"),
        ("GSK3", "Cn1cc(C2=C(c3c[nH]c4ccccc34)C(=O)NC2=O)c2ccc({R})c(F)c21"),
        ("GSK4", "O=C1NC(=O)C(c2cn(CCCN3CCN(C)CC3)c3ccccc23)=C1c1ccc({R})cc1"),
        ("GSK5", "Cn1cc(C2=C(c3cn(CCOCCO)c4ccccc34)C(=O)NC2=O)c2ccc({R})cc21"),
        ("GSK6", "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4cc(F)ccc34)C(=O)NC2=O)c2ccc({R})cc21"),
    ]
    r_groups_homogeneous = [
        ("OC", 0.00, "methoxy_baseline"),
        ("Cl", 0.65, "chloro"),
        ("F", 0.52, "fluoro"),
        ("C#N", 0.88, "cyano_polar"),
        ("C(=O)NC", 0.95, "methylamide_polar"),
        ("OCCO", 0.70, "hydroxyethoxy_polar"),
        ("C", 0.18, "methyl"),
    ]
    r_groups_heterogeneous_cell = [
        ("S(=O)(=O)C", 1.38, "methylsulfonyl_cell_assay"),
        ("S(=O)(=O)N", 1.22, "sulfonamide_cell_assay"),
    ]
    cliff_pairs = [
        (
            "Cn1cc(C2=C(c3cn(CCCN4CCN(C)CC4)c4ccccc34)C(=O)NC2=O)c2ccc(F)cc21",
            "Cn1cc(C2=C(c3cn(CCCN4CCN(CC)CC4)c4ccccc34)C(=O)NC2=O)c2ccc(F)cc21",
            8.45,
            6.60,
            "BAO_0000357",
        ),
        (
            "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(OCC3CCCC3)cc21",
            "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(OCC3CCCCC3)cc21",
            8.60,
            6.75,
            "BAO_0000357",
        ),
        (
            "Cn1cc(C2=C(c3cn(CCCN4CCN(C)CC4)c4ccccc34)C(=O)NC2=O)c2ccc(C#N)cc21",
            "Cn1cc(C2=C(c3cn(CCCN4CCN(CC)CC4)c4ccccc34)C(=O)NC2=O)c2ccc(C#N)cc21",
            8.50,
            6.80,
            "BAO_0000357",
        ),
    ]
    extra_refs = [
        # CHIR-99021 (CHEMBL412142)
        (
            "CHEMBL412142",
            "Cc1cnc(-c2cnc(NCCNc3ccc(C#N)cn3)nc2-c2ccc(Cl)cc2Cl)o1",
            8.15,
            "BAO_0000357",
        ),
        # SB-216763 (CHEMBL292323)
        (
            "CHEMBL292323",
            "Cn1cc(C2=C(c3ccc(Cl)cc3Cl)C(=O)NC2=O)c2ccccc21",
            7.46,
            "BAO_0000357",
        ),
    ]
    return _build_target_rows(
        target_chembl_id="CHEMBL262",
        scaffolds=scaffolds,
        r_groups_homogeneous=r_groups_homogeneous,
        r_groups_heterogeneous_cell=r_groups_heterogeneous_cell,
        cliff_pairs=cliff_pairs,
        extra_reference_drugs=extra_refs,
        id_prefix="GSK3B",
    )


def generate_custom_wetlab_csv_rows() -> list[dict[str, str]]:
    """Формирует лабораторную серию `custom_wetlab_series.csv` для Сценария 3.

    Содержит:
    - 25 уникальных валидных соединений (производные хиназолинов и пиразолопиримидинов)
    - 4 пары резких перепадов активности (Activity Cliffs) с Tanimoto >= 0.85 и |ΔpIC50| >= 1.5
    - 3 дубликата (включая повтор в виде гидрохлоридной соли `.Cl` и запись в `uM`)
    - 3 невалидные записи SMILES (ошибка синтаксиса скобок, пустая/битая строка, пятивалентный углерод)
    - 2 неточных (цензурированных) измерения (`> 100` uM и `> 50` uM)
    """
    valid_series: list[tuple[str, str, str, str]] = [
        # Пара Activity Cliff #1: WL-001 vs WL-002 (пропокси-морфолин vs бутокси-морфолин, Tanimoto >= 0.90, ΔpIC50 = 2.02)
        ("WL-001", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "3.8", "nM"),
        ("WL-002", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCCN1CCOCC1", "398.0", "nM"),
        # Пара Activity Cliff #2: WL-003 vs WL-004 (N-метил vs N-этилпиперазин, Tanimoto >= 0.88, ΔpIC50 = 1.85)
        ("WL-003", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C)CC1", "2.5", "nM"),
        ("WL-004", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(CC)CC1", "178.0", "nM"),
        # Пара Activity Cliff #3: WL-005 vs WL-006 (циклопентилметокси vs циклогексилметокси, Tanimoto >= 0.90, ΔpIC50 = 1.72)
        (
            "WL-005",
            "COc1cc2ncnc(Nc3ccc(OCC4CCCC4)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "4.2",
            "nM",
        ),
        (
            "WL-006",
            "COc1cc2ncnc(Nc3ccc(OCC4CCCCC4)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "220.0",
            "nM",
        ),
        # Пара Activity Cliff #4: WL-007 vs WL-008 (циклопентилметокси vs циклогексилметокси на скаффолде BTK, Tanimoto >= 0.88, ΔpIC50 = 2.10)
        (
            "WL-007",
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OCC4CCCC4)c(F)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            "1.9",
            "nM",
        ),
        (
            "WL-008",
            "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OCC4CCCCC4)c(F)c3)cc1)cn2C1CCN(C(=O)C=C)CC1",
            "240.0",
            "nM",
        ),
        # Систематический гомологический ряд заместителей вокруг двух скаффолдов (WL-009 .. WL-025)
        ("WL-009", "COc1cc2ncnc(Nc3ccc(OC)c(Cl)c3)c2cc1OCCCN1CCOCC1", "85.0", "nM"),
        ("WL-010", "COc1cc2ncnc(Nc3ccc(Cl)c(Cl)c3)c2cc1OCCCN1CCOCC1", "18.5", "nM"),
        ("WL-011", "COc1cc2ncnc(Nc3ccc(C(=O)NC)c(Cl)c3)c2cc1OCCCN1CCOCC1", "6.4", "nM"),
        ("WL-012", "COc1cc2ncnc(Nc3ccc(OCCO)c(Cl)c3)c2cc1OCCCN1CCOCC1", "11.2", "nM"),
        ("WL-013", "COc1cc2ncnc(Nc3ccc(C)c(Cl)c3)c2cc1OCCCN1CCOCC1", "62.0", "nM"),
        ("WL-014", "COc1cc2ncnc(Nc3ccc(OC)cc3)c2cc1OCCCN1CCOCC1", "140.0", "nM"),
        ("WL-015", "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OCCCN1CCOCC1", "29.0", "nM"),
        ("WL-016", "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OCCCN1CCOCC1", "9.8", "nM"),
        ("WL-017", "COc1cc2ncnc(Nc3ccc(C(=O)NC)cc3)c2cc1OCCCN1CCOCC1", "7.5", "nM"),
        ("WL-018", "COc1cc2ncnc(Nc3ccc(OCCO)cc3)c2cc1OCCCN1CCOCC1", "14.0", "nM"),
        ("WL-019", "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCCCN1CCOCC1", "44.0", "nM"),
        ("WL-020", "COc1cc2ncnc(Nc3ccc(C)cc3)c2cc1OCCCN1CCOCC1", "95.0", "nM"),
        ("WL-021", "COc1cc2ncnc(Nc3ccc(OC)c(F)c3)c2cc1OCCOC", "110.0", "nM"),
        ("WL-022", "COc1cc2ncnc(Nc3ccc(Cl)c(F)c3)c2cc1OCCOC", "24.0", "nM"),
        ("WL-023", "COc1cc2ncnc(Nc3ccc(C#N)c(F)c3)c2cc1OCCOC", "8.9", "nM"),
        ("WL-024", "COc1cc2ncnc(Nc3ccc(C(=O)NC)c(F)c3)c2cc1OCCOC", "6.8", "nM"),
        ("WL-025", "COc1cc2ncnc(Nc3ccc(OCCO)c(F)c3)c2cc1OCCOC", "12.5", "nM"),
    ]
    for _mid, smi, _val, _unit in valid_series:
        _validate_smiles(smi)

    edge_cases: list[tuple[str, str, str, str]] = [
        # Краевой случай 1: Дубликаты (повтор WL-001 в виде соли HCl и повтор WL-011 в единицах uM)
        (
            "WL-001-DUP1",
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1.Cl",
            "4.2",
            "nM",
        ),
        ("WL-001-DUP2", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "3.5", "nM"),
        (
            "WL-011-DUP1",
            "COc1cc2ncnc(Nc3ccc(C(=O)NC)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "0.0068",
            "uM",
        ),
        # Краевой случай 2: Неточные / цензурированные значения (> 100 uM, > 50 uM)
        ("WL-026-CENS", "COc1cc2ncnc(Nc3ccc(C(C)(C)C)cc3)c2cc1OC", "> 100", "uM"),
        ("WL-027-CENS", "COc1cc2ncnc(Nc3ccc(Br)cc3)c2cc1OC", ">50", "uM"),
        # Краевой случай 3: Невалидные записи SMILES (должны изолироваться в validation_errors_log)
        ("WL-ERR-01", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3", "15.0", "nM"),
        ("WL-ERR-02", "C(C)(C)(C)(C)c1ccccc1", "25.0", "nM"),
        ("WL-ERR-03", "INVALID_SMILES_TOKEN_XYZ", "50.0", "nM"),
    ]

    all_records = valid_series + edge_cases
    return [
        {"mol_id": mol_id, "SMILES": smi, "IC50_nM": ic50, "units": units}
        for mol_id, smi, ic50, units in all_records
    ]


def build_crem_sqlite_db(db_path: Path) -> int:
    """Создает компактную локальную SQLite-базу фрагментов CReM (`replacements_sa2.db`)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    replacements = [
        ("c1ccccc1[*]", "[*:1]F", "[*:1]C#N", 3, 1.8, -0.42, 23.79),
        ("c1ccccc1[*]", "[*:1]Cl", "[*:1]C#N", 3, 1.9, -0.95, 23.79),
        ("c1ccccc1[*]", "[*:1]Cl", "[*:1]C(=O)NC", 3, 2.1, -1.28, 29.10),
        ("c1ccccc1[*]", "[*:1]OC", "[*:1]OCCO", 3, 2.2, -0.54, 20.23),
        ("c1ccccc1[*]", "[*:1]OC", "[*:1]C#N", 3, 1.9, -0.25, 14.56),
        ("c1ccccc1[*]", "[*:1]C", "[*:1]C(F)(F)F", 3, 2.3, 0.65, 0.00),
        ("c1ccccc1[*]", "[*:1]Cl", "[*:1]OC", 3, 1.7, -0.64, 9.23),
        ("c1ccccc1[*]", "[*:1]F", "[*:1]OC", 3, 1.7, -0.11, 9.23),
        ("c1ccccc1[*]", "[*:1]C", "[*:1]CO", 3, 1.8, -1.03, 20.23),
        ("c1ccccc1[*]", "[*:1]Cl", "[*:1]S(=O)(=O)C", 3, 2.4, -1.45, 34.14),
    ]

    conn = sqlite3.connect(db_path.as_posix())
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE radius3 (
                context TEXT NOT NULL,
                from_smarts TEXT NOT NULL,
                to_smarts TEXT NOT NULL,
                radius INTEGER NOT NULL,
                sa_score REAL NOT NULL,
                delta_clogp REAL NOT NULL,
                delta_tpsa REAL NOT NULL
            )
            """
        )
        cur.executemany(
            "INSERT INTO radius3 VALUES (?, ?, ?, ?, ?, ?, ?)",
            replacements,
        )
        conn.commit()
    finally:
        conn.close()
    return len(replacements)


def verify_activity_cliffs_exist(
    rows: list[dict[str, Any]], smiles_key: str, pic50_fn: Any
) -> int:
    """Подтверждает наличие пар Activity Cliffs (Tanimoto >= 0.85 и |ΔpIC50| >= 1.5) в датасете."""
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    valid_entries: list[tuple[str, Any, float]] = []
    for r in rows:
        pic50 = pic50_fn(r)
        if pic50 is None:
            continue
        smi = r[smiles_key].split(".")[0]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = fp_gen.GetFingerprint(mol)
        valid_entries.append((smi, fp, pic50))

    cliff_count = 0
    for i in range(len(valid_entries)):
        for j in range(i + 1, len(valid_entries)):
            smi_i, fp_i, p_i = valid_entries[i]
            smi_j, fp_j, p_j = valid_entries[j]
            if smi_i == smi_j:
                continue
            sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
            if sim >= 0.85 and abs(p_i - p_j) >= 1.5:
                cliff_count += 1
    return cliff_count


def prepare_all_demo_datasets(base_data_dir: Path | None = None) -> dict[str, Any]:
    """Генерирует и валидирует все демонстрационные файлы в директории `/data`."""
    data_dir = base_data_dir or settings.resolved_data_dir
    chembl_dir = data_dir / "chembl"
    samples_dir = data_dir / "samples"
    crem_dir = data_dir / "crem"
    models_dir = data_dir / "models"
    receptors_dir = data_dir / "receptors"

    for d in (chembl_dir, samples_dir, crem_dir, models_dir, receptors_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_datasets = {
        "CHEMBL203": generate_chembl203_egfr(),
        "CHEMBL5251": generate_chembl5251_btk(),
        "CHEMBL262": generate_chembl262_gsk3b(),
    }

    summary: dict[str, Any] = {"targets": {}}
    combined_fallback_rows: list[dict[str, Any]] = []

    for target_id, rows in target_datasets.items():
        # Проверяем строгое соответствие колонок схеме raw_dataset
        table = pa.Table.from_pylist(rows)
        required_cols = set(PARQUET_TABLE_REQUIRED_COLUMNS["raw_dataset"])
        assert required_cols.issubset(set(table.column_names)), (
            f"Missing columns in {target_id}"
        )

        out_path = chembl_dir / f"{target_id}.parquet"
        pq.write_table(table, out_path, compression="zstd")

        # Проверяем гетерогенность BAO и наличие Activity Cliffs
        bao_counts: dict[str, int] = {}
        for r in rows:
            b = r["bao_format"]
            bao_counts[b] = bao_counts.get(b, 0) + 1

        cliffs = verify_activity_cliffs_exist(
            rows,
            smiles_key="raw_smiles",
            pic50_fn=lambda r: (
                9.0 - math.log10(float(r["standard_value"]))
                if r["standard_relation"] == "=" and r["standard_units"] == "nM"
                else None
            ),
        )
        assert len(bao_counts) >= 2, (
            f"Target {target_id} must contain heterogeneous BAO formats"
        )
        assert cliffs >= 3, (
            f"Target {target_id} must contain at least 3 Activity Cliffs, got {cliffs}"
        )

        summary["targets"][target_id] = {
            "path": out_path.as_posix(),
            "rows": len(rows),
            "bao_distribution": bao_counts,
            "activity_cliffs_verified": cliffs,
        }
        combined_fallback_rows.extend(rows)

    # Сохраняем резервный слепок fallback_targets.parquet
    fallback_path = chembl_dir / "fallback_targets.parquet"
    pq.write_table(
        pa.Table.from_pylist(combined_fallback_rows), fallback_path, compression="zstd"
    )
    summary["fallback_targets_rows"] = len(combined_fallback_rows)

    # Генерируем пользовательский CSV лабораторной серии для Сценария 3
    csv_rows = generate_custom_wetlab_csv_rows()
    csv_path = samples_dir / "custom_wetlab_series.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["mol_id", "SMILES", "IC50_nM", "units"])
        writer.writeheader()
        writer.writerows(csv_rows)

    def _csv_pic50(r: dict[str, str]) -> float | None:
        raw_val = r["IC50_nM"].strip()
        if raw_val.startswith((">", "<")):
            return None
        val = float(raw_val)
        val_nm = val * 1000.0 if r["units"] == "uM" else val
        return 9.0 - math.log10(val_nm)

    csv_cliffs = verify_activity_cliffs_exist(
        csv_rows, smiles_key="SMILES", pic50_fn=_csv_pic50
    )
    assert csv_cliffs >= 4, (
        f"CSV series must contain >= 4 Activity Cliffs, got {csv_cliffs}"
    )

    summary["custom_wetlab_csv"] = {
        "path": csv_path.as_posix(),
        "total_rows": len(csv_rows),
        "activity_cliffs_verified": csv_cliffs,
    }

    # Генерируем компактную SQLite базу CReM
    crem_count = build_crem_sqlite_db(crem_dir / "replacements_sa2.db")
    summary["crem_replacements_count"] = crem_count

    # Подготавливаем 3 предкэшированных демо-рецептора для физического 3D-докинга
    from pishovichok.config import DEMO_RECEPTOR_PRESETS
    from pishovichok.core.receptor_resolver import (
        generate_minimal_demo_receptor,
        resolve_receptor,
    )

    receptors_summary: dict[str, Any] = {}
    for target_id in DEMO_RECEPTOR_PRESETS:
        try:
            rec_meta = resolve_receptor(target_id, receptors_dir=receptors_dir)
            if rec_meta.get("status") != "READY":
                rec_meta = generate_minimal_demo_receptor(
                    target_id, receptors_dir=receptors_dir
                )
        except Exception:
            rec_meta = generate_minimal_demo_receptor(
                target_id, receptors_dir=receptors_dir
            )
        receptors_summary[target_id] = {
            "status": rec_meta.get("status"),
            "pdb_id": rec_meta.get("pdb_id"),
            "source": rec_meta.get("source"),
            "pdb_path": rec_meta.get("pdb_path"),
            "pdbqt_path": rec_meta.get("pdbqt_path"),
            "box_center": rec_meta.get("box_center"),
        }
    summary["receptors"] = receptors_summary

    return summary


def main() -> int:
    """Точка входа CLI для генерации демонстрационных данных."""
    report = prepare_all_demo_datasets()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
