"""Модуль сбора, курации и стандартизации данных биоактивности (Слой 1, без участия LLM).

Реализует:
1. Каскадную загрузку выборок из локального кэша Parquet (`/data/chembl/<target>.parquet`),
   опционального REST API ChEMBL и гарантированного резервного слепка (`fallback_targets.parquet`).
2. Парсинг пользовательских таблиц CSV с изоляцией невалидных SMILES и некорректных
   значений активности в `validation_errors_log`.
3. Химическую стандартизацию молекул через `rdkit.Chem.MolStandardize.rdMolStandardize`:
   `Cleanup` -> `LargestFragmentChooser.choose` (удаление солей и противоионов) ->
   `Uncharger.uncharge` (нейтрализация зарядов) -> `TautomerEnumerator.Canonicalize`
   (с детерминированным фолбэком на нейтрализованный родительский граф).
4. Приведение концентраций (`nM`, `uM`, `mM`, `pM`, `M`) к логарифмической шкале
   `pChEMBL = -log10(C [M])` и маркировку цензурированных измерений (`> 100 uM`).
5. Дедупликацию структур по `(InChIKey, bao_format, standard_type)` с расчетом медианы
   `pchembl_value` и стандартного отклонения `pchembl_std`.
6. Выделение базовых каркасов Bemis-Murcko (`MurckoScaffold.GetScaffoldForMol`) и
   расчет сходства Танимото к молекуле-лидеру для Сценария 2.
7. Поддержку базового (`assay_filter_strict=False`) и строгого (`assay_filter_strict=True`)
   режимов фильтрации по однородности экспериментальных протоколов (`bao_format`, `standard_type`).
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, NamedTuple

import duckdb
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from pishovichok.config import settings
from pishovichok.core.fingerprints import compute_fp_similarity, compute_mol_fingerprint
from pishovichok.state.schemas import (
    DatasetCurationSummary,
    ParquetPointer,
    QueryConstraints,
    ToolConfig,
)
from pishovichok.state.storage import write_parquet_pointer

# Инициализируем синглтоны стандартизатора RDKit
_LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()

_RELATION_VALUE_RE = re.compile(
    r"^\s*(>=|<=|>|<|=)?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

_UNIT_TO_MOLAR_FACTOR: dict[str, float] = {
    "m": 1.0,
    "mol/l": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
}


class StandardizedMoleculeResult(NamedTuple):
    """Результат химической стандартизации и выделения каркаса молекулы."""

    canonical_smiles: str
    inchi_key: str
    scaffold_smiles: str
    salts_stripped: bool


def standardize_molecule_detailed(
    raw_smiles: str,
    errors_log: list[str] | None = None,
    row_label: str = "",
    *,
    tool_config: ToolConfig | None = None,
) -> StandardizedMoleculeResult | None:
    """Выполняет полную стандартизацию молекулы в RDKit и выделяет скаффолд Bemis-Murcko.

    Параметры `strip_salts`, `neutralize_charges`, `canonicalize_tautomers`, `keep_chirality`
    и `scaffold_mode` управляются через `tool_config`.
    """
    cfg = tool_config or ToolConfig()
    prefix = f"[{row_label}] " if row_label else ""
    if not isinstance(raw_smiles, str) or not raw_smiles.strip():
        if errors_log is not None:
            errors_log.append(f"{prefix}Empty or missing SMILES string.")
        return None

    cleaned_input = raw_smiles.strip()
    mol = Chem.MolFromSmiles(cleaned_input, sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        if errors_log is not None:
            errors_log.append(
                f"{prefix}Invalid SMILES rejected by RDKit parser: '{cleaned_input}'"
            )
        return None

    try:
        initial_atoms = mol.GetNumAtoms()
        has_dot_fragments = "." in cleaned_input

        clean_mol = rdMolStandardize.Cleanup(mol)
        if clean_mol is None:
            clean_mol = mol

        if cfg.strip_salts:
            parent_mol = _LARGEST_FRAGMENT_CHOOSER.choose(clean_mol)
            if parent_mol is None:
                parent_mol = clean_mol
            salts_stripped = has_dot_fragments or (
                parent_mol.GetNumAtoms() < initial_atoms
            )
        else:
            parent_mol = clean_mol
            salts_stripped = False

        if cfg.neutralize_charges:
            neutral_mol = _UNCHARGER.uncharge(parent_mol)
            if neutral_mol is None:
                neutral_mol = parent_mol
        else:
            neutral_mol = parent_mol

        if cfg.canonicalize_tautomers:
            try:
                canon_mol = _TAUTOMER_ENUMERATOR.Canonicalize(neutral_mol)
                if canon_mol is None:
                    canon_mol = neutral_mol
            except Exception:  # noqa: BLE001
                canon_mol = neutral_mol
        else:
            canon_mol = neutral_mol

        if not cfg.keep_chirality:
            canon_mol = Chem.Mol(canon_mol)
            Chem.RemoveStereochemistry(canon_mol)

        Chem.SanitizeMol(canon_mol)
        use_isomeric = bool(cfg.keep_chirality)
        canonical_smiles = Chem.MolToSmiles(
            canon_mol, canonical=True, isomericSmiles=use_isomeric
        )
        inchi_key = Chem.MolToInchiKey(canon_mol)

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(canon_mol)
        if (
            cfg.scaffold_mode == "generic_framework"
            and scaffold_mol is not None
            and scaffold_mol.GetNumAtoms() > 0
        ):
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        scaffold_smiles = (
            Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=use_isomeric)
            if scaffold_mol is not None and scaffold_mol.GetNumAtoms() > 0
            else canonical_smiles
        )
        if not scaffold_smiles:
            scaffold_smiles = canonical_smiles

        return StandardizedMoleculeResult(
            canonical_smiles=canonical_smiles,
            inchi_key=inchi_key,
            scaffold_smiles=scaffold_smiles,
            salts_stripped=salts_stripped,
        )
    except Exception as exc:  # noqa: BLE001
        if errors_log is not None:
            errors_log.append(
                f"{prefix}Standardization failed for SMILES '{cleaned_input}': {type(exc).__name__}: {exc}"
            )
        return None


def standardize_smiles(
    raw_smiles: str,
    errors_log: list[str] | None = None,
    row_label: str = "",
    *,
    tool_config: ToolConfig | None = None,
) -> tuple[str, str, str] | None:
    """Контракт стандартизации из Раздела 3.2 ARCHITECTURE.md.

    Возвращает `(canonical_smiles, inchi_key, scaffold_smiles)` при успехе либо `None` при ошибке.
    """
    res = standardize_molecule_detailed(
        raw_smiles,
        errors_log=errors_log,
        row_label=row_label,
        tool_config=tool_config,
    )
    if res is None:
        return None
    return (res.canonical_smiles, res.inchi_key, res.scaffold_smiles)


def parse_activity_cell(
    raw_value: str | float | None,
    default_relation: str = "=",
) -> tuple[str, float] | None:
    """Разбирает ячейку активности (например, '3.8', '> 100', '<= 50') на оператор и число."""
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        val = float(raw_value)
        if math.isnan(val) or math.isinf(val) or val <= 0.0:
            return None
        rel = (
            default_relation.strip()
            if default_relation and default_relation.strip()
            else "="
        )
        return (rel, val)

    text = str(raw_value).strip()
    if not text:
        return None
    match = _RELATION_VALUE_RE.match(text)
    if not match:
        return None

    rel_token, num_token = match.groups()
    rel = (
        rel_token
        if rel_token
        else (default_relation.strip() if default_relation else "=")
    )
    try:
        val = float(num_token)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val) or val <= 0.0:
        return None
    return (rel, val)


def convert_to_pchembl(
    value: float,
    units: str = "nM",
    relation: str = "=",
    *,
    censored_floor: float | None = None,
    tool_config: ToolConfig | None = None,
) -> tuple[float, bool]:
    """Конвертирует концентрацию IC50/Ki/Kd/EC50 в шкалу pChEMBL = -log10(C [M]).

    Для правоцензурированных неактивных значений (`>` или `>=`, например `> 100 uM`)
    возвращает `is_censored = True` и фиксирует верхнюю оценку активности `pChEMBL = censored_floor`.
    """
    if value <= 0.0 or math.isnan(value) or math.isinf(value):
        raise ValueError(
            f"Bioactivity concentration must be positive finite float, got {value}"
        )

    unit_key = (units or "nM").strip().lower()
    factor = _UNIT_TO_MOLAR_FACTOR.get(unit_key)
    if factor is None:
        raise ValueError(f"Unsupported bioactivity concentration units: '{units}'")

    molar_conc = value * factor
    raw_pchembl = -math.log10(molar_conc)
    norm_rel = (relation or "=").strip()
    is_censored = norm_rel in (">", ">=")
    floor = (
        censored_floor
        if censored_floor is not None
        else (tool_config.pchembl_min_cutoff if tool_config is not None else 4.0)
    )

    if is_censored:
        pchembl_val = min(float(floor), round(raw_pchembl, 4))
    else:
        pchembl_val = round(raw_pchembl, 4)

    return (pchembl_val, is_censored)


def _load_chembl_target_rows(
    target_chembl_id: str,
    data_dir: Path,
    errors_log: list[str],
    *,
    tool_config: ToolConfig | None = None,
) -> tuple[list[dict[str, Any]], Literal["chembl_parquet", "chembl_api"]]:
    """Реализует 3-уровневый каскад загрузки данных ChEMBL (Раздел 1.2 ARCHITECTURE.md):
    Уровень 1: Локальный кэш `/data/chembl/<target>.parquet` через DuckDB.
    Уровень 2: REST API `chembl_webresource_client` (опционально, при наличии сети).
    Уровень 3: Резервный слепок `/data/chembl/fallback_targets.parquet`.
    """
    cfg = tool_config or ToolConfig()
    chembl_dir = data_dir / "chembl"
    target_id = (target_chembl_id or "CHEMBL203").strip().upper()
    local_parquet = chembl_dir / f"{target_id}.parquet"

    conn = duckdb.connect(database=":memory:")

    # Уровень 1: Прямой локальный слепок мишени
    if local_parquet.is_file():
        rel = conn.sql(
            f"SELECT * FROM read_parquet('{local_parquet.as_posix()}') ORDER BY raw_row_id"
        )
        rows = rel.to_arrow_table().to_pylist()
        if rows:
            return rows, "chembl_parquet"

    # Уровень 2: Опциональный вызов ChEMBL REST API (только если локальный кэш отсутствует)
    try:
        from chembl_webresource_client.new_client import new_client  # type: ignore[import-not-found]

        activity_client = new_client.activity
        std_types = list(cfg.standard_types) or ["IC50", "Ki"]
        query_res = list(
            activity_client.filter(
                target_chembl_id=target_id,
                standard_type__in=std_types,
                standard_units="nM",
            ).only(
                [
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "standard_type",
                    "standard_relation",
                    "standard_value",
                    "standard_units",
                    "assay_chembl_id",
                    "assay_type",
                    "bao_format",
                    "target_chembl_id",
                    "target_organism",
                ]
            )[: int(cfg.chembl_api_limit)]
        )
        if query_res:
            api_rows: list[dict[str, Any]] = []
            for idx, item in enumerate(query_res, start=1):
                if (
                    not item.get("canonical_smiles")
                    or item.get("standard_value") is None
                ):
                    continue
                api_rows.append(
                    {
                        "raw_row_id": idx,
                        "molecule_chembl_id": str(
                            item.get("molecule_chembl_id") or f"{target_id}_{idx}"
                        ),
                        "raw_smiles": str(item["canonical_smiles"]),
                        "standard_type": str(item.get("standard_type") or "IC50"),
                        "standard_relation": str(item.get("standard_relation") or "="),
                        "standard_value": float(item["standard_value"]),
                        "standard_units": str(item.get("standard_units") or "nM"),
                        "assay_chembl_id": str(
                            item.get("assay_chembl_id") or "CHEMBL_API_ASSAY"
                        ),
                        "assay_type": str(item.get("assay_type") or "B"),
                        "bao_format": str(
                            item.get("bao_format") or settings.dominant_bao_default
                        ),
                        "target_chembl_id": target_id,
                        "target_organism": str(
                            item.get("target_organism") or "Homo sapiens"
                        ),
                    }
                )
            if api_rows:
                return api_rows, "chembl_api"
    except Exception as exc:  # noqa: BLE001
        errors_log.append(
            f"[Fallback Notice] Local cache '{local_parquet.name}' not found and ChEMBL API unavailable "
            f"({type(exc).__name__}); switching to Level 3 fallback_targets.parquet."
        )

    # Уровень 3: Детерминированный резервный слепок fallback_targets.parquet
    fallback_parquet = chembl_dir / "fallback_targets.parquet"
    if not fallback_parquet.is_file():
        # Если в кастомной тестовой директории нет fallback_targets.parquet, берем из дефолтного data_dir
        default_fallback = settings.chembl_cache_dir / "fallback_targets.parquet"
        if default_fallback.is_file():
            fallback_parquet = default_fallback
        else:
            raise FileNotFoundError(
                f"Neither local target cache '{local_parquet}' nor fallback '{fallback_parquet}' exists."
            )

    rel_target = conn.sql(
        f"SELECT * FROM read_parquet('{fallback_parquet.as_posix()}') "
        f"WHERE target_chembl_id = '{target_id}' ORDER BY raw_row_id"
    )
    fallback_rows = rel_target.to_arrow_table().to_pylist()
    if not fallback_rows:
        # Если запрошен неизвестный идентификатор мишени, возвращаем эталонный срез CHEMBL203 из резервного слепка
        rel_default = conn.sql(
            f"SELECT * FROM read_parquet('{fallback_parquet.as_posix()}') "
            "WHERE target_chembl_id = 'CHEMBL203' ORDER BY raw_row_id"
        )
        fallback_rows = rel_default.to_arrow_table().to_pylist()
        if not fallback_rows:
            fallback_rows = (
                conn.sql(f"SELECT * FROM read_parquet('{fallback_parquet.as_posix()}')")
                .to_arrow_table()
                .to_pylist()
            )
    return fallback_rows, "chembl_parquet"


def _load_custom_csv_rows(
    csv_path: Path,
    target_chembl_id: str | None,
    errors_log: list[str],
) -> list[dict[str, Any]]:
    """Парсит пользовательский CSV-файл лабораторной серии и приводит к схеме `raw_dataset`."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Uploaded custom CSV file does not exist: {csv_path}")

    raw_rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader, start=1):
            mol_id = (
                row.get("mol_id")
                or row.get("molecule_chembl_id")
                or row.get("id")
                or f"CSV_ROW_{row_idx:03d}"
            ).strip()
            raw_smiles = (
                row.get("SMILES")
                or row.get("smiles")
                or row.get("raw_smiles")
                or row.get("canonical_smiles")
                or ""
            ).strip()
            raw_activity = (
                row.get("IC50_nM")
                or row.get("standard_value")
                or row.get("IC50")
                or row.get("activity")
                or ""
            )
            explicit_rel = (
                row.get("standard_relation") or row.get("relation") or "="
            ).strip()
            units = (row.get("units") or row.get("standard_units") or "nM").strip()
            standard_type = (row.get("standard_type") or "IC50").strip()
            assay_chembl_id = (row.get("assay_chembl_id") or "CUSTOM_CSV").strip()
            assay_type = (row.get("assay_type") or "CUSTOM_WETLAB").strip()
            bao_format = (
                row.get("bao_format") or settings.dominant_bao_default
            ).strip()
            target_id = (
                row.get("target_chembl_id") or target_chembl_id or "CUSTOM_TARGET"
            ).strip()
            organism = (row.get("target_organism") or "Homo sapiens").strip()

            parsed_act = parse_activity_cell(
                raw_activity, default_relation=explicit_rel
            )
            if parsed_act is None:
                errors_log.append(
                    f"[row={row_idx}, mol_id={mol_id}] Unparseable or non-positive bioactivity value: '{raw_activity}'"
                )
                continue

            relation, numeric_val = parsed_act
            raw_rows.append(
                {
                    "raw_row_id": row_idx,
                    "molecule_chembl_id": mol_id,
                    "raw_smiles": raw_smiles,
                    "standard_type": standard_type,
                    "standard_relation": relation,
                    "standard_value": float(numeric_val),
                    "standard_units": units,
                    "assay_chembl_id": assay_chembl_id,
                    "assay_type": assay_type,
                    "bao_format": bao_format,
                    "target_chembl_id": target_id,
                    "target_organism": organism,
                }
            )

    return raw_rows


def _sql_in_list(values: list[str]) -> str:
    escaped = ["'" + v.replace("'", "''") + "'" for v in values if v]
    return ", ".join(escaped)


def curate_chembl_or_csv(
    session_dir: Path | str,
    constraints: QueryConstraints,
    uploaded_csv_path: Path | str | None = None,
    assay_filter_strict: bool = False,
    *,
    active_bao_filter: str | None = None,
    data_dir: Path | str | None = None,
    lead_neighborhood_tanimoto: float | None = None,
    tool_config: ToolConfig | None = None,
) -> tuple[ParquetPointer, ParquetPointer, DatasetCurationSummary]:
    """Выполняет сбор, курацию, стандартизацию, дедупликацию и фильтрацию данных биоактивности.

    Возвращает кортеж `(raw_dataset_uri, curated_dataset_uri, curation_summary)`.
    """
    cfg = tool_config or ToolConfig()
    resolved_session_dir = Path(session_dir).resolve()
    resolved_session_dir.mkdir(parents=True, exist_ok=True)
    resolved_data_dir = (
        Path(data_dir).resolve() if data_dir is not None else settings.resolved_data_dir
    )

    validation_errors_log: list[str] = []

    # 1. Загрузка сырых записей (CSV или каскад ChEMBL)
    is_csv_mode = (
        uploaded_csv_path is not None
        or constraints.scenario_type == "custom_csv_analysis"
    )
    if is_csv_mode:
        csv_file = (
            Path(uploaded_csv_path).resolve()
            if uploaded_csv_path is not None
            else (resolved_data_dir / "samples" / "custom_wetlab_series.csv")
        )
        raw_rows = _load_custom_csv_rows(
            csv_path=csv_file,
            target_chembl_id=constraints.target_chembl_id,
            errors_log=validation_errors_log,
        )
        source_type: Literal["chembl_parquet", "chembl_api", "custom_csv"] = (
            "custom_csv"
        )
    else:
        target_id = constraints.target_chembl_id or "CHEMBL203"
        raw_rows, source_type = _load_chembl_target_rows(
            target_chembl_id=target_id,
            data_dir=resolved_data_dir,
            errors_log=validation_errors_log,
            tool_config=cfg,
        )

    initial_rows_count = len(raw_rows)

    # 2. Сохранение первичной таблицы raw_dataset.parquet (Fat Pointer)
    raw_pointer = write_parquet_pointer(
        data=raw_rows,
        table_type="raw_dataset",
        session_dir=resolved_session_dir,
        filename="raw_dataset.parquet",
    )

    # 3. Определение доминирующего протокола BAO и типа измерения через DuckDB / Counter
    raw_bao_counter: Counter[str] = Counter(
        str(r.get("bao_format") or "BAO_0000357") for r in raw_rows
    )
    raw_stype_counter: Counter[str] = Counter(
        str(r.get("standard_type") or "IC50") for r in raw_rows
    )

    dominant_bao_format = (
        raw_bao_counter.most_common(1)[0][0]
        if raw_bao_counter
        else settings.dominant_bao_default
    )
    dominant_standard_type = (
        raw_stype_counter.most_common(1)[0][0] if raw_stype_counter else "IC50"
    )

    # 4. Фильтрация по однородности протоколов, форматам BAO, standard_types и relation_mode через DuckDB
    target_bao = active_bao_filter or dominant_bao_format
    if raw_pointer.row_count > 0:
        conn = duckdb.connect(database=":memory:")
        where_clauses: list[str] = []
        if assay_filter_strict:
            safe_bao = target_bao.replace("'", "''")
            safe_stype = dominant_standard_type.replace("'", "''")
            where_clauses.append(
                f"bao_format = '{safe_bao}' AND standard_type = '{safe_stype}'"
            )
        else:
            if cfg.allowed_bao_formats:
                bao_in = _sql_in_list(cfg.allowed_bao_formats)
                if bao_in:
                    where_clauses.append(f"bao_format IN ({bao_in})")
            if cfg.standard_types:
                stype_in = _sql_in_list(cfg.standard_types)
                if stype_in:
                    where_clauses.append(f"standard_type IN ({stype_in})")
        if cfg.relation_mode == "exact_only":
            where_clauses.append("COALESCE(TRIM(standard_relation), '=') IN ('=', '==')")

        where_sql = f"WHERE {' AND '.join(where_clauses)} " if where_clauses else ""
        filtered_sql = (
            f"SELECT * FROM read_parquet('{Path(raw_pointer.uri).as_posix()}') "
            f"{where_sql}"
            "ORDER BY raw_row_id"
        )
        active_rows = conn.sql(filtered_sql).to_arrow_table().to_pylist()
    else:
        active_rows = list(raw_rows)

    active_bao_distribution: dict[str, int] = dict(
        Counter(str(r.get("bao_format") or "BAO_0000357") for r in active_rows)
    )
    active_standard_types: set[str] = {
        str(r.get("standard_type") or "IC50") for r in active_rows
    }
    is_dataset_assay_homogeneous = (
        len(active_bao_distribution) <= 1 and len(active_standard_types) <= 1
    )

    # 5. Стандартизация структур RDKit, расчет pChEMBL и сбор групп для дедупликации
    invalid_smiles_dropped = 0
    salts_stripped_count = 0
    censored_values_count = 0

    # Группируем по (inchi_key, bao_format, standard_type) согласно Разделу 6.1.1 ARCHITECTURE.md
    dedup_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in active_rows:
        mol_id = str(row.get("molecule_chembl_id") or f"ROW_{row.get('raw_row_id', 0)}")
        raw_smi = str(row.get("raw_smiles") or "")
        row_label = f"row={row.get('raw_row_id')}, mol_id={mol_id}"

        std_res = standardize_molecule_detailed(
            raw_smiles=raw_smi,
            errors_log=validation_errors_log,
            row_label=row_label,
            tool_config=cfg,
        )
        if std_res is None:
            invalid_smiles_dropped += 1
            continue

        if std_res.salts_stripped:
            salts_stripped_count += 1

        rel_str = str(row.get("standard_relation") or "=")
        units_str = str(row.get("standard_units") or "nM")
        try:
            raw_val_float = float(row.get("standard_value") or 0.0)
            pchembl_val, is_censored = convert_to_pchembl(
                value=raw_val_float,
                units=units_str,
                relation=rel_str,
                tool_config=cfg,
            )
        except ValueError as exc:
            validation_errors_log.append(f"[{row_label}] {exc}")
            invalid_smiles_dropped += 1
            continue

        if cfg.relation_mode == "exact_only" and is_censored:
            validation_errors_log.append(
                f"[{row_label}] Censored measurement dropped (relation_mode='exact_only')."
            )
            continue

        if pchembl_val < cfg.pchembl_min_cutoff:
            validation_errors_log.append(
                f"[{row_label}] pChEMBL={pchembl_val:.2f} below pchembl_min_cutoff={cfg.pchembl_min_cutoff:.2f}."
            )
            continue

        if is_censored:
            censored_values_count += 1

        bao_code = str(row.get("bao_format") or dominant_bao_format)
        std_type = str(row.get("standard_type") or dominant_standard_type)
        assay_type = str(row.get("assay_type") or "B")

        group_key = (std_res.inchi_key, bao_code, std_type)
        dedup_groups[group_key].append(
            {
                "mol_id": mol_id,
                "canonical_smiles": std_res.canonical_smiles,
                "inchi_key": std_res.inchi_key,
                "scaffold_smiles": std_res.scaffold_smiles,
                "pchembl_value": pchembl_val,
                "is_censored": is_censored,
                "assay_type": assay_type,
                "bao_format": bao_code,
                "standard_type": std_type,
            }
        )

    # 6. Агрегация дубликатов и расчет сходства к лидеру (Сценарий 2)
    lead_fp = None
    if constraints.lead_smiles:
        lead_std = standardize_molecule_detailed(
            constraints.lead_smiles, tool_config=cfg
        )
        lead_smi_for_fp = (
            lead_std.canonical_smiles
            if lead_std is not None
            else constraints.lead_smiles
        )
        lead_mol = Chem.MolFromSmiles(lead_smi_for_fp)
        if lead_mol is not None:
            lead_fp = compute_mol_fingerprint(
                lead_mol,
                fp_type=cfg.fp_type,
                fp_radius=cfg.fp_radius,
                fp_nbits=cfg.fp_nbits,
            )

    def _aggregate_values(vals: list[float]) -> float:
        if cfg.duplicate_agg_method == "mean":
            return float(np.mean(vals))
        if cfg.duplicate_agg_method == "max":
            return float(np.max(vals))
        return float(np.median(vals))

    duplicates_merged_count = 0
    all_curated_rows: list[dict[str, Any]] = []

    for items in dedup_groups.values():
        rep_count = len(items)
        if rep_count > 1:
            duplicates_merged_count += rep_count - 1

        first = items[0]
        same_mol_id_group = len({str(x["mol_id"]) for x in items}) == 1
        uncensored_vals = [
            float(x["pchembl_value"]) for x in items if not x["is_censored"]
        ]
        if uncensored_vals:
            agg_pic50 = _aggregate_values(uncensored_vals)
            std_pic50 = (
                float(np.std(uncensored_vals, ddof=1))
                if len(uncensored_vals) > 1
                else 0.0
            )
            if (
                same_mol_id_group
                and len(uncensored_vals) > 1
                and std_pic50 > cfg.max_duplicate_std
            ):
                validation_errors_log.append(
                    f"[mol_id={first['mol_id']}] Replicate group dropped: std={std_pic50:.3f} > max_duplicate_std={cfg.max_duplicate_std:.3f}"
                )
                continue
            group_is_censored = False
        else:
            all_vals = [float(x["pchembl_value"]) for x in items]
            agg_pic50 = _aggregate_values(all_vals)
            std_pic50 = (
                float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0
            )
            if (
                same_mol_id_group
                and len(all_vals) > 1
                and std_pic50 > cfg.max_duplicate_std
            ):
                validation_errors_log.append(
                    f"[mol_id={first['mol_id']}] Censored replicate group dropped: std={std_pic50:.3f} > max_duplicate_std={cfg.max_duplicate_std:.3f}"
                )
                continue
            group_is_censored = True

        tanimoto_to_lead = 1.0
        if lead_fp is not None:
            mol_obj = Chem.MolFromSmiles(first["canonical_smiles"])
            if mol_obj is not None:
                mol_fp = compute_mol_fingerprint(
                    mol_obj,
                    fp_type=cfg.fp_type,
                    fp_radius=cfg.fp_radius,
                    fp_nbits=cfg.fp_nbits,
                )
                tanimoto_to_lead = round(
                    compute_fp_similarity(
                        lead_fp, mol_fp, metric=cfg.similarity_metric
                    ),
                    4,
                )

        all_curated_rows.append(
            {
                "mol_id": first["mol_id"],
                "canonical_smiles": first["canonical_smiles"],
                "inchi_key": first["inchi_key"],
                "scaffold_smiles": first["scaffold_smiles"],
                "pchembl_value": round(agg_pic50, 4),
                "pchembl_std": round(std_pic50, 4),
                "replicate_count": int(rep_count),
                "is_censored": bool(group_is_censored),
                "assay_type": first["assay_type"],
                "bao_format": first["bao_format"],
                "standard_type": first["standard_type"],
                "tanimoto_to_lead": float(tanimoto_to_lead),
            }
        )

    # Для Сценария 2 (при заданном lead_smiles) отбираем локальную окрестность
    if lead_fp is not None:
        thresh = (
            lead_neighborhood_tanimoto
            if lead_neighborhood_tanimoto is not None
            else (
                tool_config.neighborhood_tanimoto
                if tool_config is not None
                else settings.lead_neighborhood_tanimoto
            )
        )
        neighborhood_rows = [
            r for r in all_curated_rows if r["tanimoto_to_lead"] >= thresh
        ]
        curated_rows = (
            neighborhood_rows
            if len(neighborhood_rows) >= cfg.min_neighborhood_rows_fallback
            else all_curated_rows
        )
    else:
        curated_rows = all_curated_rows

    # 7. Сохранение таблицы curated_dataset.parquet (Fat Pointer)
    curated_pointer = write_parquet_pointer(
        data=curated_rows,
        table_type="curated_dataset",
        session_dir=resolved_session_dir,
        filename="curated_dataset.parquet",
    )

    # 8. Вычисление статистики топ-скаффолдов
    scaffold_groups: dict[str, list[float]] = defaultdict(list)
    for r in curated_rows:
        if not r["is_censored"]:
            scaffold_groups[r["scaffold_smiles"]].append(float(r["pchembl_value"]))

    top_scaffolds_sorted = sorted(
        (
            {
                "scaffold_smiles": scaf_smi,
                "count": len(vals),
                "mean_pic50": round(float(np.mean(vals)), 3),
            }
            for scaf_smi, vals in scaffold_groups.items()
        ),
        key=lambda item: (int(item["count"]), float(item["mean_pic50"])),
        reverse=True,
    )[: int(cfg.top_scaffolds_limit)]

    summary = DatasetCurationSummary(
        source_type=source_type,
        initial_rows_count=initial_rows_count,
        invalid_smiles_dropped=invalid_smiles_dropped,
        salts_stripped_count=salts_stripped_count,
        duplicates_merged_count=duplicates_merged_count,
        censored_values_count=censored_values_count,
        curated_molecules_count=len(curated_rows),
        dominant_bao_format=dominant_bao_format,
        dominant_standard_type=dominant_standard_type,
        bao_distribution=active_bao_distribution,
        is_dataset_assay_homogeneous=is_dataset_assay_homogeneous,
        top_scaffolds=top_scaffolds_sorted,
        validation_errors_log=validation_errors_log,
    )

    return (raw_pointer, curated_pointer, summary)
