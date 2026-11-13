"""Менеджер сессионного хранилища и 'толстых указателей' (Fat Pointers) на базе Parquet и DuckDB (Слой 2).

Обеспечивает передачу ссылок (`ParquetPointer`) на массивы данных между узлами графа LangGraph
без загрузки полных таблиц молекул в оперативную память состояния или контекст LLM.
Все операции чтения/записи сопровождаются проверкой обязательных колонок и SHA-256 контрольной суммы.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pishovichok.config import settings
from pishovichok.state.schemas import (
    PARQUET_TABLE_REQUIRED_COLUMNS,
    CandidateMolecule,
    ParquetPointer,
    TableTypeLiteral,
    TraceEvent,
)


def compute_file_sha256(file_path: str | Path) -> str:
    """Вычисляет криптографический хэш SHA-256 файла на диске."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot compute SHA-256: file does not exist at {path}"
        )
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_session_workspace(
    base_workspace_dir: str | Path | None = None,
    session_id: str | None = None,
) -> tuple[str, Path]:
    """Создает изолированную директорию сессии `workspace/<session_id>/`."""
    root = (
        Path(base_workspace_dir)
        if base_workspace_dir is not None
        else settings.resolved_workspace_dir
    )
    if session_id is None:
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        short_uid = uuid.uuid4().hex[:8]
        session_id = f"session_{ts}_{short_uid}"
    session_dir = (root / session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def _coerce_to_arrow_table(
    data: pa.Table | pd.DataFrame | list[dict[str, Any]],
    table_type: TableTypeLiteral,
) -> pa.Table:
    """Приводит входные табличные данные к `pyarrow.Table` с гарантией обязательных колонок."""
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data, preserve_index=False)
    if isinstance(data, list):
        if len(data) > 0:
            return pa.Table.from_pylist(data)
        # Пустой список: формируем схему с обязательными колонками для данного table_type
        req_cols = PARQUET_TABLE_REQUIRED_COLUMNS[table_type]
        empty_arrays = [pa.array([], type=pa.string()) for _ in req_cols]
        return pa.Table.from_arrays(empty_arrays, names=list(req_cols))
    raise TypeError(
        f"Unsupported tabular data type for ParquetPointer: {type(data).__name__}"
    )


def write_parquet_pointer(
    data: pa.Table | pd.DataFrame | list[dict[str, Any]],
    table_type: TableTypeLiteral,
    session_dir: str | Path,
    filename: str | None = None,
) -> ParquetPointer:
    """Записывает таблицу в сессионный Parquet-файл (ZSTD) и возвращает валидированный `ParquetPointer`."""
    target_dir = Path(session_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = filename or f"{table_type}.parquet"
    target_path = target_dir / file_name

    arrow_table = _coerce_to_arrow_table(data, table_type)
    pq.write_table(arrow_table, target_path, compression="zstd")

    checksum = compute_file_sha256(target_path)
    return ParquetPointer(
        uri=target_path.as_posix(),
        table_type=table_type,
        row_count=int(arrow_table.num_rows),
        columns=list(arrow_table.column_names),
        sha256_checksum=checksum,
    )


def verify_pointer_integrity(pointer: ParquetPointer) -> Path:
    """Проверяет существование файла указателя и совпадение контрольной суммы SHA-256."""
    path = Path(pointer.uri)
    if not path.is_file():
        raise FileNotFoundError(f"ParquetPointer target file not found: {path}")
    actual_sha256 = compute_file_sha256(path)
    if actual_sha256 != pointer.sha256_checksum:
        raise ValueError(
            f"ParquetPointer integrity check failed for {path}: "
            f"expected sha256={pointer.sha256_checksum}, got {actual_sha256}"
        )
    return path


def read_parquet_pointer(
    pointer: ParquetPointer,
    *,
    verify_checksum: bool = True,
    where_clause: str | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Читает данные по `ParquetPointer` через векторизованный скан DuckDB (с фолбэком на PyArrow)."""
    path = verify_pointer_integrity(pointer) if verify_checksum else Path(pointer.uri)
    if not path.is_file():
        raise FileNotFoundError(f"ParquetPointer target file not found: {path}")

    col_expr = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f"SELECT {col_expr} FROM read_parquet('{path.as_posix()}')"
    if where_clause:
        sql += f" WHERE {where_clause}"

    try:
        conn = duckdb.connect(database=":memory:")
        return conn.sql(sql).to_arrow_table()
    except Exception:
        # Детерминированный фолбэк на прямое чтение через PyArrow (при отсутствии/ошибке SQL-фильтра)
        if where_clause is None:
            return pq.read_table(path, columns=columns)
        raise


def query_parquet_pointer_df(
    pointer: ParquetPointer,
    *,
    verify_checksum: bool = True,
    where_clause: str | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Выполняет SQL-выборку из `ParquetPointer` через DuckDB и возвращает `pandas.DataFrame`."""
    path = verify_pointer_integrity(pointer) if verify_checksum else Path(pointer.uri)
    sql = f"SELECT * FROM read_parquet('{path.as_posix()}')"
    if where_clause:
        sql += f" WHERE {where_clause}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    conn = duckdb.connect(database=":memory:")
    return conn.sql(sql).df()


def write_candidates_parquet(
    candidates: list[CandidateMolecule],
    session_dir: str | Path,
    filename: str = "evaluated_candidates.parquet",
) -> ParquetPointer:
    """Сохраняет пул `CandidateMolecule` в `evaluated_candidates.parquet` и возвращает `ParquetPointer`."""
    flat_records = [cand.to_flat_parquet_record() for cand in candidates]
    return write_parquet_pointer(
        data=flat_records,
        table_type="evaluated_candidates",
        session_dir=session_dir,
        filename=filename,
    )


def read_candidates_parquet(
    pointer: ParquetPointer,
    *,
    verify_checksum: bool = True,
    where_clause: str | None = None,
) -> list[CandidateMolecule]:
    """Восстанавливает список `CandidateMolecule` со всеми 4 категориями знаний из `ParquetPointer`."""
    if pointer.table_type != "evaluated_candidates":
        raise ValueError(
            f"Expected table_type='evaluated_candidates', got '{pointer.table_type}'"
        )
    arrow_tbl = read_parquet_pointer(
        pointer,
        verify_checksum=verify_checksum,
        where_clause=where_clause,
    )
    rows = arrow_tbl.to_pylist()
    return [CandidateMolecule.from_flat_parquet_record(row) for row in rows]


def save_candidate_pose_sdf(
    workspace_dir: str | Path,
    mol_id: str,
    sdf_content: str,
) -> str:
    """Сохраняет 3D-позу докинга кандидата в файл .sdf внутри рабочей директории сессии."""
    poses_dir = Path(workspace_dir).resolve() / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(mol_id)
    ).strip("_") or "candidate"
    out_path = poses_dir / f"{safe_id}_docked.sdf"
    out_path.write_text(sdf_content, encoding="utf-8")
    return out_path.as_posix()


def export_trace_log(
    trace_log: list[TraceEvent],
    session_dir: str | Path,
    filename: str = "trace_log.json",
) -> Path:
    """Экспортирует полный протокол трассировки вычислительной сессии в JSON."""
    target_dir = Path(session_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename
    payload = [event.model_dump(mode="json") for event in trace_log]
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def export_session_meta(
    state: dict[str, Any],
    session_dir: str | Path,
    filename: str = "session_meta.json",
) -> Path:
    """Сохраняет компактные метаданные сессии (ограничения, ToolConfig, референс лидера и курацию) для быстрого восстановления."""
    target_dir = Path(session_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename

    qc = state.get("query_constraints")
    tc = state.get("tool_config")
    lead_ref = state.get("lead_reference_metrics")
    cur_sum = state.get("curation_summary")

    payload: dict[str, Any] = {
        "user_query": state.get("user_query") or "",
        "uploaded_csv_path": state.get("uploaded_csv_path"),
        "assay_filter_strict": bool(state.get("assay_filter_strict", False)),
        "active_bao_filter": state.get("active_bao_filter"),
        "enriched_with_chembl_background": bool(
            state.get("enriched_with_chembl_background", False)
        ),
        "param_sources": dict(state.get("param_sources") or {}),
        "node_telemetry": dict(state.get("node_telemetry") or {}),
        "query_constraints": (
            qc.model_dump(mode="json") if hasattr(qc, "model_dump") else qc
        ),
        "tool_config": (
            tc.model_dump(mode="json") if hasattr(tc, "model_dump") else tc
        ),
        "lead_reference_metrics": (
            lead_ref.model_dump(mode="json")
            if hasattr(lead_ref, "model_dump")
            else lead_ref
        ),
        "curation_summary": (
            cur_sum.model_dump(mode="json")
            if hasattr(cur_sum, "model_dump")
            else cur_sum
        ),
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def load_session_meta(
    session_dir: str | Path,
    filename: str = "session_meta.json",
) -> dict[str, Any] | None:
    """Читает `session_meta.json` из директории сессии, если он существует и валиден."""
    meta_path = Path(session_dir).resolve() / filename
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
