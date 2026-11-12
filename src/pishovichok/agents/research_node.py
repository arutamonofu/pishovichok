"""Узел агента исследования и сбора данных (Research Agent) в графе LangGraph (Слой 3).

Выполняет:
1. Семантический разбор пользовательского запроса через `LLMAdapter.parse_query_constraints`
   (при первом входе `last_executed_stage == 'init'`).
2. Курацию и стандартизацию входного набора данных через `curate_chembl_or_csv` с сохранением
   результатов в Parquet-таблицы сессии (`raw_dataset.parquet`, `curated_dataset.parquet`).
3. Поддержку строгой фильтрации по BAO-протоколу при вызове из Петли самокоррекции №1
   (`assay_filter_strict=True`, `active_bao_filter='BAO_0000357'`).
4. Формирование события Explainable Trace с фиксацией времени и эшелона LLM.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pishovichok.agents.llm_adapter import LLMAdapter
from pishovichok.core.curation import curate_chembl_or_csv
from pishovichok.state.graph_state import SARGraphState
from pishovichok.state.schemas import TraceEvent


def _resolve_session_dir(workspace_dir: str | Path, session_id: str) -> Path:
    """Определяет абсолютный путь к рабочей директории сессии."""
    ws = Path(workspace_dir).resolve()
    if ws.name == session_id:
        session_dir = ws
    else:
        session_dir = ws / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def research_node(state: SARGraphState) -> dict[str, Any]:
    """Узел Research Agent: парсинг запроса и химическая курация данных."""
    start_time = time.perf_counter()
    session_dir = _resolve_session_dir(state["workspace_dir"], state["session_id"])
    llm_adapter = LLMAdapter()

    # 1. Семантический разбор запроса при начальной инициализации
    fallback_triggered: str | None = None
    if state.get("last_executed_stage") == "init":
        parsed_constraints, fallback_triggered = llm_adapter.parse_query_constraints(
            user_query=state["user_query"],
            uploaded_csv_path=state.get("uploaded_csv_path"),
        )
        constraints = parsed_constraints
    else:
        constraints = state["query_constraints"]

    # 2. Определение параметров фильтрации протоколов (Петля №1)
    assay_filter_strict = bool(state.get("assay_filter_strict", False))
    active_bao_filter = state.get("active_bao_filter")
    is_loop1_refilter = assay_filter_strict or (
        state.get("data_refilter_iteration", 0) > 0
    )

    # 3. Детерминированная курация и стандартизация
    raw_ptr, curated_ptr, summary = curate_chembl_or_csv(
        session_dir=session_dir,
        constraints=constraints,
        uploaded_csv_path=state.get("uploaded_csv_path"),
        assay_filter_strict=assay_filter_strict,
        active_bao_filter=active_bao_filter,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    phase_name = (
        f"curation_refilter_loop1_{active_bao_filter}"
        if is_loop1_refilter
        else "initial_data_ingestion_and_curation"
    )

    action_text = (
        f"Strict assay filtering enforced for format '{active_bao_filter}'. "
        f"Standardized and deduplicated target bioactivity dataset."
        if is_loop1_refilter
        else "Parsed user constraints and performed initial standardization and SMILES curation."
    )

    output_text = (
        f"Curated {summary.curated_molecules_count} molecules "
        f"(dropped {summary.invalid_smiles_dropped} invalid, {summary.censored_values_count} censored, "
        f"{summary.duplicates_merged_count} duplicates). Dominant BAO: {summary.dominant_bao_format} "
        f"(homogeneous={summary.is_dataset_assay_homogeneous})."
    )

    step_idx = len(state.get("trace_log", []))
    trace_event = TraceEvent(
        step_index=step_idx,
        timestamp_iso=datetime.now(tz=UTC).isoformat(),
        agent_name="ResearchAgent",
        phase=phase_name,
        input_summary=(
            f"Query: '{state['user_query']}' | strict={assay_filter_strict} | bao={active_bao_filter}"
        ),
        action_taken=action_text,
        output_summary=output_text,
        verdict_or_routing="proceed_to_sar",
        fallback_triggered=fallback_triggered,
        duration_ms=round(duration_ms, 2),
    )

    return {
        "query_constraints": constraints,
        "raw_dataset_uri": raw_ptr,
        "dataset_uri": curated_ptr,
        "curation_summary": summary,
        "last_executed_stage": "research",
        "trace_log": [trace_event],
    }
