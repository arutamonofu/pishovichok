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
from pishovichok.state.graph_state import (
    SARGraphState,
    resolve_tool_config,
    update_param_sources,
)
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

    tool_config = resolve_tool_config(state)
    param_sources = dict(state.get("param_sources", {}))
    nl_overrides_applied: list[str] = []

    # 1. Семантический разбор запроса и параметров ToolConfig при начальной инициализации
    fallback_triggered: str | None = None
    if state.get("last_executed_stage") == "init":
        parsed_constraints, nl_overrides, fallback_triggered = (
            llm_adapter.parse_query_with_tool_config(
                user_query=state["user_query"],
                uploaded_csv_path=state.get("uploaded_csv_path"),
            )
        )
        constraints = parsed_constraints
        if nl_overrides:
            tool_config, nl_overrides_applied = tool_config.apply_overrides(nl_overrides)
            param_sources = update_param_sources(
                param_sources,
                nl_overrides_applied,
                "nl_query",
            )
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
        tool_config=tool_config,
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

    homog_ru = "да" if summary.is_dataset_assay_homogeneous else "нет"
    output_text = (
        f"Отобрано {summary.curated_molecules_count} молекул "
        f"(отсеяно: {summary.invalid_smiles_dropped} невал., {summary.censored_values_count} цензур., "
        f"{summary.duplicates_merged_count} дубл.). Протокол BAO: {summary.dominant_bao_format} "
        f"(однородный: {homog_ru})."
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

    node_telemetry = dict(state.get("node_telemetry", {}))
    node_telemetry["research"] = {
        "active_params": {
            "allowed_bao_formats": list(tool_config.allowed_bao_formats),
            "standard_types": list(tool_config.standard_types),
            "relation_mode": tool_config.relation_mode,
            "pchembl_min_cutoff": tool_config.pchembl_min_cutoff,
            "duplicate_agg_method": tool_config.duplicate_agg_method,
            "max_duplicate_std": tool_config.max_duplicate_std,
            "strip_salts": tool_config.strip_salts,
            "neutralize_charges": tool_config.neutralize_charges,
            "canonicalize_tautomers": tool_config.canonicalize_tautomers,
            "keep_chirality": tool_config.keep_chirality,
            "scaffold_mode": tool_config.scaffold_mode,
            "fp_type": tool_config.fp_type,
            "neighborhood_tanimoto": tool_config.neighborhood_tanimoto,
            "assay_filter_strict": assay_filter_strict,
            "active_bao_filter": active_bao_filter,
        },
        "stats": {
            "nl_overrides_applied": nl_overrides_applied,
            "curated_molecules_count": summary.curated_molecules_count,
            "dominant_bao_format": summary.dominant_bao_format,
            "is_dataset_assay_homogeneous": summary.is_dataset_assay_homogeneous,
            "duration_ms": round(duration_ms, 2),
        },
    }

    return {
        "query_constraints": constraints,
        "tool_config": tool_config,
        "param_sources": param_sources,
        "node_telemetry": node_telemetry,
        "raw_dataset_uri": raw_ptr,
        "dataset_uri": curated_ptr,
        "curation_summary": summary,
        "last_executed_stage": "research",
        "trace_log": [trace_event],
    }
