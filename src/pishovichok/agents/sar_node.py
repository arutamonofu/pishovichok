"""Узел агента анализа зависимостей «структура–активность» (SAR Agent) в LangGraph (Слой 3).

Выполняет:
1. Построение сети парных молекулярных соответствий (`rdMMPA`) поверх `curated_dataset.parquet`.
2. Расчет индекса ландшафта активности SALI и выявление критических перепадов (Activity Cliffs).
3. Извлечение ранжированных правил трансформации `SARRule` со статистикой ΔpIC50 и флагом
   однородности экспериментальных протоколов `assay_homogeneous`.
4. Для лабораторных серий CSV (`custom_csv_analysis`) — автоматическое контекстное обогащение
   фоном ChEMBL при малом объеме выборки (`extract_mmp_with_background_fallback`).
5. Передачу результатов Критику через Fat Pointer `mmp_pairs.parquet` и список `SARRule`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pishovichok.core.sar_analysis import (
    extract_mmp_and_cliffs,
    extract_mmp_with_background_fallback,
)
from pishovichok.state.graph_state import SARGraphState
from pishovichok.state.schemas import TraceEvent


def _resolve_session_dir(workspace_dir: str | Path, session_id: str) -> Path:
    """Определяет абсолютный путь к рабочей директории сессии."""
    ws = Path(workspace_dir).resolve()
    session_dir = ws if ws.name == session_id else ws / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def sar_node(state: SARGraphState) -> dict[str, Any]:
    """Узел SAR Agent: извлечение правил трансформации и детекция Activity Cliffs."""
    start_time = time.perf_counter()
    session_dir = _resolve_session_dir(state["workspace_dir"], state["session_id"])
    curated_ptr = state["dataset_uri"]
    if curated_ptr is None:
        raise ValueError(
            "Cannot execute sar_node: dataset_uri is None. research_node must run first."
        )

    constraints = state["query_constraints"]

    # Для Сценария 2 фиксируем скаффолд молекулы-лидера
    fixed_scaffold: str | None = None
    if constraints.scenario_type == "lead_optimization" and constraints.lead_smiles:
        fixed_scaffold = constraints.lead_smiles

    # Выбор алгоритма в зависимости от типа сценария
    if constraints.scenario_type == "custom_csv_analysis":
        bg_target = constraints.target_chembl_id or "CHEMBL203"
        mmp_ptr, sar_rules, cliffs, enriched = extract_mmp_with_background_fallback(
            curated_pointer=curated_ptr,
            session_dir=session_dir,
            fixed_scaffold_smiles=fixed_scaffold,
            min_support_pairs=2,
            background_target_id=bg_target,
        )
    else:
        mmp_ptr, sar_rules, cliffs = extract_mmp_and_cliffs(
            curated_pointer=curated_ptr,
            session_dir=session_dir,
            fixed_scaffold_smiles=fixed_scaffold,
            min_support_pairs=2,
        )
        enriched = False

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    homo_count = sum(1 for r in sar_rules if r.assay_homogeneous)
    output_text = (
        f"Extracted {len(sar_rules)} SAR rules and {len(cliffs)} Activity Cliffs. "
        f"Homogeneous rules: {homo_count}/{len(sar_rules)} (background_enriched={enriched})."
    )

    step_idx = len(state.get("trace_log", []))
    trace_event = TraceEvent(
        step_index=step_idx,
        timestamp_iso=datetime.now(tz=UTC).isoformat(),
        agent_name="SARAgent",
        phase="sar_mmp_and_cliffs_extraction",
        input_summary=(
            f"Curated dataset ({curated_ptr.row_count} rows) | fixed_scaffold={bool(fixed_scaffold)}"
        ),
        action_taken=(
            f"Built rdMMPA network and SALI index (scaffold_locked={bool(fixed_scaffold)}, "
            f"enriched={enriched})"
        ),
        output_summary=output_text,
        verdict_or_routing="proceed_to_critic",
        fallback_triggered=None,
        duration_ms=round(duration_ms, 2),
    )

    return {
        "mmp_table_uri": mmp_ptr,
        "extracted_sar_rules": sar_rules,
        "detected_activity_cliffs": cliffs,
        "enriched_with_chembl_background": enriched,
        "last_executed_stage": "sar",
        "trace_log": [trace_event],
    }
