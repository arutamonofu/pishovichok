"""Узел агента направленного молекулярного дизайна (Design Agent) в LangGraph (Слой 3).

Выполняет:
1. Выбор родительских структур `ParentSeed` из `curated_dataset.parquet`.
   Для Сценария 2 гарантированно включает исходную молекулу-лидер (гефитиниб).
2. Генерацию 100% валидных аналогов через реакции SMIRKS (`AllChem.ReactionFromSmarts`),
   замену фрагментов CReM и каталог биоизостеров с контролем сохранения скаффолда.
3. Учет базовых и корректирующих ограничений (включая директивы Критика во 2-й петле:
   `preferred_substituent_types`, `min_tpsa`, исключение галогенов).
4. Запись промежуточного пула кандидатов на диск сессии (`raw_candidates_it{n}.json`)
   для строгого соблюдения архитектурного принципа Fat Pointers (без перегрузки памяти графа).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pishovichok.core.generation import (
    generate_candidate_structures,
    select_parent_seeds_from_curated,
)
from pishovichok.state.graph_state import SARGraphState, resolve_tool_config
from pishovichok.state.schemas import TraceEvent


def _resolve_session_dir(workspace_dir: str | Path, session_id: str) -> Path:
    """Определяет абсолютный путь к рабочей директории сессии."""
    ws = Path(workspace_dir).resolve()
    session_dir = ws if ws.name == session_id else ws / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def design_node(state: SARGraphState) -> dict[str, Any]:
    """Узел Design Agent: комбинаторная генерация структур на базе правил SAR и биоизостеров."""
    start_time = time.perf_counter()
    session_dir = _resolve_session_dir(state["workspace_dir"], state["session_id"])
    curated_ptr = state["dataset_uri"]
    if curated_ptr is None:
        raise ValueError("Cannot execute design_node: dataset_uri is None.")

    design_iteration = int(state.get("design_iteration", 0)) + 1
    constraints = state["query_constraints"]
    tool_config = resolve_tool_config(state)

    # 1. Отбор родительских затравок из очищенной выборки
    parent_seeds = select_parent_seeds_from_curated(
        curated_pointer=curated_ptr,
        constraints=constraints,
        max_parents=tool_config.max_parent_seeds,
        tool_config=tool_config,
    )

    # 2. Направленная генерация сырых структур-кандидатов
    sar_rules = state.get("extracted_sar_rules", [])
    force_graph_ga = design_iteration > 1 or bool(state.get("critic_feedback"))
    raw_candidates = generate_candidate_structures(
        parent_molecules=parent_seeds,
        sar_rules=sar_rules,
        constraints=constraints,
        generation_iteration=design_iteration,
        max_candidates_pool=tool_config.max_candidates_pool,
        tool_config=tool_config,
        force_graph_ga=force_graph_ga,
    )

    smirks_count = sum(
        1 for c in raw_candidates if c.generation_origin == "mmp_smirks_rule"
    )
    crem_count = sum(
        1 for c in raw_candidates if c.generation_origin == "crem_fragment_replacement"
    )
    bioisostere_count = sum(
        1 for c in raw_candidates if c.generation_origin == "bioisostere_catalog"
    )
    graph_ga_count = sum(
        1 for c in raw_candidates if c.generation_origin == "graph_ga"
    )

    # 3. Сохранение сырого пула на диск сессии для изоляции памяти графа (Fat Pointer)
    raw_cand_path = session_dir / f"raw_candidates_it{design_iteration}.json"
    raw_payload = [c.model_dump(mode="json") for c in raw_candidates]
    raw_cand_path.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    step_idx = len(state.get("trace_log", []))
    output_text = (
        f"Generated {len(raw_candidates)} candidate structures from {len(parent_seeds)} parent seeds "
        f"(iteration={design_iteration}, SMIRKS={smirks_count}, CReM={crem_count}, GraphGA={graph_ga_count}, "
        f"preferred_types={constraints.preferred_substituent_types})."
    )

    ga_params_used: dict[str, object] = {
        "ga_population_size": tool_config.ga_population_size,
        "ga_generations": tool_config.ga_generations,
        "enable_crem_grow": tool_config.enable_crem_grow,
        "force_graph_ga": force_graph_ga,
        "smirks_candidates": smirks_count,
        "crem_candidates": crem_count,
        "bioisostere_candidates": bioisostere_count,
        "graph_ga_candidates": graph_ga_count,
    }

    trace_event = TraceEvent(
        step_index=step_idx,
        timestamp_iso=datetime.now(tz=UTC).isoformat(),
        agent_name="DesignAgent",
        phase=f"candidate_generation_it{design_iteration}",
        input_summary=(
            f"Iteration {design_iteration} | {len(sar_rules)} rules | "
            f"prefs={constraints.preferred_substituent_types}"
        ),
        action_taken=(
            f"Applied SMIRKS transformations, CReM grow/mutate and GraphGA BRICS crossover "
            f"(force_ga={force_graph_ga}, scaffold_locked={constraints.freeze_scaffold})"
        ),
        output_summary=output_text,
        verdict_or_routing="proceed_to_evaluation",
        fallback_triggered=None,
        parameters_used=ga_params_used,
        duration_ms=round(duration_ms, 2),
    )

    node_telemetry = dict(state.get("node_telemetry", {}))
    node_telemetry["design"] = {
        "active_params": {
            "generation_engine": tool_config.generation_engine,
            "crem_mode": tool_config.crem_mode,
            "crem_context_radius": tool_config.crem_context_radius,
            "crem_min_rel_atoms": tool_config.crem_min_rel_atoms,
            "crem_max_rel_atoms": tool_config.crem_max_rel_atoms,
            "crem_max_replacements": tool_config.crem_max_replacements,
            "freeze_murcko_core": tool_config.freeze_murcko_core,
            "protected_smarts": tool_config.protected_smarts,
            "modifiable_atom_indices": tool_config.modifiable_atom_indices,
            "max_parent_seeds": tool_config.max_parent_seeds,
            "max_candidates_pool": tool_config.max_candidates_pool,
            "use_bioisostere_catalog": tool_config.use_bioisostere_catalog,
            "ga_population_size": tool_config.ga_population_size,
            "ga_generations": tool_config.ga_generations,
            "enable_crem_grow": tool_config.enable_crem_grow,
            "force_graph_ga": force_graph_ga,
        },
        "stats": {
            "design_iteration": design_iteration,
            "parent_seeds_count": len(parent_seeds),
            "raw_candidates_count": len(raw_candidates),
            "smirks_candidates": smirks_count,
            "crem_candidates": crem_count,
            "bioisostere_candidates": bioisostere_count,
            "graph_ga_candidates": graph_ga_count,
            "duration_ms": round(duration_ms, 2),
        },
    }

    return {
        "design_iteration": design_iteration,
        "node_telemetry": node_telemetry,
        "last_executed_stage": "design",
        "trace_log": [trace_event],
    }
