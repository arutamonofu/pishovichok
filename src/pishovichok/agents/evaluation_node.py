"""Узел агента физико-химической и ADMET-оценки (Evaluation Agent) в LangGraph (Слой 3).

Выполняет:
1. Загрузку сырых кандидатов текущей итерации из сессионного файла `raw_candidates_it{n}.json`.
2. Расчет детерминированных физико-химических дескрипторов RDKit, структурных фильтров
   PAINS/Brenk/Glaxo и ретросинтетической фрагментации по связям BRICS.
3. Прогон через каскад моделей ADMET (ADMET-AI DMPNN -> CatBoost -> аналитический QSAR).
4. Расчет обобщенной функции желательности Харрингтона (MPO) с жестким вето при алертах.
5. Семантическое обогащение научных гипотез Категории 4 через `LLMAdapter.generate_hypothesis_rationales`.
6. Оценку референсной карточки исходного лидера (гефитиниба) для расчета дельт в Сценарии 2.
7. Сохранение полной таблицы в Fat Pointer `evaluated_candidates.parquet` и передачу топа Критику.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pishovichok.agents.llm_adapter import LLMAdapter
from pishovichok.core.evaluation import evaluate_docking_batch
from pishovichok.core.generation import select_parent_seeds_from_curated
from pishovichok.core.scoring import (
    evaluate_and_rank_candidates,
    evaluate_lead_reference,
    write_candidates_parquet,
)
from pishovichok.state.graph_state import SARGraphState, resolve_tool_config
from pishovichok.state.schemas import (
    CandidateMolecule,
    GeneratedRawCandidate,
    TraceEvent,
)


def _resolve_session_dir(workspace_dir: str | Path, session_id: str) -> Path:
    """Определяет абсолютный путь к рабочей директории сессии."""
    ws = Path(workspace_dir).resolve()
    session_dir = ws if ws.name == session_id else ws / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def evaluation_node(state: SARGraphState) -> dict[str, Any]:
    """Узел Evaluation Agent: физико-химический и токсикологический расчет, скоринг и BRICS-декомпозиция."""
    start_time = time.perf_counter()
    session_dir = _resolve_session_dir(state["workspace_dir"], state["session_id"])
    iteration = int(state.get("design_iteration", 1))
    constraints = state["query_constraints"]
    tool_config = resolve_tool_config(state)
    param_sources = state.get("param_sources", {})
    llm_adapter = LLMAdapter()

    # 1. Чтение сгенерированного сырого пула текущей итерации
    raw_cand_path = session_dir / f"raw_candidates_it{iteration}.json"
    if not raw_cand_path.is_file():
        raise FileNotFoundError(
            f"Missing raw candidates file for iteration {iteration} at {raw_cand_path}"
        )

    raw_data = json.loads(raw_cand_path.read_text(encoding="utf-8"))
    raw_candidates = [GeneratedRawCandidate.model_validate(c) for c in raw_data]

    # 2. Детерминированный расчет, ADMET и многопараметрический скоринг Харрингтона
    _cand_pointer, evaluated_pool = evaluate_and_rank_candidates(
        raw_candidates=raw_candidates,
        constraints=constraints,
        session_dir=session_dir,
        filename=f"evaluated_candidates_it{iteration}.parquet",
        tool_config=tool_config,
    )

    # 3. Выборка среза лучших кандидатов (топ-K)
    top_k_limit = (
        tool_config.top_k_output
        if param_sources.get("top_k_output") not in (None, "default")
        or tool_config.top_k_output != 5
        else constraints.top_k_candidates
    )
    top_k = evaluated_pool[:top_k_limit]

    # 3.5. Управляемый этап физического 3D-докинга
    docking_mode = state.get("docking_mode") or tool_config.docking_mode
    docking_exhaustiveness = int(
        state.get("docking_exhaustiveness") or tool_config.docking_exhaustiveness
    )
    docking_top_k = int(state.get("docking_top_k") or tool_config.docking_top_k)
    custom_pdb_id = state.get("custom_pdb_id") or tool_config.custom_pdb_id
    custom_box_center = state.get("custom_box_center") or tool_config.custom_box_center
    target_id = constraints.target_chembl_id

    docking_events: list[TraceEvent] = []
    docking_summary: dict[str, Any]
    step_idx_base = len(state.get("trace_log", []))

    if docking_mode == "auto_top_k":
        dock_start = time.perf_counter()
        docked_top, docking_summary = evaluate_docking_batch(
            candidates=top_k,
            target_id=target_id,
            top_k=docking_top_k,
            exhaustiveness=docking_exhaustiveness,
            custom_pdb_id=custom_pdb_id,
            custom_box_center=custom_box_center,
            workspace_dir=session_dir,
            constraints=constraints,
            tool_config=tool_config,
        )
        top_k = sorted(
            docked_top,
            key=lambda c: (
                bool(c.passes_all_constraints),
                float(c.mpo_harrington_score),
                float(c.pred_admet.predicted_pic50),
                -float(c.calc_rdkit.clogp),
            ),
            reverse=True,
        )
        dock_duration = round((time.perf_counter() - dock_start) * 1000.0, 2)
        dock_status = docking_summary.get("status", "SKIPPED")
        best_dg = docking_summary.get("best_score_kcal")
        dg_text = f"{best_dg:.2f} ккал/моль" if best_dg is not None else "Н/Д"
        _st_ru = {
            "SUCCESS": "Успешно",
            "SKIPPED": "Пропущено",
            "FAILED": "Ошибка",
            "NO_RECEPTOR": "Рецептор не найден",
        }.get(str(dock_status), str(dock_status))
        _src_raw = docking_summary.get("source")
        _src_ru = {
            "DEMO_CACHE": "демо-кэш",
            "RCSB_LIVE": "RCSB PDB",
            "RCSB_CACHE": "RCSB кэш",
            "LOCAL_PDB": "локальный PDB",
        }.get(str(_src_raw or ""), str(_src_raw or "Н/Д"))
        _pdb_val = docking_summary.get("pdb_id") or "Н/Д"
        docking_events.append(
            TraceEvent(
                step_index=step_idx_base,
                timestamp_iso=datetime.now(tz=UTC).isoformat(),
                agent_name="EvaluationAgent",
                phase=f"docking_3d_it{iteration}",
                input_summary=(
                    f"mode=auto_top_k | target={target_id or 'N/A'} | "
                    f"custom_pdb={custom_pdb_id or 'auto'} | top_k={docking_top_k} | exhaustiveness={docking_exhaustiveness}"
                ),
                action_taken=(
                    "Подготовлен карман рецептора (кэш/RCSB PDB), выполнен 3D-докинг и профилирование взаимодействий."
                ),
                output_summary=(
                    f"Статус: {_st_ru} | PDB: {_pdb_val} ({_src_ru}) | "
                    f"состыковано: {docking_summary.get('success_count', 0)}/{docking_summary.get('docked_count', 0)} | "
                    f"лучшая ΔG = {dg_text}."
                ),
                verdict_or_routing="docking_completed" if dock_status == "SUCCESS" else "docking_skipped",
                fallback_triggered=None,
                duration_ms=dock_duration,
            )
        )
    else:
        skip_reason = (
            "3D-докинг отложен пользователем (режим 'manual_only'): доступен ручной запуск из таблицы или инспектора."
            if docking_mode == "manual_only"
            else "3D-докинг отключён перед запуском (режим 'skip')."
        )
        docking_summary = {
            "status": "SKIPPED",
            "docked_count": 0,
            "success_count": 0,
            "pdb_id": custom_pdb_id,
            "source": None,
            "best_score_kcal": None,
            "message": skip_reason,
        }
        docking_events.append(
            TraceEvent(
                step_index=step_idx_base,
                timestamp_iso=datetime.now(tz=UTC).isoformat(),
                agent_name="EvaluationAgent",
                phase=f"docking_3d_it{iteration}",
                input_summary=f"mode={docking_mode} | target={target_id or 'N/A'}",
                action_taken="Skipped automatic 3D docking according to docking_mode.",
                output_summary=skip_reason,
                verdict_or_routing="docking_skipped",
                fallback_triggered=None,
                duration_ms=0.0,
            )
        )

    # 4. Формирование детализированных текстовых обоснований гипотез через LLM-адаптер
    id_to_text, fallback_hypotheses = llm_adapter.generate_hypothesis_rationales(top_k)
    enriched_top: list[CandidateMolecule] = []
    for c in top_k:
        new_text = id_to_text.get(c.candidate_id)
        if new_text and new_text != c.hypothesis_rationale.text_summary:
            updated_rationale = c.hypothesis_rationale.model_copy(
                update={"text_summary": new_text}
            )
            enriched_top.append(
                c.model_copy(update={"hypothesis_rationale": updated_rationale})
            )
        else:
            enriched_top.append(c)

    # Обновляем основной Parquet-файл evaluated_candidates.parquet с финальными кандидатами
    id_to_enriched = {c.candidate_id: c for c in enriched_top}
    final_pool: list[CandidateMolecule] = [
        id_to_enriched.get(c.candidate_id, c) for c in evaluated_pool
    ]
    main_cand_pointer = write_candidates_parquet(
        final_pool, session_dir, "evaluated_candidates.parquet"
    )

    # 5. Оценка референсной карточки лидера (для Сценария 2)
    lead_reference = state.get("lead_reference_metrics")
    if (
        lead_reference is None
        and constraints.scenario_type == "lead_optimization"
        and constraints.lead_smiles
    ):
        curated_ptr = state["dataset_uri"]
        if curated_ptr is not None:
            lead_seeds = select_parent_seeds_from_curated(
                curated_ptr, constraints, max_parents=1, tool_config=tool_config
            )
            if lead_seeds:
                lead_reference = evaluate_lead_reference(
                    lead_seeds[0], constraints, tool_config=tool_config
                )

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    valid_count = sum(1 for c in enriched_top if c.passes_all_constraints)
    top_score = enriched_top[0].mpo_harrington_score if enriched_top else 0.0

    output_text = (
        f"Оценено кандидатов: {len(evaluated_pool)}; выбрано топ-{len(enriched_top)} "
        f"(макс. D Харрингтона = {top_score:.3f}). Ограничения соблюдены: {valid_count}/{len(enriched_top)}."
    )

    step_idx = len(state.get("trace_log", [])) + len(docking_events)
    trace_event = TraceEvent(
        step_index=step_idx,
        timestamp_iso=datetime.now(tz=UTC).isoformat(),
        agent_name="EvaluationAgent",
        phase=f"evaluation_and_admet_it{iteration}",
        input_summary=(
            f"{len(raw_candidates)} raw candidates | max_logp={constraints.max_logp} | "
            f"max_sa={constraints.max_sa_score}"
        ),
        action_taken=(
            "Calculated RDKit 2D metrics, PAINS/Brenk alerts, BRICS precursors, ADMET probabilities, "
            "and Harrington MPO desirability index."
        ),
        output_summary=output_text,
        verdict_or_routing="proceed_to_critic",
        fallback_triggered=fallback_hypotheses,
        duration_ms=round(duration_ms, 2),
    )

    node_telemetry = dict(state.get("node_telemetry", {}))
    node_telemetry["evaluation"] = {
        "active_params": {
            "active_filter_catalogs": list(tool_config.active_filter_catalogs),
            "custom_ban_smarts": list(tool_config.custom_ban_smarts),
            "brics_min_fragment_size": tool_config.brics_min_fragment_size,
            "brics_only_leaf_nodes": tool_config.brics_only_leaf_nodes,
            "admet_backend": tool_config.admet_backend,
            "cns_profile_mode": tool_config.cns_profile_mode,
            "alert_penalty_mode": tool_config.alert_penalty_mode,
            "alert_soft_multiplier": tool_config.alert_soft_multiplier,
            "mpo_weights": dict(tool_config.mpo_weights),
            "extra_rdkit_descriptors": list(tool_config.extra_rdkit_descriptors),
            "top_k_output": top_k_limit,
            "docking_mode": docking_mode,
            "docking_exhaustiveness": docking_exhaustiveness,
            "docking_top_k": docking_top_k,
        },
        "stats": {
            "iteration": iteration,
            "raw_candidates_count": len(raw_candidates),
            "evaluated_pool_count": len(evaluated_pool),
            "top_k_count": len(enriched_top),
            "valid_top_k_count": valid_count,
            "max_harrington_d": round(top_score, 4),
            "docking_status": docking_summary.get("status"),
            "docking_pdb_id": docking_summary.get("pdb_id"),
            "docking_best_dg_kcal": docking_summary.get("best_score_kcal"),
            "duration_ms": round(duration_ms, 2),
        },
    }

    result_dict: dict[str, Any] = {
        "candidates_table_uri": main_cand_pointer,
        "top_candidates": enriched_top,
        "node_telemetry": node_telemetry,
        "last_executed_stage": "evaluation",
        "trace_log": [*docking_events, trace_event],
    }
    if lead_reference is not None:
        result_dict["lead_reference_metrics"] = lead_reference

    return result_dict
