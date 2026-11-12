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
from pishovichok.core.generation import select_parent_seeds_from_curated
from pishovichok.core.scoring import (
    evaluate_and_rank_candidates,
    evaluate_lead_reference,
    write_candidates_parquet,
)
from pishovichok.state.graph_state import SARGraphState
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
    )

    # 3. Выборка среза лучших кандидатов (топ-K)
    top_k = evaluated_pool[: constraints.top_k_candidates]

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
                curated_ptr, constraints, max_parents=1
            )
            if lead_seeds:
                lead_reference = evaluate_lead_reference(lead_seeds[0], constraints)

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    valid_count = sum(1 for c in enriched_top if c.passes_all_constraints)
    top_score = enriched_top[0].mpo_harrington_score if enriched_top else 0.0

    output_text = (
        f"Evaluated {len(evaluated_pool)} candidates; top-{len(enriched_top)} selected "
        f"(max Harrington D={top_score:.3f}). Constraints fully satisfied: {valid_count}/{len(enriched_top)}."
    )

    step_idx = len(state.get("trace_log", []))
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

    result_dict: dict[str, Any] = {
        "candidates_table_uri": main_cand_pointer,
        "top_candidates": enriched_top,
        "last_executed_stage": "evaluation",
        "trace_log": [trace_event],
    }
    if lead_reference is not None:
        result_dict["lead_reference_metrics"] = lead_reference

    return result_dict
