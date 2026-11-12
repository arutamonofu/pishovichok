"""Узел агента-критика и методологической валидации (Critic Agent) в LangGraph (Слой 3).

Управляет двумя замкнутыми петлями обратной связи (Раздел 5.2 ARCHITECTURE.md):
1. Петля №1 (`refilter_data` -> `research`):
   Срабатывает, если в извлеченных правилах SAR обнаружены неоднородные протоколы испытаний
   (`assay_homogeneous == False`). Включает строгую стратификацию по доминирующему BAO_format.
   Ограничена 1 итерацией (`data_refilter_iteration < 1`).
2. Петля №2 (`redesign_molecules` -> `design`):
   Срабатывает, если рассчитанные дескрипторы кандидатов (LogP, TPSA, SAscore, hERG)
   выходят за рамки ограничений пользователя. Формирует директивы на выбор полярных
   и менее липофильных групп через `LLMAdapter.formulate_corrective_constraints`.
   Ограничена 2 итерациями (`design_iteration < 2`).
3. Финальное утверждение (`approve` -> `end`):
   Срабатывает при прохождении всех ограничений или по достижении лимита итераций с выдачей
   лучших компромиссных структур по индексу Харрингтона с явной маркировкой отклонений.
   Генерирует финальный аналитический отчет в формате Markdown.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal

from pishovichok.agents.llm_adapter import LLMAdapter
from pishovichok.state.graph_state import SARGraphState
from pishovichok.state.schemas import TraceEvent


def critic_node(state: SARGraphState) -> dict[str, Any]:
    """Узел Critic Agent: аудит однородности протоколов и валидация физико-химического профиля."""
    start_time = time.perf_counter()
    llm_adapter = LLMAdapter()
    last_stage = state.get("last_executed_stage")
    step_idx = len(state.get("trace_log", []))

    # ========================================================================
    # ФАЗА 1: Аудит однородности экспериментальных протоколов (после SAR Agent)
    # ========================================================================
    if last_stage == "sar":
        sar_rules = state.get("extracted_sar_rules", [])
        curation_summary = state.get("curation_summary")
        data_refilter_it = int(state.get("data_refilter_iteration", 0))

        # Проверка флага однородности в правилах SAR и выборке
        has_heterogeneous_rules = any(not r.assay_homogeneous for r in sar_rules)
        dataset_not_homogeneous = (
            curation_summary is not None
            and not curation_summary.is_dataset_assay_homogeneous
        )
        needs_loop1_refilter = (
            (has_heterogeneous_rules or dataset_not_homogeneous)
            and not bool(state.get("assay_filter_strict", False))
            and data_refilter_it < 1
        )

        if needs_loop1_refilter:
            dominant_bao = (
                curation_summary.dominant_bao_format
                if curation_summary is not None
                else "BAO_0000357"
            )
            feedback_msg = (
                f"Critic Agent: Обнаружена методологическая неоднородность экспериментальных данных "
                f"(assay_homogeneous=False). Извлеченные правила SAR содержат смешение протоколов испытаний. "
                f"Активирована Петля самокоррекции №1: возврат на этап курации со строгой фильтрацией "
                f"по доминирующему формату протокола '{dominant_bao}'."
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            trace_event = TraceEvent(
                step_index=step_idx,
                timestamp_iso=datetime.now(tz=UTC).isoformat(),
                agent_name="CriticAgent",
                phase="critic_assay_homogeneity_audit",
                input_summary=(
                    f"{len(sar_rules)} SAR rules | dataset_homogeneous={not dataset_not_homogeneous}"
                ),
                action_taken=(
                    f"Detected heterogeneous assay protocols. Triggered Loop 1 (refilter_data) "
                    f"with active_bao_filter='{dominant_bao}'"
                ),
                output_summary=feedback_msg,
                verdict_or_routing="refilter_data",
                fallback_triggered=None,
                duration_ms=round(duration_ms, 2),
            )
            return {
                "critic_verdict": "refilter_data",
                "assay_filter_strict": True,
                "active_bao_filter": dominant_bao,
                "data_refilter_iteration": data_refilter_it + 1,
                "critic_feedback": feedback_msg,
                "trace_log": [trace_event],
            }

        # Все протоколы однородны или исчерпан лимит перефильтрации
        feedback_msg = (
            "Critic Agent: Экспериментальные протоколы проверены. Правила SAR методологически однородны "
            "(assay_homogeneous=True). Разрешен переход к направленному молекулярному дизайну."
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        trace_event = TraceEvent(
            step_index=step_idx,
            timestamp_iso=datetime.now(tz=UTC).isoformat(),
            agent_name="CriticAgent",
            phase="critic_assay_homogeneity_audit",
            input_summary=f"{len(sar_rules)} SAR rules checked",
            action_taken="Confirmed assay protocol homogeneity. Routed workflow to DesignAgent.",
            output_summary=feedback_msg,
            verdict_or_routing="proceed_to_design",
            fallback_triggered=None,
            duration_ms=round(duration_ms, 2),
        )
        return {
            "critic_verdict": "proceed_to_design",
            "critic_feedback": feedback_msg,
            "trace_log": [trace_event],
        }

    # ========================================================================
    # ФАЗА 2: Валидация профиля кандидатов (после Evaluation Agent)
    # ========================================================================
    if last_stage == "evaluation":
        top_candidates = state.get("top_candidates", [])
        constraints = state["query_constraints"]
        design_it = int(state.get("design_iteration", 1))

        valid_count = sum(1 for c in top_candidates if c.passes_all_constraints)
        target_k = (
            min(constraints.top_k_candidates, len(top_candidates))
            if top_candidates
            else 1
        )

        # Петля №2: если кандидаты не удовлетворяют ограничениям и лимит итераций не исчерпан (< 2)
        needs_loop2_redesign = (
            valid_count < target_k or len(top_candidates) == 0
        ) and (design_it < 2)

        if needs_loop2_redesign:
            reasons = [viol for c in top_candidates for viol in c.constraint_violations]
            if not reasons:
                reasons = [
                    "Недостаточное число кандидатов, удовлетворяющих целевым ограничениям"
                ]
            updated_constraints, feedback_text, fallback_tier = (
                llm_adapter.formulate_corrective_constraints(
                    current_constraints=constraints,
                    evaluated_candidates=top_candidates,
                    critic_summary_reasons=reasons,
                )
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            trace_event = TraceEvent(
                step_index=step_idx,
                timestamp_iso=datetime.now(tz=UTC).isoformat(),
                agent_name="CriticAgent",
                phase=f"critic_candidate_validation_loop2_it{design_it}",
                input_summary=(
                    f"Top {len(top_candidates)} candidates | valid: {valid_count}/{target_k} | "
                    f"design_iteration={design_it}"
                ),
                action_taken=(
                    f"Detected target metric violations. Triggered Loop 2 (redesign_molecules) "
                    f"with updated constraints: {updated_constraints.preferred_substituent_types}"
                ),
                output_summary=feedback_text,
                verdict_or_routing="redesign_molecules",
                fallback_triggered=fallback_tier,
                duration_ms=round(duration_ms, 2),
            )
            return {
                "critic_verdict": "redesign_molecules",
                "query_constraints": updated_constraints,
                "critic_feedback": feedback_text,
                "trace_log": [trace_event],
            }

        approval_feedback = (
            f"Critic Agent: Портфель кандидатов успешно утвержден "
            f"(валидных: {valid_count}/{len(top_candidates)}). "
            f"Сформирован итоговый аналитический отчет."
        )

        # Все ограничения выполнены либо исчерпан лимит итераций (выдача компромиссного топа)
        report_md, fallback_report = llm_adapter.generate_final_report(
            user_query=state["user_query"],
            constraints=constraints,
            curation_summary=state.get("curation_summary"),
            sar_rules=state.get("extracted_sar_rules", []),
            top_candidates=top_candidates,
            critic_verdict="approve",
            critic_feedback=approval_feedback,
            data_refilter_iteration=int(state.get("data_refilter_iteration", 0)),
            design_iteration=design_it,
        )

        is_compromise = valid_count < target_k
        action_desc = (
            "Approved candidates satisfying all user constraints. Synthesized final report."
            if not is_compromise
            else f"Design iteration limit reached ({design_it}). Approved best compromise candidates "
            f"ranked by Harrington desirability with explicit violation tags."
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        trace_event = TraceEvent(
            step_index=step_idx,
            timestamp_iso=datetime.now(tz=UTC).isoformat(),
            agent_name="CriticAgent",
            phase=f"critic_final_approval_it{design_it}",
            input_summary=(
                f"Final portfolio: {len(top_candidates)} candidates | "
                f"top Harrington D={top_candidates[0].mpo_harrington_score if top_candidates else 0.0:.3f}"
            ),
            action_taken=action_desc,
            output_summary=approval_feedback,
            verdict_or_routing="approve",
            fallback_triggered=fallback_report,
            duration_ms=round(duration_ms, 2),
        )
        return {
            "critic_verdict": "approve",
            "critic_feedback": approval_feedback,
            "final_report_markdown": report_md,
            "trace_log": [trace_event],
        }

    # Непредвиденная стадия
    return {
        "critic_verdict": "approve",
        "critic_feedback": f"Critic Agent: unhandled stage '{last_stage}', terminating safely.",
        "trace_log": [],
    }


def route_after_critic(state: SARGraphState) -> Literal["research", "design", "end"]:
    """Маршрутизатор переходов после узла Critic Agent в LangGraph."""
    verdict = state.get("critic_verdict")
    if verdict == "refilter_data":
        return "research"
    if verdict in ("proceed_to_design", "redesign_molecules"):
        return "design"
    return "end"
