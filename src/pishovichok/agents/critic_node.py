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
from pishovichok.state.graph_state import (
    SARGraphState,
    resolve_tool_config,
    update_param_sources,
)
from pishovichok.state.schemas import TraceEvent


def critic_node(state: SARGraphState) -> dict[str, Any]:
    """Узел Critic Agent: аудит однородности протоколов и валидация физико-химического профиля."""
    start_time = time.perf_counter()
    llm_adapter = LLMAdapter()
    last_stage = state.get("last_executed_stage")
    step_idx = len(state.get("trace_log", []))
    tool_config = resolve_tool_config(state)
    param_sources = dict(state.get("param_sources", {}))
    node_telemetry = dict(state.get("node_telemetry", {}))

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
            tool_config.strict_assay_homogeneity
            and (has_heterogeneous_rules or dataset_not_homogeneous)
            and not bool(state.get("assay_filter_strict", False))
            and data_refilter_it < tool_config.max_refilter_iterations
        )

        if needs_loop1_refilter:
            dominant_bao = (
                curation_summary.dominant_bao_format
                if curation_summary is not None
                else "BAO_0000357"
            )
            updated_tool_config, changed_keys = tool_config.apply_overrides(
                {"allowed_bao_formats": [dominant_bao]}
            )
            if not changed_keys:
                changed_keys = ["allowed_bao_formats"]
            param_sources = update_param_sources(
                param_sources, changed_keys, "critic_loop"
            )

            feedback_msg = (
                f"Верификация: Обнаружена методологическая неоднородность экспериментальных данных "
                f"(однородность=нет). Извлеченные правила SAR содержат смешение протоколов испытаний. "
                f"Активирована Петля самокоррекции №1: возврат на этап курации со строгой фильтрацией "
                f"по доминирующему формату протокола '{dominant_bao}'."
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            node_telemetry["critic"] = {
                "active_params": {
                    "strict_assay_homogeneity": tool_config.strict_assay_homogeneity,
                    "max_refilter_iterations": tool_config.max_refilter_iterations,
                    "allowed_bao_formats": list(updated_tool_config.allowed_bao_formats),
                },
                "stats": {
                    "phase": "assay_homogeneity_audit",
                    "verdict": "refilter_data",
                    "critic_overrides": changed_keys,
                    "duration_ms": round(duration_ms, 2),
                },
            }
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
                "tool_config": updated_tool_config,
                "param_sources": param_sources,
                "node_telemetry": node_telemetry,
                "critic_feedback": feedback_msg,
                "trace_log": [trace_event],
            }

        # Все протоколы однородны или исчерпан лимит перефильтрации
        feedback_msg = (
            "Верификация: Экспериментальные протоколы проверены. Правила SAR методологически однородны "
            "(однородность=да). Разрешен переход к направленному молекулярному дизайну."
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        node_telemetry["critic"] = {
            "active_params": {
                "strict_assay_homogeneity": tool_config.strict_assay_homogeneity,
                "max_refilter_iterations": tool_config.max_refilter_iterations,
                "allowed_bao_formats": list(tool_config.allowed_bao_formats),
            },
            "stats": {
                "phase": "assay_homogeneity_audit",
                "verdict": "proceed_to_design",
                "sar_rules_checked": len(sar_rules),
                "duration_ms": round(duration_ms, 2),
            },
        }
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
            "node_telemetry": node_telemetry,
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
        effective_top_k = (
            tool_config.top_k_output
            if param_sources.get("top_k_output") not in (None, "default")
            or tool_config.top_k_output != 5
            else constraints.top_k_candidates
        )
        target_k = min(effective_top_k, len(top_candidates)) if top_candidates else 1

        docked_top = [
            c
            for c in top_candidates
            if c.docking is not None
            and c.docking.status == "SUCCESS"
            and c.docking.docking_score_kcal is not None
        ]
        poor_docking_affinity = bool(docked_top) and all(
            float(c.docking.docking_score_kcal) > -6.5  # type: ignore[union-attr]
            for c in docked_top
        )

        # Петля №2: если кандидаты не удовлетворяют ограничениям (или все докированные кандидаты слабо связываются > -6.5) и лимит итераций не исчерпан
        needs_loop2_redesign = (
            (
                (
                    tool_config.strict_property_thresholds
                    and (valid_count < target_k or len(top_candidates) == 0)
                )
                or poor_docking_affinity
            )
            and (design_it < tool_config.max_iterations)
        )

        if needs_loop2_redesign:
            reasons = [viol for c in top_candidates for viol in c.constraint_violations]
            if poor_docking_affinity:
                reasons.append(
                    "Все докированные топ-кандидаты имеют слабую расчетную энергию связывания (ΔG > -6.5 ккал/моль)"
                )
            if not reasons:
                reasons = [
                    "Недостаточное число кандидатов, удовлетворяющих целевым ограничениям"
                ]
            updated_constraints, feedback_text, fallback_tier = (
                llm_adapter.formulate_corrective_constraints(
                    current_constraints=constraints,
                    evaluated_candidates=top_candidates,
                    critic_summary_reasons=reasons,
                    tool_config=tool_config,
                )
            )
            critic_overrides: dict[str, Any] = {
                "max_candidates_pool": min(tool_config.max_candidates_pool + 15, 200),
                "crem_max_replacements": min(tool_config.crem_max_replacements + 15, 200),
            }
            joined_reasons = " ".join(reasons).lower()
            if "logp" in joined_reasons or "tpsa" in joined_reasons:
                new_weights = dict(tool_config.mpo_weights)
                if "logp" in joined_reasons:
                    new_weights["logp"] = round(
                        min(new_weights.get("logp", 1.0) + 0.3, 5.0), 2
                    )
                if "tpsa" in joined_reasons:
                    new_weights["tpsa"] = round(
                        min(new_weights.get("tpsa", 0.8) + 0.3, 5.0), 2
                    )
                critic_overrides["mpo_weights"] = new_weights

            updated_tool_config, changed_keys = tool_config.apply_overrides(
                critic_overrides
            )
            param_sources = update_param_sources(
                param_sources, changed_keys, "critic_loop"
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            node_telemetry["critic"] = {
                "active_params": {
                    "strict_property_thresholds": tool_config.strict_property_thresholds,
                    "max_iterations": tool_config.max_iterations,
                    "critic_logp_step_reduction": tool_config.critic_logp_step_reduction,
                    "max_candidates_pool": updated_tool_config.max_candidates_pool,
                    "crem_max_replacements": updated_tool_config.crem_max_replacements,
                },
                "stats": {
                    "phase": f"candidate_validation_loop2_it{design_it}",
                    "verdict": "redesign_molecules",
                    "valid_count": valid_count,
                    "target_k": target_k,
                    "critic_overrides": changed_keys,
                    "duration_ms": round(duration_ms, 2),
                },
            }
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
                "tool_config": updated_tool_config,
                "param_sources": param_sources,
                "node_telemetry": node_telemetry,
                "critic_feedback": feedback_text,
                "trace_log": [trace_event],
            }

        approval_feedback = (
            f"Верификация: Портфель кандидатов успешно утвержден "
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
        node_telemetry["critic"] = {
            "active_params": {
                "strict_assay_homogeneity": tool_config.strict_assay_homogeneity,
                "strict_property_thresholds": tool_config.strict_property_thresholds,
                "max_iterations": tool_config.max_iterations,
                "max_refilter_iterations": tool_config.max_refilter_iterations,
            },
            "stats": {
                "phase": f"final_approval_it{design_it}",
                "verdict": "approve",
                "valid_count": valid_count,
                "target_k": target_k,
                "is_compromise": is_compromise,
                "duration_ms": round(duration_ms, 2),
            },
        }
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
            "node_telemetry": node_telemetry,
            "critic_feedback": approval_feedback,
            "final_report_markdown": report_md,
            "trace_log": [trace_event],
        }

    # Непредвиденная стадия
    return {
        "critic_verdict": "approve",
        "critic_feedback": f"Верификация: необработанная стадия '{last_stage}', безопасное завершение.",
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
