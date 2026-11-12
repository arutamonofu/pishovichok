"""Адаптер языковой модели с каскадной отказоустойчивостью (Слой 3).

Реализует трехуровневый каскад (Раздел 1.2 ARCHITECTURE.md):
- Уровень 1: Облачный OpenAI-совместимый API (OpenAI / DeepSeek / OpenRouter / vLLM)
  через `ChatOpenAI.with_structured_output(..., method="json_schema", strict=True)`.
- Уровень 2: Локальный эндпоинт Ollama (`http://localhost:11434/v1`).
- Уровень 3: Детерминированный Rule-Based Engine (парсер регулярных выражений,
  шаблонизатор корректирующих директив и синтезатор научных гипотез).

Все три уровня возвращают абсолютно идентичные схемы Pydantic v2 и строки отчетов.
"""

from __future__ import annotations

import re
import socket
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from pishovichok.agents.prompts import (
    CRITIC_CORRECTIVE_SYSTEM_PROMPT,
    FINAL_REPORT_SYSTEM_PROMPT,
    HYPOTHESIS_RATIONALE_SYSTEM_PROMPT,
    QUERY_PARSER_SYSTEM_PROMPT,
)
from pishovichok.config import settings
from pishovichok.state.schemas import (
    CandidateMolecule,
    DatasetCurationSummary,
    QueryConstraints,
    SARRule,
)


class CorrectiveDesignDirective(BaseModel):
    """Схема структурированного вывода директивы самокоррекции от Critic Agent."""

    model_config = ConfigDict(extra="forbid")

    preferred_substituent_types: list[
        Literal["polar", "less_lipophilic", "exclude_halogens", "any"]
    ] = Field(..., description="Корректирующие требования к типу вводимых заместителей")
    adjusted_min_tpsa: float | None = Field(
        default=None,
        description="Рекомендуемая минимальная полярная площадь поверхности TPSA, Å²",
    )
    adjusted_max_sa_score: float | None = Field(
        default=None, description="Уточненный порог синтетической доступности SAscore"
    )
    critic_feedback: str = Field(
        ..., description="Научное обоснование отклонения и директива для Design Agent"
    )


class CandidateRationaleItem(BaseModel):
    """Структурированное научное обоснование для одной молекулы-кандидата."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    text_summary: str


class BatchHypothesisRationales(BaseModel):
    """Пакет структурированных научных обоснований кандидатов."""

    model_config = ConfigDict(extra="forbid")

    items: list[CandidateRationaleItem]


def _is_network_endpoint_reachable(url: str, timeout_sec: float = 0.20) -> bool:
    """Быстро проверяет доступность TCP-сокета хоста без блокировки выполнения."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except (TimeoutError, OSError):
        return False


class LLMAdapter:
    """Адаптер взаимодействия со слоем LLM с гарантированным локальным фолбэком."""

    def __init__(self) -> None:
        self.openai_api_key = settings.openai_api_key
        self.openai_base_url = settings.openai_base_url
        self.llm_model = settings.llm_model
        self.ollama_base_url = settings.ollama_base_url
        self.ollama_model = settings.ollama_model
        self.timeout_seconds = settings.llm_timeout_seconds
        self.force_rule_based = settings.force_rule_based_llm

    # ========================================================================
    # 1. СЕМАНТИЧЕСКИЙ РАЗБОР ВХОДНОГО ЗАПРОСА В QueryConstraints
    # ========================================================================
    def parse_query_constraints(
        self,
        user_query: str,
        uploaded_csv_path: str | None = None,
    ) -> tuple[QueryConstraints, str | None]:
        """Парсит запрос пользователя на естественном языке в типизированную схему `QueryConstraints`.

        Возвращает кортеж `(constraints, fallback_triggered)`.
        """
        # Уровень 1: Облачный API (OpenAI / DeepSeek / vLLM)
        if not self.force_rule_based and self.openai_api_key:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=self.llm_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.0,
                    timeout=self.timeout_seconds,
                )
                structured_llm = llm.with_structured_output(
                    QueryConstraints,
                    method="json_schema",
                    strict=True,
                )
                csv_ctx = (
                    f"\n[Контекст: передан путь к CSV-файлу: {uploaded_csv_path}]"
                    if uploaded_csv_path
                    else ""
                )
                res = structured_llm.invoke(
                    [
                        SystemMessage(content=QUERY_PARSER_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"Запрос исследователя: {user_query}{csv_ctx}"
                        ),
                    ]
                )
                if isinstance(res, QueryConstraints):
                    return res, None
            except Exception:  # noqa: BLE001, S110
                pass

        # Уровень 2: Локальный сервер Ollama
        if not self.force_rule_based and _is_network_endpoint_reachable(
            self.ollama_base_url
        ):
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                ollama_llm = ChatOpenAI(
                    model=self.ollama_model,
                    api_key="ollama",
                    base_url=self.ollama_base_url,
                    temperature=0.0,
                    timeout=self.timeout_seconds,
                )
                structured_ollama = ollama_llm.with_structured_output(
                    QueryConstraints,
                    method="function_calling",
                )
                res = structured_ollama.invoke(
                    [
                        SystemMessage(content=QUERY_PARSER_SYSTEM_PROMPT),
                        HumanMessage(content=f"Запрос исследователя: {user_query}"),
                    ]
                )
                if isinstance(res, QueryConstraints):
                    return res, "ollama_local_llm"
            except Exception:  # noqa: BLE001, S110
                pass

        # Уровень 3: Детерминированный Rule-Based Engine
        deterministic_constraints = self._deterministic_parse_query(
            user_query, uploaded_csv_path
        )
        return deterministic_constraints, "deterministic_rule_based_engine"

    def _deterministic_parse_query(
        self,
        user_query: str,
        uploaded_csv_path: str | None = None,
    ) -> QueryConstraints:
        """Детерминированный парсер запросов на чистом Python на базе регулярных выражений."""
        text = user_query.strip()
        text_lower = text.lower()

        # 1. Определение типа сценария
        is_csv = (
            bool(uploaded_csv_path)
            or ("csv" in text_lower)
            or ("лабораторн" in text_lower and "сери" in text_lower)
        )
        has_lead = (
            bool(re.search(r"COc1[a-zA-Z0-9@#\+\-\(\)=\[\]]+", text))
            or ("гефитиниб" in text_lower)
            or ("лидер" in text_lower and "оптимиз" in text_lower)
        )

        if is_csv:
            scenario_type: Literal[
                "target_based_sar", "lead_optimization", "custom_csv_analysis"
            ] = "custom_csv_analysis"
        elif has_lead:
            scenario_type = "lead_optimization"
        else:
            scenario_type = "target_based_sar"

        # 2. Определение мишени ChEMBL
        target_chembl_id = None
        match_target = re.search(r"(CHEMBL\d+)", text, re.IGNORECASE)
        if match_target:
            target_chembl_id = match_target.group(1).upper()
        elif "btk" in text_lower or "брутон" in text_lower:
            target_chembl_id = "CHEMBL5251"
        elif "egfr" in text_lower:
            target_chembl_id = "CHEMBL203"
        elif "gsk" in text_lower or "гликоген" in text_lower:
            target_chembl_id = "CHEMBL262"
        elif scenario_type == "target_based_sar":
            target_chembl_id = "CHEMBL203"

        # 3. Определение SMILES лидера для Сценария 2
        lead_smiles = None
        match_smi = re.search(
            r"(COc1cc2ncnc\(Nc3ccc\(F\)c\(Cl\)c3\)c2cc1OCCCN1CCOCC1)", text
        )
        if match_smi:
            lead_smiles = match_smi.group(1)
        elif "гефитиниб" in text_lower or scenario_type == "lead_optimization":
            lead_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"

        # 4. Диапазоны липофильности cLogP
        min_logp = None
        max_logp = 4.5
        match_range_logp = re.search(
            r"logp\s*(?:от|between)\s*([+-]?\d+(?:\.\d+)?)\s*(?:до|to|and)\s*([+-]?\d+(?:\.\d+)?)",
            text_lower,
        )
        if match_range_logp:
            min_logp = float(match_range_logp.group(1))
            max_logp = float(match_range_logp.group(2))
        else:
            match_max_logp = re.search(
                r"logp\s*(?:не\s*более|ниже|<|<=|менее)\s*([+-]?\d+(?:\.\d+)?)",
                text_lower,
            )
            if match_max_logp:
                max_logp = float(match_max_logp.group(1))

        # 5. Синтетическая доступность SAscore
        max_sa_score = 3.2
        match_sa = re.search(
            r"sascore\s*(?:ниже|менее|<|<=|не\s*более)\s*(\d+(?:\.\d+)?)", text_lower
        )
        if match_sa:
            max_sa_score = float(match_sa.group(1))

        # 6. Кардиотоксичность hERG
        max_herg_prob = 0.50
        if "herg" in text_lower and any(
            w in text_lower for w in ("минимиз", "снизь", "низк", "безопасн")
        ):
            max_herg_prob = 0.35
        match_herg = re.search(r"herg\s*(?:<|<=|менее)\s*(\d+(?:\.\d+)?)", text_lower)
        if match_herg:
            max_herg_prob = float(match_herg.group(1))

        # 7. Запрет PAINS/Brenk
        require_no_pains = not (
            "разрешить pains" in text_lower or "допустить алерты" in text_lower
        )

        # 8. Число кандидатов
        top_k = 5
        match_k = re.search(r"(\d+)\s+(?:нов\w+|структур\w+|кандидат\w+)", text_lower)
        if match_k:
            parsed_k = int(match_k.group(1))
            if 1 <= parsed_k <= 20:
                top_k = parsed_k

        return QueryConstraints(
            scenario_type=scenario_type,
            target_chembl_id=target_chembl_id,
            lead_smiles=lead_smiles,
            freeze_scaffold=True,
            min_logp=min_logp,
            max_logp=max_logp,
            max_sa_score=max_sa_score,
            max_herg_prob=max_herg_prob,
            require_no_pains=require_no_pains,
            top_k_candidates=top_k,
            preferred_substituent_types=["any"],
        )

    # ========================================================================
    # 2. ФОРМИРОВАНИЕ КОРРЕКТИРУЮЩИХ ОГРАНИЧЕНИЙ ДЛЯ ПЕТЛИ №2 (Critic Agent)
    # ========================================================================
    def formulate_corrective_constraints(
        self,
        current_constraints: QueryConstraints,
        evaluated_candidates: list[CandidateMolecule],
        critic_summary_reasons: list[str],
    ) -> tuple[QueryConstraints, str, str | None]:
        """Анализирует причины отклонения кандидатов и формирует директиву самокоррекции для Петли №2.

        Возвращает `(updated_constraints, critic_feedback, fallback_triggered)`.
        """
        # Уровень 1: Облачный API со структурированным выводом CorrectiveDesignDirective
        if not self.force_rule_based and self.openai_api_key:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=self.llm_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.0,
                    timeout=self.timeout_seconds,
                )
                structured_llm = llm.with_structured_output(
                    CorrectiveDesignDirective,
                    method="json_schema",
                    strict=True,
                )
                reasons_text = "\n".join(f"- {r}" for r in critic_summary_reasons)
                candidates_brief = "\n".join(
                    f"- {c.candidate_id}: cLogP={c.calc_rdkit.clogp:.2f}, SAscore={c.calc_rdkit.sa_score:.2f}, "
                    f"TPSA={c.calc_rdkit.tpsa:.1f}, hERG_prob={c.pred_admet.herg_liability_prob:.2f}, "
                    f"Violations={c.constraint_violations}"
                    for c in evaluated_candidates[:5]
                )
                directive: CorrectiveDesignDirective = structured_llm.invoke(
                    [
                        SystemMessage(content=CRITIC_CORRECTIVE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                f"Текущие ограничения: {current_constraints.model_dump_json()}\n"
                                f"Выявленные нарушения:\n{reasons_text}\n"
                                f"Кандидаты текущей итерации:\n{candidates_brief}\n"
                                "Сформируй корректирующие параметры для следующей итерации Design Agent."
                            )
                        ),
                    ]
                )
                updated = current_constraints.model_copy(
                    update={
                        "preferred_substituent_types": directive.preferred_substituent_types,
                        "min_tpsa": directive.adjusted_min_tpsa
                        or current_constraints.min_tpsa
                        or 85.0,
                        "max_sa_score": directive.adjusted_max_sa_score
                        or current_constraints.max_sa_score,
                    }
                )
                return updated, directive.critic_feedback, None
            except Exception:  # noqa: BLE001, S110
                pass

        # Уровень 3: Детерминированный фолбэк
        updated = current_constraints.model_copy(
            update={
                "preferred_substituent_types": [
                    "polar",
                    "less_lipophilic",
                    "exclude_halogens",
                ],
                "min_tpsa": 85.0,
            }
        )
        reasons_str = "; ".join(critic_summary_reasons)
        feedback = (
            f"Выявлены отклонения параметров кандидатов от целевых порогов ({reasons_str}). "
            "Директива Critic Agent для Итерации 2: исключить липофильные галогеновые трансформации ([*:1]Cl, [*:1]F); "
            "переключиться строго на полярные биоизостерные модификации (C#N, C(=O)NC, OCCO) с положительным сдвигом "
            "полярной площади (Delta TPSA > 0) для удержания cLogP < 3.5 и снижения риска hERG."
        )
        return updated, feedback, "deterministic_rule_based_engine"

    # ========================================================================
    # 3. ГЕНЕРАЦИЯ ИНТЕРПРЕТИРУЕМЫХ НАУЧНЫХ ГИПОТЕЗ ДЛЯ КАНДИДАТОВ
    # ========================================================================
    def generate_hypothesis_rationales(
        self,
        candidates: list[CandidateMolecule],
    ) -> tuple[dict[str, str], str | None]:
        """Формирует научные текстовые обоснования `text_summary` для списка кандидатов.

        Возвращает `(dict_of_candidate_id_to_summary, fallback_triggered)`.
        """
        if not self.force_rule_based and self.openai_api_key:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=self.llm_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.0,
                    timeout=self.timeout_seconds,
                )
                structured_llm = llm.with_structured_output(
                    BatchHypothesisRationales,
                    method="json_schema",
                    strict=True,
                )
                facts_text = "\n".join(
                    f"Кандидат {c.candidate_id}:\n"
                    f"- Родитель: {c.parent_chembl_id} (pIC50={c.parent_exp_pic50:.2f}, BAO={c.parent_bao_format})\n"
                    f"- Правило: {c.hypothesis_rationale.applied_smirks} ({c.hypothesis_rationale.rule_id})\n"
                    f"- Статистика MMP: support={c.hypothesis_rationale.support_pairs_count}, ΔpIC50={c.hypothesis_rationale.historical_mean_delta_pic50:+.2f}, однороден={c.hypothesis_rationale.assay_homogeneous}\n"
                    f"- Детерминированный сдвиг: ΔcLogP={c.calc_rdkit.delta_clogp_vs_parent:+.2f}, ΔTPSA={c.calc_rdkit.delta_tpsa_vs_parent:+.1f}, SAscore={c.calc_rdkit.sa_score:.2f}\n"
                    f"- ADMET: прогноз pIC50={c.pred_admet.predicted_pic50:.2f}, hERG prob={c.pred_admet.herg_liability_prob:.2f} ({c.pred_admet.herg_risk_level}), растворимость={c.pred_admet.solubility_class}\n"
                    f"- BRICS прекурсоры: {', '.join(c.calc_rdkit.brics_fragments[:3])}"
                    for c in candidates
                )
                batch_res: BatchHypothesisRationales = structured_llm.invoke(
                    [
                        SystemMessage(content=HYPOTHESIS_RATIONALE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"Сформулируй краткие научные гипотезы для кандидатов:\n{facts_text}"
                        ),
                    ]
                )
                res_map = {
                    item.candidate_id: item.text_summary for item in batch_res.items
                }
                return res_map, None
            except Exception:  # noqa: BLE001, S110
                pass

        # Детерминированный фолбэк: возвращаем существующие детерминированные гипотезы из ядра
        return {
            c.candidate_id: c.hypothesis_rationale.text_summary for c in candidates
        }, "deterministic_rule_based_engine"

    # ========================================================================
    # 4. СИНТЕЗ ИТОГОВОГО АКАДЕМИЧЕСКОГО Markdown-ОТЧЕТА
    # ========================================================================
    def generate_final_report(
        self,
        user_query: str,
        constraints: QueryConstraints,
        curation_summary: DatasetCurationSummary | None,
        sar_rules: list[SARRule],
        top_candidates: list[CandidateMolecule],
        critic_verdict: str,
        critic_feedback: str | None,
        data_refilter_iteration: int,
        design_iteration: int,
    ) -> tuple[str, str | None]:
        """Синтезирует академический отчет по вычислительной сессии."""
        if not self.force_rule_based and self.openai_api_key:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=self.llm_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.0,
                    timeout=self.timeout_seconds,
                )
                brief = (
                    f"Запрос: {user_query}\n"
                    f"Сценарий: {constraints.scenario_type}, Мишень: {constraints.target_chembl_id}\n"
                    f"Курация: исходно {curation_summary.initial_rows_count if curation_summary else 0} строк, "
                    f"отобрано {curation_summary.curated_molecules_count if curation_summary else 0}, "
                    f"доминирующий BAO={curation_summary.dominant_bao_format if curation_summary else 'N/A'}\n"
                    f"Петля №1 (refilter_data): итераций {data_refilter_iteration}\n"
                    f"Петля №2 (redesign_molecules): итераций дизайна {design_iteration}\n"
                    f"Вердикт Critic Agent: {critic_verdict}\n"
                    f"Директива Critic: {critic_feedback or 'Все критерии выполнены.'}\n"
                    f"Топ-правила SAR: {', '.join(f'{r.smirks} (ΔpIC50={r.mean_delta_pic50:+.2f}, n={r.support_pairs_count})' for r in sar_rules[:4])}\n"
                    f"Кандидаты ({len(top_candidates)}):\n"
                    + "\n".join(
                        f"- {c.candidate_id} ({c.canonical_smiles}): cLogP={c.calc_rdkit.clogp:.2f}, "
                        f"SAscore={c.calc_rdkit.sa_score:.2f}, hERG={c.pred_admet.herg_liability_prob:.2f}, "
                        f"D={c.mpo_harrington_score:.3f}, violations={c.constraint_violations}"
                        for c in top_candidates
                    )
                )
                ai_msg = llm.invoke(
                    [
                        SystemMessage(content=FINAL_REPORT_SYSTEM_PROMPT),
                        HumanMessage(content=f"Сформируй итоговый отчет:\n{brief}"),
                    ]
                )
                return str(ai_msg.content), None
            except Exception:  # noqa: BLE001, S110
                pass

        # Детерминированный структурированный отчет на Markdown
        report = self._deterministic_final_report(
            user_query=user_query,
            constraints=constraints,
            curation_summary=curation_summary,
            sar_rules=sar_rules,
            top_candidates=top_candidates,
            critic_verdict=critic_verdict,
            critic_feedback=critic_feedback,
            data_refilter_iteration=data_refilter_iteration,
            design_iteration=design_iteration,
        )
        return report, "deterministic_rule_based_engine"

    def _deterministic_final_report(
        self,
        user_query: str,
        constraints: QueryConstraints,
        curation_summary: DatasetCurationSummary | None,
        sar_rules: list[SARRule],
        top_candidates: list[CandidateMolecule],
        critic_verdict: str,
        critic_feedback: str | None,
        data_refilter_iteration: int,
        design_iteration: int,
    ) -> str:
        """Детерминированный генератор отчета в академическом стиле."""
        target_name = constraints.target_chembl_id or "Custom Dataset"
        init_count = curation_summary.initial_rows_count if curation_summary else 0
        cur_count = curation_summary.curated_molecules_count if curation_summary else 0
        dom_bao = (
            curation_summary.dominant_bao_format if curation_summary else "BAO_0000357"
        )

        rules_md = (
            "\n".join(
                f"| `{r.rule_id}` | `{r.smirks}` | `{r.mean_delta_pic50:+.2f}` | `{r.support_pairs_count}` | `{r.assay_homogeneous}` |"
                for r in sar_rules[:5]
            )
            or "| - | Нет правил | - | - | - |"
        )

        cands_md = "\n".join(
            f"| `{c.candidate_id}` | `{c.canonical_smiles}` | `{c.parent_chembl_id}` (`{c.parent_exp_pic50:.2f}`) | "
            f"MW={c.calc_rdkit.mw:.1f}, LogP={c.calc_rdkit.clogp:.2f}, SA={c.calc_rdkit.sa_score:.2f} | "
            f"pIC50={c.pred_admet.predicted_pic50:.2f}, hERG={c.pred_admet.herg_liability_prob:.2f} ({c.pred_admet.herg_risk_level}) | "
            f"`D = {c.mpo_harrington_score:.3f}` |"
            for c in top_candidates
        )

        brics_md = "\n".join(
            f"- **{c.candidate_id}**: {', '.join(f'`{b}`' for b in c.calc_rdkit.brics_fragments[:4])}"
            for c in top_candidates
        )

        compromise_notice = ""
        has_violations = any(not c.passes_all_constraints for c in top_candidates)
        if has_violations and design_iteration >= settings.max_design_iterations:
            compromise_notice = (
                "> [!WARNING]\n"
                "> **Компромиссная выдача при достижении лимита итераций дизайна:**\n"
                "> Процесс завершен по лимиту шагов (`design_iteration >= 2`). Отобраны лучшие компромиссные структуры "
                "по многопараметрическому индексу желательности Харрингтона. Остаточные отклонения явно маркированы в карточках кандидатов.\n\n"
            )

        return (
            f"# Научно-технический отчет: Направленный дизайн кандидатов по мишени {target_name}\n\n"
            f"**Входной запрос:** _{user_query}_\n\n"
            f"## 1. Курация данных и методологический аудит испытаний (Петля №1)\n"
            f"- **Источник данных:** `{constraints.scenario_type}` ({target_name})\n"
            f"- **Объем сырой выборки:** `{init_count}` записей\n"
            f"- **Объем стандартизированной выборки:** `{cur_count}` уникальных структур\n"
            f"- **Доминирующий протокол испытания:** `{dom_bao}` (`IC50`)\n"
            f"- **Число срабатываний Петли №1 (`refilter_data`):** `{data_refilter_iteration}`\n\n"
            f"## 2. Извлеченные правила «структура–активность» (MMP SAR)\n\n"
            f"| ID правила | SMIRKS-трансформация | Средний ΔpIC50 | Число пар | Гомогенность протокола |\n"
            f"| :--- | :--- | :--- | :--- | :--- |\n"
            f"{rules_md}\n\n"
            f"## 3. Направленный дизайн и самокоррекция параметров (Петля №2)\n"
            f"- **Выполнено итераций дизайна:** `{design_iteration}` из `{settings.max_design_iterations}`\n"
            f"- **Итоговый вердикт Critic Agent:** `{critic_verdict}`\n"
            f"- **Директива контроля качества:** {critic_feedback or 'Все целевые параметры выполнены.'}\n\n"
            f"{compromise_notice}"
            f"## 4. Сводная таблица отобранных кандидатов\n\n"
            f"| ID | SMILES кандидата | Экспериментальный факт | Детерминированный расчет RDKit | Модельный прогноз ADMET | Индекс Харрингтона |\n"
            f"| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            f"{cands_md}\n\n"
            f"## 5. Ретросинтетическая декомпозиция на доступные строительные блоки (BRICS)\n"
            f"{brics_md}\n\n"
            f"## 6. Рекомендации для органического синтеза\n"
            f"Предложенные структуры сохраняют базовое фармакофорное ядро, демонстрируют оптимизированный баланс "
            f"липофильности и полярной поверхности и декомпозируются на стандартные коммерчески доступные синтоны."
        )


llm_adapter = LLMAdapter()
