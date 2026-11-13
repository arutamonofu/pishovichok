"""Адаптер языковой модели с каскадной отказоустойчивостью (Слой 3).

Реализует трехуровневый каскад:
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
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field
from rdkit.Chem import Descriptors

from pishovichok.agents.prompts import (
    CRITIC_CORRECTIVE_SYSTEM_PROMPT,
    FINAL_REPORT_SYSTEM_PROMPT,
    HYPOTHESIS_RATIONALE_SYSTEM_PROMPT,
    QUERY_PARSER_SYSTEM_PROMPT,
)
from pishovichok.config import settings
from pishovichok.state.schemas import (
    ADMETBackendLiteral,
    CandidateMolecule,
    DatasetCurationSummary,
    FilterCatalogLiteral,
    QueryConstraints,
    SARRule,
    ToolConfig,
)

_KNOWN_RDKIT_DESCRIPTOR_NAMES: frozenset[str] = frozenset(
    name for name, _fn in Descriptors._descList
)


class MPOWeightsOverride(BaseModel):
    """Частичное переопределение весов MPO Харрингтона в структурированном выводе LLM."""

    model_config = ConfigDict(extra="forbid")

    pic50: float | None = None
    logp: float | None = None
    sascore: float | None = None
    qed: float | None = None
    tpsa: float | None = None
    logs: float | None = None
    herg: float | None = None


class ToolConfigOverrides(BaseModel):
    """Строгая схема извлекаемых из NL-запроса переопределений настроек ToolConfig."""

    model_config = ConfigDict(extra="forbid")

    # 1. Курация
    allowed_bao_formats: list[str] | None = None
    standard_types: list[str] | None = None
    relation_mode: Literal["exact_only", "include_censored"] | None = None
    pchembl_min_cutoff: float | None = None
    duplicate_agg_method: Literal["median", "mean", "max"] | None = None
    max_duplicate_std: float | None = None
    strip_salts: bool | None = None
    neutralize_charges: bool | None = None
    canonicalize_tautomers: bool | None = None
    keep_chirality: bool | None = None
    scaffold_mode: Literal["bemis_murcko", "generic_framework"] | None = None
    chembl_api_limit: int | None = None
    min_neighborhood_rows_fallback: int | None = None
    top_scaffolds_limit: int | None = None

    # 2. SAR и SALI
    mmpa_max_cuts: Literal[1, 2, 3] | None = None
    mmpa_max_cut_bonds: int | None = None
    min_support_pairs: int | None = None
    min_mean_delta_pic50: float | None = None
    max_rgroup_heavy_atoms: int | None = None
    mmpa_min_core_heavy_atoms: int | None = None
    fp_type: Literal["ecfp", "fcfp", "rdkit", "maccs"] | None = None
    fp_radius: Literal[1, 2, 3] | None = None
    fp_nbits: Literal[1024, 2048, 4096] | None = None
    similarity_metric: Literal["tanimoto", "dice", "cosine"] | None = None
    cliff_similarity_cutoff: float | None = None
    cliff_delta_pic50_cutoff: float | None = None
    sali_epsilon: float | None = None
    min_rules_before_enrichment: int | None = None
    background_target_id: str | None = None
    prioritize_positive_rules: bool | None = None

    # 3. Генерация и биоизостеры
    generation_engine: Literal["mmp_and_crem", "mmp_only", "crem_only"] | None = None
    crem_mode: Literal["mutate", "grow"] | None = None
    crem_context_radius: Literal[1, 2, 3] | None = None
    crem_min_rel_atoms: int | None = None
    crem_max_rel_atoms: int | None = None
    crem_max_replacements: int | None = None
    freeze_murcko_core: bool | None = None
    protected_smarts: str | None = None
    modifiable_atom_indices: list[int] | None = None
    neighborhood_tanimoto: float | None = None
    max_candidates_pool: int | None = None
    top_k_output: int | None = None
    max_parent_seeds: int | None = None
    use_bioisostere_catalog: bool | None = None
    bioisostere_fallback_threshold: int | None = None
    crem_max_fragment_heavy_atoms: int | None = None
    ga_population_size: int | None = None
    ga_generations: int | None = None
    enable_crem_grow: bool | None = None

    # 4. Оценка, дескрипторы, фильтры, BRICS, ADMET, MPO и 3D
    active_filter_catalogs: list[FilterCatalogLiteral] | None = None
    custom_ban_smarts: list[str] | None = None
    extra_rdkit_descriptors: list[str] | None = None
    brenk_ignore_long_chain: bool | None = None
    brics_min_fragment_size: int | None = None
    brics_only_leaf_nodes: bool | None = None
    admet_backend: ADMETBackendLiteral | None = None
    cns_profile_mode: (
        Literal["peripheral_low_bbb", "cns_high_bbb", "neutral"] | None
    ) = None
    alert_penalty_mode: Literal["veto", "soft_penalty", "ignore"] | None = None
    alert_soft_multiplier: float | None = None
    mpo_weights: MPOWeightsOverride | None = None
    etkdg_variant: Literal["ETKDGv3", "ETKDGv2"] | None = None
    force_field: Literal["MMFF94s", "MMFF94", "UFF", "none"] | None = None
    num_conformers: int | None = None
    lipinski_mw_cutoff: float | None = None
    lipinski_logp_cutoff: float | None = None
    lipinski_hbd_cutoff: int | None = None
    lipinski_hba_cutoff: int | None = None
    conformer_opt_max_iters: int | None = None
    docking_mode: Literal["auto_top_k", "manual_only", "skip"] | None = None
    docking_exhaustiveness: int | None = None
    docking_top_k: int | None = None
    custom_pdb_id: str | None = None
    custom_box_center: list[float] | None = None

    # 5. Оркестрация
    max_iterations: int | None = None
    max_refilter_iterations: int | None = None
    strict_assay_homogeneity: bool | None = None
    strict_property_thresholds: bool | None = None

    def to_override_dict(self) -> dict[str, Any]:
        """Преобразует непустые переопределения в словарь для ToolConfig.apply_overrides."""
        raw = self.model_dump(mode="python", exclude_none=True)
        if isinstance(raw.get("mpo_weights"), dict):
            weights = {
                k: float(v) for k, v in raw["mpo_weights"].items() if v is not None
            }
            if weights:
                raw["mpo_weights"] = weights
            else:
                raw.pop("mpo_weights", None)
        return raw


class ParsedQueryWithConfig(BaseModel):
    """Объединенный структурированный вывод парсера NL-запроса."""

    model_config = ConfigDict(extra="forbid")

    constraints: QueryConstraints
    tool_config_overrides: ToolConfigOverrides = Field(
        default_factory=ToolConfigOverrides
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
    # 1. СЕМАНТИЧЕСКИЙ РАЗБОР ВХОДНОГО ЗАПРОСА В QueryConstraints + ToolConfig
    # ========================================================================
    @staticmethod
    def _sync_constraints_and_tool_overrides(
        constraints: QueryConstraints,
        overrides: dict[str, Any],
    ) -> tuple[QueryConstraints, dict[str, Any]]:
        """Синхронизирует пересекающиеся настройки QueryConstraints и ToolConfigOverrides."""
        synced_overrides = dict(overrides)
        c_updates: dict[str, Any] = {}

        if "freeze_murcko_core" in synced_overrides:
            c_updates["freeze_scaffold"] = bool(synced_overrides["freeze_murcko_core"])
        elif not constraints.freeze_scaffold:
            synced_overrides["freeze_murcko_core"] = False

        if "top_k_output" in synced_overrides:
            top_v = int(synced_overrides["top_k_output"])
            if 1 <= top_v <= 20:
                c_updates["top_k_candidates"] = top_v
        elif constraints.top_k_candidates != 5:
            synced_overrides["top_k_output"] = constraints.top_k_candidates

        if synced_overrides.get("alert_penalty_mode") == "ignore":
            c_updates["require_no_pains"] = False
        elif not constraints.require_no_pains and "alert_penalty_mode" not in synced_overrides:
            synced_overrides["alert_penalty_mode"] = "ignore"

        if c_updates:
            constraints = constraints.model_copy(update=c_updates)
        return constraints, synced_overrides

    def parse_query_with_tool_config(
        self,
        user_query: str,
        uploaded_csv_path: str | None = None,
    ) -> tuple[QueryConstraints, dict[str, Any], str | None]:
        """Парсит запрос пользователя в `(QueryConstraints, tool_config_overrides, fallback_triggered)`."""
        # Детерминированный экстрактор работает всегда как страховка для точных технических флагов
        det_overrides = self._deterministic_parse_tool_config(user_query)

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
                    max_retries=0,
                )
                structured_llm = llm.with_structured_output(
                    ParsedQueryWithConfig,
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
                if isinstance(res, ParsedQueryWithConfig):
                    merged_overrides = {
                        **res.tool_config_overrides.to_override_dict(),
                        **det_overrides,
                    }
                    c, synced = self._sync_constraints_and_tool_overrides(
                        res.constraints, merged_overrides
                    )
                    return c, synced, None
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
                    ParsedQueryWithConfig,
                    method="function_calling",
                )
                res = structured_ollama.invoke(
                    [
                        SystemMessage(content=QUERY_PARSER_SYSTEM_PROMPT),
                        HumanMessage(content=f"Запрос исследователя: {user_query}"),
                    ]
                )
                if isinstance(res, ParsedQueryWithConfig):
                    merged_overrides = {
                        **res.tool_config_overrides.to_override_dict(),
                        **det_overrides,
                    }
                    c, synced = self._sync_constraints_and_tool_overrides(
                        res.constraints, merged_overrides
                    )
                    return c, synced, "ollama_local_llm"
            except Exception:  # noqa: BLE001, S110
                pass

        # Уровень 3: Детерминированный Rule-Based Engine
        deterministic_constraints = self._deterministic_parse_query(
            user_query, uploaded_csv_path
        )
        c, synced = self._sync_constraints_and_tool_overrides(
            deterministic_constraints, det_overrides
        )
        return c, synced, "deterministic_rule_based_engine"

    def parse_query_constraints(
        self,
        user_query: str,
        uploaded_csv_path: str | None = None,
    ) -> tuple[QueryConstraints, str | None]:
        """Совместимый контракт разбора запроса: возвращает `(QueryConstraints, fallback_triggered)`."""
        constraints, _overrides, fallback = self.parse_query_with_tool_config(
            user_query, uploaded_csv_path
        )
        return constraints, fallback

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

        # 9. Явное отключение фиксации скаффолда
        freeze_scaffold = not (
            "не фиксируй скаффолд" in text_lower
            or "не сохраняй каркас" in text_lower
            or "разморозить скаффолд" in text_lower
            or "freeze_murcko_core=false" in text_lower
            or "freeze_scaffold=false" in text_lower
        )

        return QueryConstraints(
            scenario_type=scenario_type,
            target_chembl_id=target_chembl_id,
            lead_smiles=lead_smiles,
            freeze_scaffold=freeze_scaffold,
            min_logp=min_logp,
            max_logp=max_logp,
            max_sa_score=max_sa_score,
            max_herg_prob=max_herg_prob,
            require_no_pains=require_no_pains,
            top_k_candidates=top_k,
            preferred_substituent_types=["any"],
        )

    def _deterministic_parse_tool_config(self, user_query: str) -> dict[str, Any]:
        """Детерминированно извлекает явно упомянутые настройки ToolConfig из текста запроса."""
        text = user_query.strip()
        text_lower = text.lower()
        overrides: dict[str, Any] = {}

        # --- 1. Курация ---
        bao_codes = re.findall(r"\b(BAO_\d+)\b", text, re.IGNORECASE)
        if bao_codes:
            overrides["allowed_bao_formats"] = list(
                dict.fromkeys(c.upper() for c in bao_codes)
            )

        if (
            "standard_types" in text_lower
            or "только ki" in text_lower
            or "только ic50" in text_lower
            or "только ec50" in text_lower
            or "только kd" in text_lower
            or "типы активности" in text_lower
            or "измерениях" in text_lower
            or "измерения " in text_lower
            or re.search(r"\bпо\s+(?:ic50|ki|kd|ec50)\b", text_lower)
        ):
            stypes = re.findall(r"\b(IC50|Ki|Kd|EC50)\b", text)
            if stypes:
                overrides["standard_types"] = list(dict.fromkeys(stypes))

        if (
            "exact_only" in text_lower
            or "только точные" in text_lower
            or "без цензурир" in text_lower
        ):
            overrides["relation_mode"] = "exact_only"
        elif "include_censored" in text_lower or "включая цензурир" in text_lower:
            overrides["relation_mode"] = "include_censored"

        m_pchembl = re.search(
            r"(?:pchembl_min_cutoff|min_pchembl|pchembl)\s*(?:>=|>|не\s*ниже|от|=|:)?\s*(\d+(?:\.\d+)?)",
            text_lower,
        )
        if m_pchembl:
            overrides["pchembl_min_cutoff"] = float(m_pchembl.group(1))

        if (
            "duplicate_agg_method=mean" in text_lower
            or "агрег" in text_lower
            and "средн" in text_lower
            or "по среднему" in text_lower
        ):
            overrides["duplicate_agg_method"] = "mean"
        elif (
            "duplicate_agg_method=max" in text_lower
            or "агрег" in text_lower
            and "максим" in text_lower
            or "по максимуму" in text_lower
        ):
            overrides["duplicate_agg_method"] = "max"
        elif "duplicate_agg_method=median" in text_lower:
            overrides["duplicate_agg_method"] = "median"

        m_dup_std = re.search(
            r"(?:max_duplicate_std|разброс\s*дубликат\w*|std\s*дубликат\w*)\s*(?:<=|<|=|:)?\s*(\d+(?:\.\d+)?)",
            text_lower,
        )
        if m_dup_std:
            overrides["max_duplicate_std"] = float(m_dup_std.group(1))

        if (
            "не удаляй соли" in text_lower
            or "не удалять соли" in text_lower
            or "сохраняй соли" in text_lower
            or "strip_salts=false" in text_lower
        ):
            overrides["strip_salts"] = False
        if (
            "не нейтрализуй" in text_lower
            or "сохраняй заряды" in text_lower
            or "neutralize_charges=false" in text_lower
        ):
            overrides["neutralize_charges"] = False
        if (
            "не канонизируй таутомер" in text_lower
            or "без канонизации таутомеров" in text_lower
            or "canonicalize_tautomers=false" in text_lower
        ):
            overrides["canonicalize_tautomers"] = False
        if (
            "без хиральности" in text_lower
            or "удали стереохим" in text_lower
            or "удали хиральность" in text_lower
            or "keep_chirality=false" in text_lower
        ):
            overrides["keep_chirality"] = False

        if (
            re.search(r"\bgeneric[\s_-]*framework\b", text_lower)
            or "общий граф скаффолд" in text_lower
            or "универсальный каркас" in text_lower
        ):
            overrides["scaffold_mode"] = "generic_framework"

        # --- 2. SAR и SALI ---
        m_cuts = re.search(
            r"(?:mmpa_max_cuts\s*(?:=|:)?\s*|макс\.?\s*)([123])(?:\s*(?:разрез\w*|cuts?))?",
            text_lower,
        ) or re.search(r"\b([123])\s*(?:разрез\w*|cuts?)\b", text_lower)
        if m_cuts and ("mmpa_max_cuts" in text_lower or "разрез" in text_lower or "cut" in text_lower):
            overrides["mmpa_max_cuts"] = int(m_cuts.group(1))

        m_bonds = re.search(
            r"(?:mmpa_max_cut_bonds|max_cut_bonds)\s*(?:=|:)?\s*(\d+)", text_lower
        )
        if m_bonds:
            overrides["mmpa_max_cut_bonds"] = int(m_bonds.group(1))

        m_sup = re.search(
            r"(?:min_support_pairs\s*(?:=|:)?\s*|минимум\s+|не\s*менее\s+)(\d+)\s*(?:подтверждающ\w*\s*)?пар",
            text_lower,
        ) or re.search(r"min_support_pairs\s*(?:=|:)?\s*(\d+)", text_lower)
        if m_sup:
            overrides["min_support_pairs"] = int(m_sup.group(1))

        m_dpic = re.search(
            r"(?:min_mean_delta_pic50|мин\.?\s*(?:средн\w*\s*)?(?:delta|Δ)_?pic50)\s*(?:>=|>|=|:)?\s*([+-]?\d+(?:\.\d+)?)",
            text_lower,
        )
        if m_dpic:
            overrides["min_mean_delta_pic50"] = float(m_dpic.group(1))

        m_rg = re.search(
            r"(?:max_rgroup_heavy_atoms|макс\.?\s*тяж[её]л\w*\s*атомов\s*(?:в\s*)?r-?групп\w*)\s*(?:<=|=|:)?\s*(\d+)",
            text_lower,
        )
        if m_rg:
            overrides["max_rgroup_heavy_atoms"] = int(m_rg.group(1))

        m_core_ha = re.search(
            r"(?:mmpa_min_core_heavy_atoms|мин\.?\s*атомов\s*ядра)\s*(?:>=|=|:)?\s*(\d+)",
            text_lower,
        )
        if m_core_ha:
            overrides["mmpa_min_core_heavy_atoms"] = int(m_core_ha.group(1))

        if "ecfp6" in text_lower:
            overrides["fp_type"] = "ecfp"
            overrides["fp_radius"] = 3
        elif "ecfp4" in text_lower:
            overrides["fp_type"] = "ecfp"
            overrides["fp_radius"] = 2
        elif "ecfp2" in text_lower:
            overrides["fp_type"] = "ecfp"
            overrides["fp_radius"] = 1
        elif "fcfp6" in text_lower:
            overrides["fp_type"] = "fcfp"
            overrides["fp_radius"] = 3
        elif "fcfp4" in text_lower:
            overrides["fp_type"] = "fcfp"
            overrides["fp_radius"] = 2
        elif "fcfp" in text_lower:
            overrides["fp_type"] = "fcfp"
        elif re.search(r"\bmaccs\b", text_lower):
            overrides["fp_type"] = "maccs"
        elif re.search(
            r"(?:\brdkit\s*(?:fp|фингерпринт\w*|fingerprint\w*)|fp_type\s*(?:=|:)\s*rdkit\b)",
            text_lower,
        ):
            overrides["fp_type"] = "rdkit"

        m_rad = re.search(
            r"(?:fp_radius|радиус\s*(?:фингерпринт\w*|морган\w*|fp))\s*(?:=|:)?\s*([123])",
            text_lower,
        )
        if m_rad:
            overrides["fp_radius"] = int(m_rad.group(1))

        m_bits = re.search(
            r"(?:fp_nbits\s*(?:=|:)?\s*(1024|2048|4096)|\b(1024|2048|4096)\s*(?:бит|bits?)\b)",
            text_lower,
        )
        if m_bits:
            bits_val = int(m_bits.group(1) or m_bits.group(2))
            overrides["fp_nbits"] = bits_val

        if re.search(r"(?:\bdice\b|дайс)", text_lower):
            overrides["similarity_metric"] = "dice"
        elif re.search(r"(?:\bcosine\b|косинус)", text_lower):
            overrides["similarity_metric"] = "cosine"
        elif re.search(
            r"(?:similarity_metric\s*(?:=|:)\s*tanimoto|метрик\w*\s*танимото)",
            text_lower,
        ):
            overrides["similarity_metric"] = "tanimoto"

        m_csim = re.search(
            r"(?:cliff_similarity_cutoff|cliff[\s_]*sim(?:ilarity)?|порог\s*сходства\s*(?:клиффов|sali))\s*(?:>=|>|=|:)?\s*(0?\.\d+|1\.0)",
            text_lower,
        )
        if m_csim:
            overrides["cliff_similarity_cutoff"] = float(m_csim.group(1))

        m_cdelta = re.search(
            r"(?:cliff_delta_pic50_cutoff|порог\s*(?:перепада|дельта)\s*sali)\s*(?:>=|>|=|:)?\s*(\d+(?:\.\d+)?)",
            text_lower,
        )
        if m_cdelta:
            overrides["cliff_delta_pic50_cutoff"] = float(m_cdelta.group(1))

        m_eps = re.search(
            r"(?:sali[\s_]*epsilon|epsilon\s*sali|эпсилон\s*sali)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)",
            text_lower,
        )
        if m_eps:
            overrides["sali_epsilon"] = float(m_eps.group(1))

        # --- 3. Генерация и биоизостеры ---
        if re.search(r"\bmmp[\s_-]*only\b", text_lower) or "только mmp" in text_lower:
            overrides["generation_engine"] = "mmp_only"
        elif re.search(r"\bcrem[\s_-]*only\b", text_lower) or "только crem" in text_lower:
            overrides["generation_engine"] = "crem_only"
        elif re.search(r"\bmmp[\s_-]*and[\s_-]*crem\b", text_lower):
            overrides["generation_engine"] = "mmp_and_crem"

        if (
            "crem_mode=grow" in text_lower
            or "режим grow" in text_lower
            or "выращивани" in text_lower
            or (re.search(r"\bgrow\b", text_lower) and ("crem" in text_lower or "radius" in text_lower))
        ):
            overrides["crem_mode"] = "grow"
        elif "crem_mode=mutate" in text_lower or "режим mutate" in text_lower:
            overrides["crem_mode"] = "mutate"

        m_crad = re.search(
            r"(?:crem_context_radius|crem\s*(?:радиус|radius)|\bradius\b)\s*(?:=|:)?\s*([123])",
            text_lower,
        )
        if m_crad:
            overrides["crem_context_radius"] = int(m_crad.group(1))

        m_crem_min = re.search(
            r"crem_min_rel_atoms\s*(?:=|:)?\s*([+-]?\d+)", text_lower
        )
        if m_crem_min:
            overrides["crem_min_rel_atoms"] = int(m_crem_min.group(1))

        m_crem_max = re.search(
            r"crem_max_rel_atoms\s*(?:=|:)?\s*([+-]?\d+)", text_lower
        )
        if m_crem_max:
            overrides["crem_max_rel_atoms"] = int(m_crem_max.group(1))

        m_crem_rep = re.search(
            r"crem_max_replacements\s*(?:=|:)?\s*(\d+)", text_lower
        )
        if m_crem_rep:
            overrides["crem_max_replacements"] = int(m_crem_rep.group(1))

        if (
            "не фиксируй скаффолд" in text_lower
            or "не замораживать скаффолд" in text_lower
            or "не сохраняй каркас" in text_lower
            or "разморозить скаффолд" in text_lower
            or "freeze_murcko_core=false" in text_lower
        ):
            overrides["freeze_murcko_core"] = False

        m_prot = re.search(
            r"(?:protected[\s_]*smarts|защищ\w*\s*smarts|фиксир\w*\s*smarts)\s*(?:=|:)?\s*['\"]?([A-Za-z0-9@+\-\[\]\(\)=#:\*\\/]+)['\"]?",
            text,
            re.IGNORECASE,
        )
        if m_prot:
            overrides["protected_smarts"] = m_prot.group(1).strip()

        m_mod = re.search(
            r"(?:modifiable_atom_indices|модифицируем\w*\s*индекс\w*\s*атом\w*|модифицируем\w*\s*атом\w*)\s*(?:=|:)?\s*\[?(\d+(?:\s*,\s*\d+)*)\]?",
            text_lower,
        )
        if m_mod:
            overrides["modifiable_atom_indices"] = [
                int(x.strip()) for x in m_mod.group(1).split(",") if x.strip()
            ]

        m_nt = re.search(
            r"(?:neighborhood_tanimoto|окрестност\w*\s*(?:лидера|танимото))\s*(?:>=|>|=|:)?\s*(0?\.\d+|1\.0)",
            text_lower,
        )
        if m_nt:
            overrides["neighborhood_tanimoto"] = float(m_nt.group(1))

        m_pool = re.search(
            r"(?:max_candidates_pool|размер\s*пула\s*кандидатов|пул\s*кандидатов)\s*(?:=|:)?\s*(\d+)",
            text_lower,
        )
        if m_pool:
            overrides["max_candidates_pool"] = int(m_pool.group(1))

        m_seeds = re.search(
            r"(?:max_parent_seeds|родительских\s*структур|число\s*родителе\w*)\s*(?:=|:)?\s*(\d+)",
            text_lower,
        )
        if m_seeds:
            overrides["max_parent_seeds"] = int(m_seeds.group(1))

        m_topk = re.search(
            r"(?:top_k_output\s*(?:=|:)?\s*(\d+)|\b(?:топ|top)\s*-?\s*(\d+)(?:\s*(?:кандидат\w*|молекул\w*|структур\w*))?)",
            text_lower,
        )
        if m_topk:
            topk_val = int(m_topk.group(1) or m_topk.group(2))
            if 1 <= topk_val <= 50:
                overrides["top_k_output"] = topk_val

        if (
            "без биоизостеров" in text_lower
            or "отключи биоизостеры" in text_lower
            or "use_bioisostere_catalog=false" in text_lower
        ):
            overrides["use_bioisostere_catalog"] = False

        m_ga_pop = re.search(
            r"(?:ga_population_size\s*(?:=|:)?\s*(\d+)|популяц\w*\s*(?:graphga|ga)?\s*(?:=|:|\s)\s*(\d+))",
            text_lower,
        )
        if m_ga_pop:
            ga_pop_val = int(m_ga_pop.group(1) or m_ga_pop.group(2))
            if 6 <= ga_pop_val <= 60:
                overrides["ga_population_size"] = ga_pop_val

        m_ga_gen = re.search(
            r"(?:ga_generations\s*(?:=|:)?\s*(\d+)|поколен\w*\s*(?:эволюц\w*|graphga|ga)?\s*(?:=|:|\s)\s*(\d+))",
            text_lower,
        )
        if m_ga_gen:
            ga_gen_val = int(m_ga_gen.group(1) or m_ga_gen.group(2))
            if 1 <= ga_gen_val <= 5:
                overrides["ga_generations"] = ga_gen_val

        if (
            "без crem grow" in text_lower
            or "отключи crem grow" in text_lower
            or "enable_crem_grow=false" in text_lower
        ):
            overrides["enable_crem_grow"] = False
        elif "enable_crem_grow=true" in text_lower:
            overrides["enable_crem_grow"] = True

        # --- 4. Фильтры, дескрипторы, BRICS, ADMET, MPO, 3D ---
        catalog_tokens: list[FilterCatalogLiteral] = []
        has_extra_catalog = False
        for cat_name in ("Glaxo", "SureChEMBL", "NIH", "ZINC"):
            if re.search(rf"\b{cat_name}\b", text, re.IGNORECASE):
                has_extra_catalog = True
                catalog_tokens.append(cat_name)
        for cat_name in ("PAINS_A", "PAINS_B", "PAINS_C", "Brenk"):
            if re.search(rf"\b{cat_name}\b", text, re.IGNORECASE):
                catalog_tokens.append(cat_name)
        if re.search(r"\bPAINS\b", text, re.IGNORECASE) and not any(
            p in catalog_tokens for p in ("PAINS_A", "PAINS_B", "PAINS_C")
        ):
            catalog_tokens.extend(["PAINS_A", "PAINS_B", "PAINS_C"])

        explicit_catalog_cmd = (
            has_extra_catalog
            or "active_filter_catalogs" in text_lower
            or "каталог" in text_lower
            or "фильтр" in text_lower
            or any(p in text for p in ("PAINS_A", "PAINS_B", "PAINS_C"))
        )
        if explicit_catalog_cmd and catalog_tokens:
            if any(w in text_lower for w in ("добавь", "включи также", "плюс")):
                merged_cats = list(
                    dict.fromkeys(
                        ["PAINS_A", "PAINS_B", "PAINS_C", "Brenk", *catalog_tokens]
                    )
                )
                overrides["active_filter_catalogs"] = merged_cats
            else:
                overrides["active_filter_catalogs"] = list(
                    dict.fromkeys(catalog_tokens)
                )

        m_ban = re.search(
            r"(?:custom_ban_smarts|запрещенн\w*\s*smarts|забань\s*smarts)\s*(?:=|:)?\s*\[?([^;\n]+)\]?",
            text,
            re.IGNORECASE,
        )
        if m_ban:
            raw_ban = m_ban.group(1)
            ban_items = [
                s.strip().strip("'\"") for s in raw_ban.split(",") if s.strip().strip("'\"")
            ]
            if ban_items:
                overrides["custom_ban_smarts"] = ban_items

        m_extra_desc = re.search(
            r"(?:extra_rdkit_descriptors|доп\w*\s*дескриптор\w*)\s*(?:=|:)?\s*\[?([A-Za-z0-9_,\s]+)\]?",
            text,
            re.IGNORECASE,
        )
        found_descs: list[str] = []
        if m_extra_desc:
            for tok in re.split(r"[\s,]+", m_extra_desc.group(1)):
                if tok in _KNOWN_RDKIT_DESCRIPTOR_NAMES and tok not in found_descs:
                    found_descs.append(tok)
        for cand_tok in re.findall(r"\b([A-Z][A-Za-z0-9_]{2,})\b", text):
            if (
                cand_tok
                in (
                    "BertzCT",
                    "MolMR",
                    "HeavyAtomCount",
                    "RingCount",
                    "NumAromaticRings",
                    "BalabanJ",
                    "HallKierAlpha",
                    "LabuteASA",
                    "NumHeteroatoms",
                )
                and cand_tok in _KNOWN_RDKIT_DESCRIPTOR_NAMES
                and cand_tok not in found_descs
            ):
                found_descs.append(cand_tok)
        if found_descs:
            overrides["extra_rdkit_descriptors"] = found_descs

        if (
            "brenk_ignore_long_chain=false" in text_lower
            or "не игнорируй длинные цепи" in text_lower
        ):
            overrides["brenk_ignore_long_chain"] = False

        m_brics_sz = re.search(
            r"(?:brics_min_fragment_size|мин\.?\s*размер\s*(?:фрагмента\s*)?brics)\s*(?:>=|=|:)?\s*(\d+)",
            text_lower,
        )
        if m_brics_sz:
            overrides["brics_min_fragment_size"] = int(m_brics_sz.group(1))

        if (
            "brics_only_leaf_nodes=false" in text_lower
            or "нелистовые узлы brics" in text_lower
            or "промежуточные фрагменты brics" in text_lower
        ):
            overrides["brics_only_leaf_nodes"] = False

        for backend_pattern, backend_val in (
            (r"\bempirical[\s_-]*only\b", "empirical_only"),
            (r"\brdkit[\s_-]*analytical[\s_-]*qsar\b", "rdkit_analytical_qsar"),
            (r"\bcatboost[\s_-]*local[\s_-]*qsar\b", "catboost_local_qsar"),
            (r"\badmet[\s_-]*ai[\s_-]*dmpnn\b", "admet_ai_dmpnn"),
            (r"\bauto[\s_-]*catboost\b", "auto_catboost"),
        ):
            if re.search(backend_pattern, text_lower):
                overrides["admet_backend"] = backend_val
                break
        if "admet_backend" not in overrides and (
            "только эмпирический admet" in text_lower
            or "без ml admet" in text_lower
        ):
            overrides["admet_backend"] = "empirical_only"

        if (
            "cns_high_bbb" in text_lower
            or "проникновение в цнс" in text_lower
            or "через гэб" in text_lower
            or "через гтб" in text_lower
            or "цнс гтб" in text_lower
            or "цнс гэб" in text_lower
            or "для цнс" in text_lower
        ):
            overrides["cns_profile_mode"] = "cns_high_bbb"
        elif "cns_profile_mode=neutral" in text_lower or "нейтральный профиль г" in text_lower:
            overrides["cns_profile_mode"] = "neutral"
        elif "peripheral_low_bbb" in text_lower:
            overrides["cns_profile_mode"] = "peripheral_low_bbb"

        if re.search(r"\bsoft[\s_-]*penalty\b", text_lower) or "мягкий штраф" in text_lower:
            overrides["alert_penalty_mode"] = "soft_penalty"
        elif "alert_penalty_mode=ignore" in text_lower or "игнорируй алерты" in text_lower:
            overrides["alert_penalty_mode"] = "ignore"
        elif "alert_penalty_mode=veto" in text_lower or "жесткое вето" in text_lower:
            overrides["alert_penalty_mode"] = "veto"

        m_mult = re.search(
            r"(?:alert_soft_multiplier|soft[\s_-]*penalty|множитель\s*штрафа)\s*(?:=|:)?\s*(0?\.\d+|1\.0)",
            text_lower,
        )
        if m_mult:
            overrides["alert_soft_multiplier"] = float(m_mult.group(1))

        weight_matches = re.findall(
            r"(?:вес|weight)\s+(pic50|logp|sascore|qed|tpsa|logs|herg)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)",
            text_lower,
        )
        if weight_matches:
            overrides["mpo_weights"] = {
                wk: float(wv) for wk, wv in weight_matches
            }

        if re.search(r"\bETKDGv2\b", text, re.IGNORECASE):
            overrides["etkdg_variant"] = "ETKDGv2"
        elif re.search(r"\bETKDGv3\b", text, re.IGNORECASE):
            overrides["etkdg_variant"] = "ETKDGv3"

        if re.search(r"\bMMFF94s\b", text, re.IGNORECASE):
            overrides["force_field"] = "MMFF94s"
        elif re.search(r"\bMMFF94\b", text, re.IGNORECASE):
            overrides["force_field"] = "MMFF94"
        elif re.search(r"\bUFF\b", text, re.IGNORECASE):
            overrides["force_field"] = "UFF"
        elif (
            "force_field=none" in text_lower
            or "без оптимизации конформеров" in text_lower
        ):
            overrides["force_field"] = "none"

        m_confs = re.search(
            r"(?:num_conformers\s*(?:=|:)?\s*(\d+)|(\d+)\s*конформер\w*)",
            text_lower,
        )
        if m_confs:
            overrides["num_conformers"] = int(m_confs.group(1) or m_confs.group(2))

        # --- 5. Оркестрация ---
        m_max_it = re.search(
            r"(?:max_iterations|макс\.?\s*итерац\w*\s*(?:дизайна)?)\s*(?:=|:)?\s*(\d+)",
            text_lower,
        )
        if m_max_it:
            overrides["max_iterations"] = int(m_max_it.group(1))

        m_ref_it = re.search(
            r"(?:max_refilter_iterations|итерац\w*\s*рекурац\w*)\s*(?:=|:)?\s*(\d+)",
            text_lower,
        )
        if m_ref_it:
            overrides["max_refilter_iterations"] = int(m_ref_it.group(1))

        if (
            "strict_assay_homogeneity=false" in text_lower
            or "игнорируй неоднородность bao" in text_lower
            or "разреши смешанные тесты" in text_lower
        ):
            overrides["strict_assay_homogeneity"] = False

        if (
            "strict_property_thresholds=false" in text_lower
            or "мягкие пороги свойств" in text_lower
        ):
            overrides["strict_property_thresholds"] = False

        # --- 6. Физический 3D-докинг ---
        if (
            "без докинга" in text_lower
            or "отключи докинг" in text_lower
            or "пропусти докинг" in text_lower
            or "docking_mode=skip" in text_lower
        ):
            overrides["docking_mode"] = "skip"
        elif (
            "докинг вручную" in text_lower
            or "ручной докинг" in text_lower
            or "docking_mode=manual_only" in text_lower
        ):
            overrides["docking_mode"] = "manual_only"
        elif "docking_mode=auto_top_k" in text_lower:
            overrides["docking_mode"] = "auto_top_k"

        m_pdb = re.search(r"\bpdb\s*(?:id)?\s*[:=]?\s*([1-9][a-z0-9]{3})\b", text, re.IGNORECASE)
        if m_pdb:
            overrides["custom_pdb_id"] = m_pdb.group(1).upper()

        return overrides

    # ========================================================================
    # 2. ФОРМИРОВАНИЕ КОРРЕКТИРУЮЩИХ ОГРАНИЧЕНИЙ ДЛЯ ПЕТЛИ №2 (Critic Agent)
    # ========================================================================
    def formulate_corrective_constraints(
        self,
        current_constraints: QueryConstraints,
        evaluated_candidates: list[CandidateMolecule],
        critic_summary_reasons: list[str],
        *,
        tool_config: ToolConfig | None = None,
    ) -> tuple[QueryConstraints, str, str | None]:
        """Анализирует причины отклонения кандидатов и формирует директиву самокоррекции для Петли №2.

        Возвращает `(updated_constraints, critic_feedback, fallback_triggered)`.
        """
        fallback_tpsa = (
            tool_config.critic_fallback_min_tpsa
            if tool_config is not None
            else 85.0
        )
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
                    max_retries=0,
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
                        or fallback_tpsa,
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
                "min_tpsa": current_constraints.min_tpsa or fallback_tpsa,
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
                    max_retries=0,
                )
                structured_llm = llm.with_structured_output(
                    BatchHypothesisRationales,
                    method="json_schema",
                    strict=True,
                )
                def _docking_fact_line(cand: CandidateMolecule) -> str:
                    if cand.docking is None or cand.docking.status != "SUCCESS":
                        return "- 3D-моделирование / Докинг: не выполнялся"
                    contacts = "; ".join(
                        f"{k}: {', '.join(v)}"
                        for k, v in cand.docking.interactions.items()
                        if v
                    ) or "контакты не обнаружены"
                    dg_str = (
                        f"{cand.docking.docking_score_kcal:.2f} ккал/моль"
                        if cand.docking.docking_score_kcal is not None
                        else "N/A"
                    )
                    return (
                        f"- 3D-моделирование / Докинг ({cand.docking.pdb_id or 'PDB'}, {cand.docking.backend or '3D'}): "
                        f"ΔG={dg_str}, контакты=[{contacts}]"
                    )

                facts_text = "\n".join(
                    f"Кандидат {c.candidate_id}:\n"
                    f"- Родитель: {c.parent_chembl_id} (pIC50={c.parent_exp_pic50:.2f}, BAO={c.parent_bao_format})\n"
                    f"- Правило/метод: {c.hypothesis_rationale.applied_smirks} ({c.hypothesis_rationale.rule_id}, origin={c.hypothesis_rationale.generation_origin})\n"
                    f"- Статистика MMP: support={c.hypothesis_rationale.support_pairs_count}, ΔpIC50={c.hypothesis_rationale.historical_mean_delta_pic50:+.2f}, однороден={c.hypothesis_rationale.assay_homogeneous}\n"
                    f"- Детерминированный сдвиг: ΔcLogP={c.calc_rdkit.delta_clogp_vs_parent:+.2f}, ΔTPSA={c.calc_rdkit.delta_tpsa_vs_parent:+.1f}, SAscore={c.calc_rdkit.sa_score:.2f}\n"
                    f"- ADMET: прогноз pIC50={c.pred_admet.predicted_pic50:.2f}, hERG prob={c.pred_admet.herg_liability_prob:.2f} ({c.pred_admet.herg_risk_level}), растворимость={c.pred_admet.solubility_class}\n"
                    f"{_docking_fact_line(c)}\n"
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

        # Детерминированный фолбэк: дополняем базовую гипотезу результатами 3D-докинга при наличии
        enriched_summaries: dict[str, str] = {}
        for c in candidates:
            base_summary = c.hypothesis_rationale.text_summary
            if (
                c.docking is not None
                and c.docking.status == "SUCCESS"
                and c.docking.docking_score_kcal is not None
                and "3D-моделирование / Докинг" not in base_summary
            ):
                contacts_parts = [
                    f"{k}: {', '.join(v[:3])}"
                    for k, v in c.docking.interactions.items()
                    if v
                ]
                contacts_str = (
                    "; ".join(contacts_parts[:3])
                    if contacts_parts
                    else "стерическая комплементарность карману"
                )
                base_summary = (
                    f"{base_summary} [3D-моделирование / Докинг ({c.docking.pdb_id or 'PDB'}): "
                    f"ΔG={c.docking.docking_score_kcal:.2f} ккал/моль; контакты: {contacts_str}.]"
                )
            enriched_summaries[c.candidate_id] = base_summary
        return enriched_summaries, "deterministic_rule_based_engine"

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
                    max_retries=0,
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

        def _dock_cell(cand: CandidateMolecule) -> str:
            if cand.docking is None:
                return "Не запускался"
            if cand.docking.status == "SUCCESS" and cand.docking.docking_score_kcal is not None:
                top_res = [
                    r for vals in cand.docking.interactions.values() for r in vals
                ][:3]
                res_txt = f" ({', '.join(top_res)})" if top_res else ""
                return f"ΔG={cand.docking.docking_score_kcal:.2f} ккал/моль [{cand.docking.pdb_id or 'PDB'}]{res_txt}"
            return f"{cand.docking.status}"

        cands_md = "\n".join(
            f"| `{c.candidate_id}` | `{c.canonical_smiles}` | `{c.parent_chembl_id}` (`{c.parent_exp_pic50:.2f}`) | "
            f"MW={c.calc_rdkit.mw:.1f}, LogP={c.calc_rdkit.clogp:.2f}, SA={c.calc_rdkit.sa_score:.2f} | "
            f"pIC50={c.pred_admet.predicted_pic50:.2f}, hERG={c.pred_admet.herg_liability_prob:.2f} ({c.pred_admet.herg_risk_level}) | "
            f"{_dock_cell(c)} | "
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
            f"| ID | SMILES кандидата | Экспериментальный факт | Детерминированный расчет RDKit | Модельный прогноз ADMET | 3D-моделирование / Докинг | Индекс Харрингтона |\n"
            f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            f"{cands_md}\n\n"
            f"## 5. Ретросинтетическая декомпозиция на доступные строительные блоки (BRICS)\n"
            f"{brics_md}\n\n"
            f"## 6. Рекомендации для органического синтеза\n"
            f"Предложенные структуры сохраняют базовое фармакофорное ядро, демонстрируют оптимизированный баланс "
            f"липофильности и полярной поверхности и декомпозируются на стандартные коммерчески доступные синтоны."
        )


llm_adapter = LLMAdapter()
