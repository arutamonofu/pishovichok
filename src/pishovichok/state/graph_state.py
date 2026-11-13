"""Схема глобального состояния мультиагентного графа LangGraph (Blackboard + Fat Pointers).

В память графа передаются только неизменяемые указатели `ParquetPointer` и компактные
агрегированные сводки, исключая попадание сырых таблиц молекул в контекстное окно LLM.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pishovichok.state.schemas import (
    ActivityCliff,
    CandidateMolecule,
    DatasetCurationSummary,
    ParamSourceLiteral,
    ParquetPointer,
    QueryConstraints,
    SARRule,
    ToolConfig,
    TraceEvent,
)

# Запрещенные ключи в состоянии графа
FORBIDDEN_RAW_COLLECTION_KEYS: frozenset[str] = frozenset(
    {
        "all_molecules",
        "raw_data_df",
        "curated_data_df",
        "mmp_df",
        "full_candidate_list",
        "raw_smiles_list",
    }
)


class SARGraphState(TypedDict):
    """Глобальное состояние мультиагентного графа LangGraph (Blackboard + Fat Pointers)."""

    # Идентификация сессии и входные данные
    session_id: str
    workspace_dir: str
    user_query: str
    uploaded_csv_path: str | None

    # Распознанные ограничения и параметры фильтрации
    query_constraints: QueryConstraints
    tool_config: ToolConfig
    param_sources: dict[str, ParamSourceLiteral]
    node_telemetry: dict[str, dict[str, Any]]
    assay_filter_strict: (
        bool  # Флаг строгой стратификации по BAO (включается Петлей №1)
    )
    active_bao_filter: (
        str | None
    )  # Целевой формат BAO при строгой фильтрации (напр., 'BAO_0000357')

    # Счетчики итераций и фаза выполнения для управления петлями самокоррекции
    last_executed_stage: Literal["init", "research", "sar", "design", "evaluation"]
    data_refilter_iteration: int  # Счетчик Петли №1 (Critic -> Research), макс. 1
    design_iteration: int  # Счетчик Петли №2 (Critic -> Design), макс. 2

    # Толстые указатели (Fat Pointers) на Parquet-таблицы в workspace/<session_id>/
    dataset_uri: ParquetPointer | None  # Указатель на curated_dataset.parquet
    raw_dataset_uri: ParquetPointer | None  # Указатель на raw_dataset.parquet
    mmp_table_uri: ParquetPointer | None  # Указатель на mmp_pairs.parquet
    candidates_table_uri: (
        ParquetPointer | None
    )  # Указатель на evaluated_candidates.parquet

    # Компактные типизированные агрегаты в памяти графа
    curation_summary: DatasetCurationSummary | None
    extracted_sar_rules: list[SARRule]
    detected_activity_cliffs: list[ActivityCliff]
    enriched_with_chembl_background: (
        bool  # Флаг обогащения малой серии CSV фоном ChEMBL (Сценарий 3)
    )
    top_candidates: list[
        CandidateMolecule
    ]  # Ранжированный срез лучших кандидатов текущей итерации (топ-5..20)
    lead_reference_metrics: (
        CandidateMolecule | None
    )  # Карточка исходного лидера для сравнения дельт в Сценарии 2

    # Поля управления и обратной связи Critic Agent
    critic_verdict: (
        Literal["refilter_data", "proceed_to_design", "redesign_molecules", "approve"]
        | None
    )
    critic_feedback: (
        str | None
    )  # Текстовая и структурная директива для Research или Design Agent

    # Накапливаемый журнал трассировки (append-only reducer) и финальный отчет
    trace_log: Annotated[list[TraceEvent], operator.add]
    final_report_markdown: str | None

    # Управление физическим 3D-докингом
    docking_mode: Literal["auto_top_k", "manual_only", "skip"]
    docking_exhaustiveness: int
    docking_top_k: int
    custom_pdb_id: str | None
    custom_box_center: list[float] | None


def resolve_tool_config(state: dict[str, Any] | SARGraphState) -> ToolConfig:
    """Безопасно извлекает ToolConfig из состояния графа (с fallback к дефолтному)."""
    raw = state.get("tool_config") if isinstance(state, dict) else None
    if isinstance(raw, ToolConfig):
        return raw
    if isinstance(raw, dict):
        return ToolConfig.model_validate(raw)
    return ToolConfig()


def update_param_sources(
    existing: dict[str, ParamSourceLiteral] | None,
    changed_keys: list[str] | tuple[str, ...] | set[str],
    source: ParamSourceLiteral,
) -> dict[str, ParamSourceLiteral]:
    """Возвращает обновленный словарь источников параметров с дефолтами для всех полей ToolConfig."""
    sources: dict[str, ParamSourceLiteral] = {
        field_name: "default" for field_name in ToolConfig.model_fields
    }
    if existing:
        sources.update(existing)
    for key in changed_keys:
        if key in ToolConfig.model_fields:
            sources[key] = source
    return sources


def create_initial_graph_state(
    user_query: str,
    session_id: str,
    workspace_dir: str | Path,
    query_constraints: QueryConstraints | None = None,
    uploaded_csv_path: str | Path | None = None,
    tool_config: ToolConfig | None = None,
    param_sources: dict[str, ParamSourceLiteral] | None = None,
) -> SARGraphState:
    """Создает валидированное начальное состояние графа для новой вычислительной сессии."""
    default_constraints = query_constraints or QueryConstraints(
        scenario_type="custom_csv_analysis" if uploaded_csv_path else "target_based_sar"
    )
    resolved_config = tool_config or ToolConfig()
    default_dump = ToolConfig().model_dump(mode="python")
    resolved_dump = resolved_config.model_dump(mode="python")
    initial_sources: dict[str, ParamSourceLiteral] = {
        k: ("ui_manual" if resolved_dump.get(k) != default_dump.get(k) else "default")
        for k in ToolConfig.model_fields
    }
    if param_sources:
        for k, v in param_sources.items():
            if k in ToolConfig.model_fields:
                initial_sources[k] = v

    return SARGraphState(
        session_id=session_id,
        workspace_dir=str(Path(workspace_dir)),
        user_query=user_query,
        uploaded_csv_path=str(uploaded_csv_path)
        if uploaded_csv_path is not None
        else None,
        query_constraints=default_constraints,
        tool_config=resolved_config,
        param_sources=initial_sources,
        node_telemetry={},
        assay_filter_strict=False,
        active_bao_filter=None,
        last_executed_stage="init",
        data_refilter_iteration=0,
        design_iteration=0,
        dataset_uri=None,
        raw_dataset_uri=None,
        mmp_table_uri=None,
        candidates_table_uri=None,
        curation_summary=None,
        extracted_sar_rules=[],
        detected_activity_cliffs=[],
        enriched_with_chembl_background=False,
        top_candidates=[],
        lead_reference_metrics=None,
        critic_verdict=None,
        critic_feedback=None,
        trace_log=[],
        final_report_markdown=None,
        docking_mode=resolved_config.docking_mode,
        docking_exhaustiveness=resolved_config.docking_exhaustiveness,
        docking_top_k=resolved_config.docking_top_k,
        custom_pdb_id=resolved_config.custom_pdb_id,
        custom_box_center=resolved_config.custom_box_center,
    )


def assert_fat_pointer_isolation(state_dict: dict[str, Any]) -> None:
    """Проверяет Инвариант 2 (Fat Pointers): отсутствие сырых таблиц и DataFrame в состоянии графа."""
    illegal_keys = FORBIDDEN_RAW_COLLECTION_KEYS.intersection(state_dict.keys())
    if illegal_keys:
        raise ValueError(
            f"Fat Pointer Invariant violated: raw collection keys {sorted(illegal_keys)} "
            "are prohibited inside SARGraphState."
        )
    for key in (
        "dataset_uri",
        "raw_dataset_uri",
        "mmp_table_uri",
        "candidates_table_uri",
    ):
        val = state_dict.get(key)
        if val is not None and not isinstance(val, ParquetPointer):
            raise TypeError(
                f"Fat Pointer Invariant violated: '{key}' must be a ParquetPointer or None, got {type(val).__name__}"
            )
