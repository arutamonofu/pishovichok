"""Верхняя статусная панель и левый рельс оркестрации агентов (Chemical IDE)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from pishovichok.config import settings
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import (
    ParamSourceLiteral,
    QueryConstraints,
    ToolConfig,
    TraceEvent,
)
from pishovichok.ui.bridge import agent_bridge
from pishovichok.ui.components.chem_widgets import create_sketcher
from pishovichok.ui.session import HITLDecision, SessionRuntime, session_registry
from pishovichok.ui.state_vm import PARAM_SOURCE_BADGE_META

DEMO_SCENARIOS: dict[str, dict[str, str]] = {
    "scenario1_btk": {
        "title": "Сценарий 1: BTK / Ибрутиниб (ChEMBL5251)",
        "scenario_type": "target_based_sar",
        "query": "Найди наиболее активные скаффолды и SAR-правила для мишени CHEMBL5251 (BTK)",
        "lead_smiles": "",
    },
    "scenario2_gefitinib": {
        "title": "Сценарий 2: Гефитиниб / EGFR (ChEMBL203)",
        "scenario_type": "lead_optimization",
        "query": "Оптимизируй Гефитиниб для мишени CHEMBL203 (EGFR) с logP <= 4.0 и без PAINS",
        "lead_smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    },
    "scenario3_wetlab": {
        "title": "Сценарий 3: Локальный Wet-Lab CSV",
        "scenario_type": "custom_csv_analysis",
        "query": "Проанализируй лабораторную серию Wet-Lab CSV и предложи безопасные биоизостеры",
        "lead_smiles": "COc1ccc(Nc2ncnc3ccccc23)cc1",
    },
}

_AGENT_TO_STAGE: dict[str, str] = {
    "ResearchAgent": "research",
    "SARAgent": "sar",
    "DesignAgent": "design",
    "EvaluationAgent": "evaluation",
    "CriticAgent": "critic",
}

_PIPELINE_STAGES: list[tuple[str, str]] = [
    ("research", "Research"),
    ("sar", "SAR"),
    ("design", "Design"),
    ("evaluation", "Evaluation"),
    ("critic", "Critic"),
]


def _compact_telemetry_val(val: Any) -> str:
    """Компактно форматирует значение параметра или счётчика для чипов телеметрии."""
    if isinstance(val, list):
        if len(val) > 3:
            return f"[{', '.join(str(x) for x in val[:2])}, +{len(val) - 2}]"
        return f"[{', '.join(str(x) for x in val)}]"
    if isinstance(val, dict):
        return f"{{{len(val)} items}}"
    if isinstance(val, float):
        return f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
    return str(val)


def format_node_telemetry_chips(
    telemetry: dict[str, Any] | None,
    param_sources: dict[str, ParamSourceLiteral] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    """Преобразует `node_telemetry[node]` в структурированные чипы параметров (с учётом `param_sources`) и счётчиков."""
    if not isinstance(telemetry, dict):
        return [], []
    sources = param_sources or {}
    raw_params = telemetry.get("active_params") or {}
    raw_stats = telemetry.get("stats") or {}

    param_chips: list[dict[str, str]] = []
    if isinstance(raw_params, dict):
        for k, v in raw_params.items():
            if v is None or v == [] or v == "":
                continue
            src: ParamSourceLiteral = sources.get(str(k), "default")
            _, color = PARAM_SOURCE_BADGE_META.get(src, ("default", "slate"))
            param_chips.append(
                {
                    "key": str(k),
                    "label": f"{k}: {_compact_telemetry_val(v)}",
                    "source": src,
                    "color": color,
                }
            )

    stat_chips: list[tuple[str, str]] = []
    if isinstance(raw_stats, dict):
        for k, v in raw_stats.items():
            if k == "duration_ms" or v is None or v == []:
                continue
            stat_chips.append((str(k), _compact_telemetry_val(v)))

    return param_chips, stat_chips


class TopStatusBar:
    """Компактная верхняя панель статусов пайплайна, HITL-режима и загрузки сценариев."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_session_switched: Callable[[str], Any] | None = None,
        on_csv_uploaded: Callable[[Path], Any] | None = None,
        on_toggle_settings: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_session_switched = on_session_switched
        self._on_csv_uploaded = on_csv_uploaded
        self._on_toggle_settings = on_toggle_settings
        self._stage_badges: dict[str, ui.badge] = {}
        self._counters_label: ui.label | None = None
        self._settings_btn: ui.button | None = None
        self._overrides_badge: ui.badge | None = None
        self._hitl_select: ui.select | None = None
        self._scenario_select: ui.select | None = None
        self._build()

    def _build(self) -> None:
        with ui.header(elevated=True).classes(
            "bg-slate-900 border-b border-slate-800 px-4 py-2 items-center justify-between gap-3"
        ):
            with ui.row().classes("items-center gap-3 no-wrap"):
                ui.icon("science", size="sm", color="emerald-400")
                ui.label("Pishovichok · Chemical IDE").classes(
                    "text-sm font-semibold tracking-wide text-slate-100 whitespace-nowrap"
                )
                with ui.row().classes("items-center gap-1 ml-2"):
                    for idx, (stage_key, stage_label) in enumerate(_PIPELINE_STAGES):
                        if idx > 0:
                            ui.icon("chevron_right", size="xs", color="slate-500")
                        badge = ui.badge(stage_label, color="slate").props("outline")
                        badge.classes("text-xs px-2 py-0.5")
                        self._stage_badges[stage_key] = badge

            with ui.row().classes("items-center gap-2 no-wrap"):
                self._counters_label = ui.label("").classes(
                    "text-xs text-slate-300 font-mono whitespace-nowrap mr-2"
                )
                self._hitl_select = (
                    ui.select(
                        options={
                            "auto": "HITL: Авто",
                            "pause_after_scaffolds": "HITL: Пауза после SAR",
                            "pause_before_critic": "HITL: Пауза перед Critic",
                        },
                        value=self.runtime.hitl_mode,
                        on_change=self._handle_hitl_change,
                    )
                    .props("dense outlined dark options-dense")
                    .classes("w-48 text-xs")
                )
                scenario_opts = {k: v["title"] for k, v in DEMO_SCENARIOS.items()}
                self._scenario_select = (
                    ui.select(
                        options=scenario_opts,
                        value=next(iter(scenario_opts)),
                        on_change=None,
                    )
                    .props("dense outlined dark options-dense")
                    .classes("w-60 text-xs")
                )
                ui.button(
                    "Открыть",
                    icon="folder_open",
                    on_click=self._handle_open_scenario,
                ).props("dense flat color=emerald").classes("text-xs")
                ui.button(
                    "Очистить",
                    icon="cleaning_services",
                    on_click=self._handle_clear_session,
                ).props("dense flat color=slate").classes("text-xs").tooltip(
                    "Очистить рабочую область и начать новую сессию"
                )
                self._settings_btn = ui.button(
                    "Ядро",
                    icon="tune",
                    on_click=self._handle_toggle_settings,
                ).props("dense outline color=sky").classes("text-xs").tooltip(
                    "Открыть конфигурацию вычислительного ядра (ToolConfig)"
                )
                with self._settings_btn:
                    self._overrides_badge = ui.badge("0", color="purple").props(
                        "floating"
                    )
                    self._overrides_badge.set_visibility(False)

        self.refresh()

    def _handle_toggle_settings(self) -> None:
        if self._on_toggle_settings is not None:
            self._on_toggle_settings()

    def _handle_hitl_change(self, e: Any) -> None:
        val = str(e.value)
        if val in ("auto", "pause_after_scaffolds", "pause_before_critic"):
            self.runtime.hitl_mode = val  # type: ignore[assignment]

    async def _handle_open_scenario(self) -> None:
        if self._scenario_select is None or not self._scenario_select.value:
            return
        sid = str(self._scenario_select.value)
        restored = session_registry.get(sid) or await session_registry.restore_from_workspace(sid)
        if restored is None:
            ui.notify(f"Сохранённый сценарий '{sid}' не найден", type="warning")
            return
        if self._on_session_switched is not None:
            res = self._on_session_switched(sid)
            if asyncio.iscoroutine(res):
                await res

    async def _handle_clear_session(self) -> None:
        await agent_bridge.cancel_current_run(self.runtime.session_id)
        new_sid = f"sess_{uuid.uuid4().hex[:8]}"
        session_registry.get_or_create(session_id=new_sid)
        if self._on_session_switched is not None:
            res = self._on_session_switched(new_sid)
            if asyncio.iscoroutine(res):
                await res

    async def _handle_csv_upload(self, e: Any) -> None:
        upload_dir = settings.resolved_workspace_dir / self.runtime.session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        target_path = upload_dir / (getattr(e, "name", "uploaded_series.csv") or "uploaded_series.csv")
        content = getattr(e, "content", None)
        if content is not None:
            data = content.read()
            if asyncio.iscoroutine(data):
                data = await data
            target_path.write_bytes(data)
            ui.notify(f"Загружен CSV: {target_path.name}", type="positive")
            if self._on_csv_uploaded is not None:
                res = self._on_csv_uploaded(target_path)
                if asyncio.iscoroutine(res):
                    await res

    def refresh(self) -> None:
        """Обновляет индикаторы этапов и счётчики по текущему состоянию сессии."""
        completed = {
            _AGENT_TO_STAGE.get(ev.agent_name, ev.agent_name.lower())
            for ev in self.runtime.state.get("trace_log", [])
        }
        active = self.runtime.current_node
        is_paused = self.runtime.status == "paused_hitl"
        is_running = self.runtime.status == "running"
        for stage_key, badge in self._stage_badges.items():
            if is_paused and stage_key == active:
                badge.props("color=amber")
                badge.props(remove="outline")
            elif stage_key == active and is_running:
                badge.props("color=sky")
                badge.props(remove="outline")
            elif stage_key in completed:
                badge.props("color=emerald")
                badge.props(remove="outline")
            else:
                badge.props("color=slate outline")

        vm = self.runtime.view_model
        total = len(vm.rows_by_id)
        passed = sum(
            1
            for r in vm.rows_by_id.values()
            if r.passes_all and r.pains_count == 0 and r.brenk_count == 0 and not r.excluded
        )
        rejected = total - passed
        it = int(self.runtime.state.get("design_iteration") or 0)
        if it == 0 and vm.candidates_by_id:
            it = max(
                (c.generation_iteration for c in vm.candidates_by_id.values()),
                default=0,
            )
        if self._counters_label is not None:
            self._counters_label.set_text(
                f"Итер: {it} | Кандидаты: {total} (OK: {passed}) | Откл: {rejected}"
            )
        if self._overrides_badge is not None:
            sources = self.runtime.state.get("param_sources", {})
            non_default = sum(1 for src in sources.values() if src != "default")
            self._overrides_badge.set_text(str(non_default))
            self._overrides_badge.set_visibility(non_default > 0)


class AgentRail:
    """Левая панель: запрос, скетчер, HITL-шлюз, скаффолды, SAR-правила и Trace Feed."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_scaffold_selected: Callable[[str], Any] | None = None,
        on_tool_config_changed: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_scaffold_selected = on_scaffold_selected
        self._on_tool_config_changed = on_tool_config_changed
        self.uploaded_csv_path: Path | None = None
        self._syncing_smarts: bool = False
        self._last_sketcher_smiles: str = ""

        self._query_input: ui.textarea | None = None
        self._smiles_input: ui.input | None = None
        self._smarts_input: ui.input | None = None
        self._sketcher: Any = None
        self._hitl_card: ui.card | None = None
        self._hitl_prompt_label: ui.label | None = None
        self._run_btn: ui.button | None = None
        self._cancel_btn: ui.button | None = None
        self._scaffolds_box: ui.row | None = None
        self._sar_box: ui.column | None = None
        self._trace_box: ui.column | None = None
        self._critic_box: ui.column | None = None
        self._build()

    def _build(self) -> None:
        with ui.column().classes("w-full h-full gap-3 p-3 overflow-y-auto bg-slate-950"):
            ui.label("Оркестрация & Запрос").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400"
            )

            with ui.row().classes("w-full gap-1 flex-wrap"):
                for sc_key, sc_label in (
                    ("scenario1_btk", "1: BTK"),
                    ("scenario2_gefitinib", "2: EGFR"),
                    ("scenario3_wetlab", "3: Wet-Lab"),
                ):
                    ui.button(
                        sc_label,
                        on_click=lambda k=sc_key: self._apply_preset(k),
                    ).props("dense outline size=sm color=emerald").classes("text-[11px]")

            initial_query = str(self.runtime.state.get("user_query") or "")
            self._query_input = (
                ui.textarea(
                    label="Исследовательский запрос",
                    value=initial_query,
                    placeholder="Введите исследовательский запрос или выберите пресет выше...",
                )
                .props("outlined dark autogrow rows=2")
                .classes("w-full text-xs")
            )

            with ui.row().classes("w-full items-center gap-2"):
                self._run_btn = ui.button(
                    "Запустить граф",
                    icon="play_arrow",
                    on_click=self._handle_start_pipeline,
                ).props("dense color=emerald").classes("flex-1 text-xs")
                self._cancel_btn = ui.button(
                    "Стоп",
                    icon="stop",
                    on_click=self._handle_cancel,
                ).props("dense flat color=rose").classes("text-xs")

            # HITL-панель паузы (появляется при остановке графа)
            self._hitl_card = ui.card().classes(
                "w-full bg-amber-950/50 border border-amber-500/60 p-3 gap-2"
            )
            with self._hitl_card:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("pause_circle", color="amber-400", size="sm")
                    ui.label("Human-in-the-Loop пауза").classes(
                        "text-xs font-semibold text-amber-200"
                    )
                self._hitl_prompt_label = ui.label("").classes(
                    "text-xs text-amber-100/90"
                )
                with ui.row().classes("w-full gap-2"):
                    ui.button(
                        "Утвердить",
                        icon="check",
                        on_click=self._handle_hitl_approve,
                    ).props("dense color=emerald").classes("flex-1 text-xs")
                    ui.button(
                        "Перезапустить ветку",
                        icon="refresh",
                        on_click=self._handle_hitl_restart,
                    ).props("dense color=amber").classes("flex-1 text-xs")
            self._hitl_card.set_visibility(False)

            # Скетчер + SMILES / SMARTS
            with (
                ui.expansion("Молекулярный скетчер & Ограничения", icon="draw")
                .props("dense dark header-class='text-xs text-slate-200 bg-slate-900 rounded'")
                .classes("w-full border border-slate-800 rounded"),
                ui.column().classes("w-full gap-2 p-2"),
            ):
                    constraints_obj = self.runtime.state.get("query_constraints")
                    initial_lead_smiles = (
                        str(constraints_obj.lead_smiles)
                        if constraints_obj is not None and constraints_obj.lead_smiles
                        else ""
                    )
                    self._smiles_input = (
                        ui.input(
                            label="Lead / Scaffold SMILES",
                            value=initial_lead_smiles,
                        )
                        .props("dense outlined dark clearable")
                        .classes("w-full text-xs font-mono")
                    )
                    self._smarts_input = (
                        ui.input(
                            label="SMARTS фиксации кора (опц.)",
                            value=str(
                                resolve_tool_config(self.runtime.state).protected_smarts
                                or ""
                            ),
                            on_change=self._handle_smarts_input_change,
                        )
                        .props("dense outlined dark clearable")
                        .classes("w-full text-xs font-mono")
                    )
                    self._sketcher = create_sketcher(
                        backend=settings.sketcher_backend,
                        initial_smiles=self._smiles_input.value or "",
                        height="220px",
                        on_smiles_change=self._handle_sketcher_change,
                    )
                    ui.button(
                        "Синхронизировать SMILES → Скетчер",
                        on_click=self._sync_smiles_to_sketcher,
                    ).props("dense flat size=sm color=sky").classes("text-xs")

            # Скаффолды Bemis-Murcko
            ui.label("Скаффолды (Bemis-Murcko)").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400 mt-1"
            )
            self._scaffolds_box = ui.row().classes("w-full gap-1 flex-wrap")

            # Правила SAR и Critic
            ui.label("SAR-правила и Critic").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400 mt-1"
            )
            self._sar_box = ui.column().classes("w-full gap-1")
            self._critic_box = ui.column().classes("w-full gap-1")

            # Explainable Trace Feed
            ui.label("Лента рассуждений агентов").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400 mt-1"
            )
            self._trace_box = ui.column().classes("w-full gap-1.5")

        self.refresh()

    def _handle_sketcher_change(self, smiles: str) -> None:
        self._last_sketcher_smiles = smiles
        if self._smiles_input is not None and smiles:
            self._smiles_input.set_value(smiles)

    def get_sketcher_smiles(self) -> str:
        """Возвращает текущий SMILES из скетчера или поля Lead / Scaffold SMILES."""
        if self._last_sketcher_smiles:
            return self._last_sketcher_smiles
        if self._smiles_input is not None:
            return str(self._smiles_input.value or "").strip()
        return ""

    def set_sketcher_smiles(self, smiles: str) -> None:
        """Устанавливает SMILES в 2D-скетчер."""
        self._last_sketcher_smiles = smiles
        if self._sketcher is not None:
            self._sketcher.set_smiles(smiles)

    def set_protected_smarts(self, smarts: str) -> None:
        """Синхронизирует поле `_smarts_input` со значением из `ToolSettingsDrawer`."""
        if self._smarts_input is None:
            return
        self._syncing_smarts = True
        try:
            self._smarts_input.set_value(smarts)
        finally:
            self._syncing_smarts = False

    def _handle_smarts_input_change(self, e: Any) -> None:
        if self._syncing_smarts:
            return
        raw = str(e.value or "").strip()
        if ">>" in raw:
            return
        if raw:
            from rdkit import Chem

            if Chem.MolFromSmarts(raw) is None:
                return
        changed = agent_bridge.apply_ui_tool_overrides(
            self.runtime, {"protected_smarts": raw or None}
        )
        if changed and self._on_tool_config_changed is not None:
            self._on_tool_config_changed()

    def _sync_smiles_to_sketcher(self) -> None:
        if self._sketcher is not None and self._smiles_input is not None:
            smi = str(self._smiles_input.value or "")
            self._last_sketcher_smiles = smi
            self._sketcher.set_smiles(smi)

    def _apply_preset(self, scenario_key: str) -> None:
        preset = DEMO_SCENARIOS.get(scenario_key)
        if not preset:
            return
        if self._query_input is not None:
            self._query_input.set_value(preset["query"])
        lead_smi = preset.get("lead_smiles", "")
        if self._smiles_input is not None:
            self._smiles_input.set_value(lead_smi)
        if self._sketcher is not None and lead_smi:
            self._sketcher.set_smiles(lead_smi)
        ui.notify(f"Пресет загружен: {preset['title']}. Нажмите «Запустить граф».", type="info")

    async def _handle_start_pipeline(self) -> None:
        query = str(self._query_input.value or "").strip() if self._query_input else ""
        if not query:
            ui.notify("Введите исследовательский запрос", type="warning")
            return
        lead_smi = (
            str(self._smiles_input.value or "").strip() if self._smiles_input else ""
        )
        constraints: QueryConstraints | None = None
        if lead_smi:
            scenario_type = (
                "custom_csv_analysis"
                if self.uploaded_csv_path is not None
                else "lead_optimization"
            )
            constraints = QueryConstraints(
                scenario_type=scenario_type,
                lead_smiles=lead_smi,
                freeze_scaffold=True,
            )
        smarts_val = (
            str(self._smarts_input.value or "").strip() if self._smarts_input else ""
        )
        tool_cfg: ToolConfig | None = None
        param_srcs: dict[str, Any] | None = None
        if smarts_val and ">>" not in smarts_val:
            tool_cfg = ToolConfig(protected_smarts=smarts_val)
            param_srcs = {"protected_smarts": "ui_manual"}

        await agent_bridge.start_pipeline(
            self.runtime.session_id,
            query,
            uploaded_csv_path=str(self.uploaded_csv_path) if self.uploaded_csv_path else None,
            hitl_mode=self.runtime.hitl_mode,
            query_constraints=constraints,
            tool_config=tool_cfg,
            param_sources=param_srcs,
        )

    async def _handle_cancel(self) -> None:
        await agent_bridge.cancel_current_run(self.runtime.session_id)
        ui.notify("Выполнение графа остановлено", type="info")

    async def _handle_hitl_approve(self) -> None:
        await agent_bridge.submit_hitl_decision(
            self.runtime.session_id,
            HITLDecision(action="approve"),
        )

    async def _handle_hitl_restart(self) -> None:
        lead_smi = (
            str(self._smiles_input.value or "").strip() if self._smiles_input else ""
        )
        smarts = (
            str(self._smarts_input.value or "").strip() if self._smarts_input else ""
        )
        await agent_bridge.restart_branch(
            self.runtime.session_id,
            target_stage="design",
            modified_lead_smiles=lead_smi or None,
            modified_smarts=smarts or None,
        )

    def _select_scaffold_chip(self, scaffold_smiles: str) -> None:
        if self._smiles_input is not None:
            self._smiles_input.set_value(scaffold_smiles)
        if self._sketcher is not None:
            self._sketcher.set_smiles(scaffold_smiles)
        if self._on_scaffold_selected is not None:
            self._on_scaffold_selected(scaffold_smiles)
        ui.notify("Скаффолд загружен в скетчер", type="info")

    def refresh(self) -> None:
        """Синхронизирует содержимое левой панели с состоянием `SessionRuntime`."""
        if self._hitl_card is not None:
            self._hitl_card.set_visibility(self.runtime.status == "paused_hitl")
        if self._hitl_prompt_label is not None:
            self._hitl_prompt_label.set_text(self.runtime.hitl_pause_reason or "")

        is_running = self.runtime.status == "running"
        if self._run_btn is not None:
            self._run_btn.set_enabled(not is_running)
        if self._cancel_btn is not None:
            self._cancel_btn.set_enabled(is_running)

        tc = resolve_tool_config(self.runtime.state)
        if self._smarts_input is not None:
            cur_input_val = str(self._smarts_input.value or "").strip()
            if ">>" not in cur_input_val:
                target_smarts = str(tc.protected_smarts or "")
                if cur_input_val != target_smarts:
                    self.set_protected_smarts(target_smarts)

        if self._scaffolds_box is not None:
            self._scaffolds_box.clear()
            summary = self.runtime.state.get("curation_summary")
            scaffolds = getattr(summary, "top_scaffolds", []) if summary else []
            with self._scaffolds_box:
                if not scaffolds:
                    ui.label("Нет выделенных скаффолдов").classes(
                        "text-xs text-slate-500"
                    )
                for idx, scaf in enumerate(scaffolds[:6]):
                    smi = str(scaf.get("scaffold_smiles", ""))
                    cnt = scaf.get("count", 0)
                    label = f"S{idx + 1} (n={cnt})"
                    ui.button(
                        label,
                        on_click=lambda s=smi: self._select_scaffold_chip(s),
                    ).props("dense outline size=sm color=sky").classes(
                        "text-xs font-mono"
                    ).tooltip(smi)

        if self._sar_box is not None:
            self._sar_box.clear()
            rules = self.runtime.state.get("extracted_sar_rules", [])
            with self._sar_box:
                for rule in rules[:4]:
                    delta = getattr(rule, "mean_delta_pic50", 0.0)
                    color = "text-emerald-400" if delta >= 0 else "text-rose-400"
                    with ui.card().classes("w-full bg-slate-900 border border-slate-800 p-2"):
                        ui.label(getattr(rule, "smirks", "")).classes(
                            "text-[11px] font-mono text-slate-300 truncate"
                        )
                        ui.label(f"ΔpIC50: {delta:+.2f}").classes(
                            f"text-xs font-semibold {color}"
                        )

        if self._critic_box is not None:
            self._critic_box.clear()
            feedback = str(self.runtime.state.get("critic_feedback") or "").strip()
            if feedback:
                with (
                    self._critic_box,
                    ui.card().classes("w-full bg-rose-950/40 border border-rose-700/50 p-2"),
                ):
                    ui.label("Critic Feedback").classes("text-xs font-semibold text-rose-300")
                    ui.label(feedback).classes("text-xs text-slate-200")

        if self._trace_box is not None:
            self._trace_box.clear()
            traces: list[TraceEvent] = list(self.runtime.state.get("trace_log", []))
            node_telemetry = self.runtime.state.get("node_telemetry", {})
            param_sources = self.runtime.state.get("param_sources", {})
            with self._trace_box:
                if not traces:
                    ui.label("Ожидание запуска пайплайна...").classes(
                        "text-xs text-slate-500"
                    )
                for ev in reversed(traces[-12:]):
                    dot_color = "amber" if ev.fallback_triggered else "emerald"
                    stage_key = _AGENT_TO_STAGE.get(ev.agent_name, ev.agent_name.lower())
                    telemetry = (
                        node_telemetry.get(stage_key)
                        if isinstance(node_telemetry, dict)
                        else None
                    )
                    param_chips, stat_chips = format_node_telemetry_chips(
                        telemetry, param_sources
                    )
                    with ui.card().classes(
                        "w-full bg-slate-900/90 border border-slate-800 p-2 gap-1"
                    ):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.badge(f"{ev.agent_name} · {ev.phase}", color=dot_color).props("dense")
                            ui.label(f"{ev.duration_ms:.0f} ms").classes(
                                "text-[11px] font-mono text-slate-400"
                            )
                        ui.label(ev.output_summary or ev.action_taken).classes(
                            "text-xs text-slate-300"
                        )
                        if stat_chips:
                            with ui.row().classes("w-full gap-1 flex-wrap"):
                                for sk, sv in stat_chips[:6]:
                                    ui.badge(f"{sk}={sv}", color="emerald").props(
                                        "dense outline"
                                    ).classes("text-[10px] font-mono")
                        if param_chips:
                            with ui.row().classes("w-full gap-1 flex-wrap"):
                                for chip in param_chips[:10]:
                                    is_override = chip["source"] != "default"
                                    props_s = "dense" if is_override else "dense outline"
                                    ui.badge(
                                        chip["label"],
                                        color=chip["color"],
                                    ).props(props_s).classes(
                                        "text-[10px] font-mono"
                                    ).tooltip(f"Источник: {chip['source']}")
