"""Верхняя статусная панель и левый рельс оркестрации агентов (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

import asyncio
import html
import json
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import (
    ParamSourceLiteral,
    TraceEvent,
)
from pishovichok.ui.bridge import agent_bridge
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import SessionRuntime, session_registry
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
        "title": "Сценарий 3: Локальная серия (.csv)",
        "scenario_type": "custom_csv_analysis",
        "query": "Проанализируй лабораторную серию из CSV и предложи безопасные биоизостеры",
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

_AGENT_TO_RU: dict[str, str] = {
    "ResearchAgent": "Курация",
    "SARAgent": "Анализ SAR",
    "DesignAgent": "Генерация",
    "EvaluationAgent": "Оценка ADMET",
    "CriticAgent": "Верификация",
}

_PIPELINE_STAGES: list[tuple[str, str]] = [
    ("research", "Курация"),
    ("sar", "Анализ SAR"),
    ("design", "Генерация"),
    ("evaluation", "Оценка ADMET"),
    ("critic", "Верификация"),
]

_COMMON_FRAGMENT_NAMES: dict[str, str] = {
    "C": "Me",
    "CC": "Et",
    "OC": "OMe",
    "OCC": "OEt",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "C#N": "CN",
    "C(F)(F)F": "CF₃",
    "C(=O)NC": "CONHMe",
    "C(=O)N": "CONH₂",
    "OCCO": "OCH₂CH₂OH",
    "S(=O)(=O)C": "SO₂Me",
    "[H]": "H",
}


def _format_readable_fragment(frag: str) -> str:
    """Преобразует фрагмент SMILES с точкой крепления `[*:1]` в читаемое обозначение заместителя."""
    cleaned = frag.replace("[*:1]", "").replace("*", "").strip()
    if not cleaned:
        return "H"
    return _COMMON_FRAGMENT_NAMES.get(cleaned, cleaned)


def _compact_telemetry_val(val: Any) -> str:
    """Компактно форматирует значение параметра или счётчика для чипов телеметрии."""
    if isinstance(val, list):
        if len(val) > 3:
            return f"[{', '.join(str(x) for x in val[:2])}, +{len(val) - 2}]"
        return f"[{', '.join(str(x) for x in val)}]"
    if isinstance(val, dict):
        return f"{{{len(val)}}}"
    if isinstance(val, float):
        return f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
    return str(val)


def format_node_telemetry_chips(
    telemetry: dict[str, Any] | None,
    param_sources: dict[str, ParamSourceLiteral] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    """Преобразует `node_telemetry[node]` в структурированные чипы параметров и счётчиков."""
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
            _, color = PARAM_SOURCE_BADGE_META.get(src, ("", "slate"))
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
    """Верхняя панель: степпер этапов с бейджами возвратов, переключатель ролей, сценарии и экспорт протокола."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_session_switched: Callable[[str], Any] | None = None,
        on_csv_uploaded: Callable[[Path], Any] | None = None,
        on_toggle_settings: Callable[[], Any] | None = None,
        on_scenario_selected: Callable[[str], Any] | None = None,
        on_depth_mode_changed: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_session_switched = on_session_switched
        self._on_csv_uploaded = on_csv_uploaded
        self._on_toggle_settings = on_toggle_settings
        self._on_scenario_selected = on_scenario_selected
        self._on_depth_mode_changed = on_depth_mode_changed
        self._stepper_container: ui.row | None = None
        self._settings_btn: ui.button | None = None
        self._overrides_badge: ui.badge | None = None
        self._scenario_select: ui.select | None = None
        self._depth_toggle: ui.toggle | None = None
        self._build()

    def _build(self) -> None:
        with ui.header(elevated=False).classes(
            "bg-white border-t-[3px] border-t-[#8B495C] border-b border-[#E2DCE0] text-[#0F172A] px-3 py-1 items-center justify-between gap-2 h-[52px]"
        ):
            with ui.row().classes("items-center gap-2 no-wrap min-w-0"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.html(
                        '<div class="w-8 h-8 rounded-full bg-[#FBF5F7] border border-[#E5CFD6] overflow-hidden flex items-center justify-center shrink-0">'
                        '<img src="/static/mascot_head.png" alt="Пишовичок" class="w-7 h-7 object-contain" />'
                        "</div>",
                        sanitize=False,
                    )
                    ui.label("ПИШОВИЧОК · SAR-дизайн").classes(
                        "text-sm font-semibold tracking-tight text-[#5C2D3B] whitespace-nowrap"
                    )
                self._stepper_container = ui.row().classes("items-center ml-2 no-wrap")

            with ui.row().classes("items-center gap-1.5 no-wrap"):
                self._depth_toggle = (
                    ui.toggle(
                        {"lab": "Синтетик", "expert": "Хемоинформатик"},
                        value=self.runtime.depth_mode,
                        on_change=self._handle_depth_mode_change,
                    )
                    .props("dense no-caps toggle-color=primary color=grey-2 text-color=grey-9 size=sm")
                    .classes("border border-[#D5C8CD] rounded compact-toggle")
                )
                scenario_opts = {k: v["title"] for k, v in DEMO_SCENARIOS.items()}
                initial_sc = (
                    self.runtime.session_id
                    if self.runtime.session_id in scenario_opts
                    else (self.runtime.active_scenario_key or next(iter(scenario_opts)))
                )
                self._scenario_select = (
                    ui.select(
                        options=scenario_opts,
                        value=initial_sc,
                        on_change=self._handle_scenario_dropdown_change,
                    )
                    .props("dense outlined options-dense")
                    .classes("w-44 text-xs bg-white")
                )

                open_btn = (
                    ui.button(on_click=self._handle_open_scenario)
                    .props("dense outline color=grey-8 size=sm")
                    .classes("p-1.5 min-w-0 bg-white border-[#CBD5E1]")
                    .tooltip("Открыть выбранный сценарий")
                )
                open_btn._props["title"] = "Открыть сценарий"
                with open_btn:
                    ui.html(svg_icon("folder", 14), sanitize=False)

                clear_btn = (
                    ui.button(on_click=self._handle_clear_session)
                    .props("dense outline color=grey-8 size=sm")
                    .classes("p-1.5 min-w-0 bg-white border-[#CBD5E1]")
                    .tooltip("Очистить рабочую область и начать новую сессию")
                )
                clear_btn._props["title"] = "Очистить сессию"
                with clear_btn:
                    ui.html(svg_icon("clear", 14), sanitize=False)

                with ui.button(on_click=self._export_trace).props(
                    "dense outline no-caps size=sm color=grey-8"
                ).classes("text-[11px] bg-white border-[#CBD5E1] px-2 gap-1").tooltip(
                    "Выгрузить журнал верификации и расчёта (.json)"
                ):
                    ui.html(svg_icon("download", 13), sanitize=False)
                    ui.label("Протокол .json").classes("text-[11px] text-[#1E293B]")

                with ui.button(on_click=self._export_report).props(
                    "dense outline no-caps size=sm color=grey-8"
                ).classes("text-[11px] bg-white border-[#CBD5E1] px-2 gap-1").tooltip(
                    "Выгрузить научный отчёт (.md)"
                ):
                    ui.html(svg_icon("download", 13), sanitize=False)
                    ui.label("Отчёт .md").classes("text-[11px] text-[#1E293B]")

                self._settings_btn = (
                    ui.button(on_click=self._handle_toggle_settings)
                    .props("dense outline no-caps color=grey-8 size=sm")
                    .classes("text-xs font-medium px-2.5 gap-1.5 bg-[#FBF5F7] border-[#D9BCC5] text-[#73394B]")
                    .tooltip("Открыть настройки вычислительного ядра и оркестрации")
                )
                with self._settings_btn:
                    ui.html(svg_icon("sliders", 14), sanitize=False)
                    ui.label("Параметры").classes("text-xs font-medium")
                    self._overrides_badge = ui.badge("0", color="amber-8").props(
                        "floating"
                    )
                    self._overrides_badge.set_visibility(False)

        self.refresh()

    async def _export_trace(self) -> None:
        traces = [ev.model_dump(mode="json") for ev in self.runtime.state.get("trace_log", [])]
        payload = json.dumps(traces, ensure_ascii=False, indent=2).encode("utf-8")
        ui.download(payload, filename=f"{self.runtime.session_id}_protocol.json")

    async def _export_report(self) -> None:
        report_md = str(
            self.runtime.state.get("final_report_markdown") or "# Отчёт ещё не сформирован"
        )
        ui.download(
            report_md.encode("utf-8"),
            filename=f"{self.runtime.session_id}_report.md",
        )

    def _handle_depth_mode_change(self, e: Any) -> None:
        val = str(e.value)
        if val in ("lab", "expert"):
            self.runtime.depth_mode = val  # type: ignore[assignment]
            if self._on_depth_mode_changed is not None:
                self._on_depth_mode_changed(val)

    def _handle_scenario_dropdown_change(self, e: Any) -> None:
        val = str(e.value or "")
        self.runtime.active_scenario_key = val
        if self._on_scenario_selected is not None:
            self._on_scenario_selected(val)

    def _handle_toggle_settings(self) -> None:
        if self._on_toggle_settings is not None:
            self._on_toggle_settings()

    async def _handle_open_scenario(self) -> None:
        if self._scenario_select is None or not self._scenario_select.value:
            return
        sid = str(self._scenario_select.value)
        restored = session_registry.get(sid) or await session_registry.restore_from_workspace(sid)
        if restored is None:
            ui.notify(f"Сохранённый сценарий '{sid}' не найден", type="warning")
            return
        restored.active_scenario_key = sid
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

    def _build_stepper_nodes(self) -> None:
        """Рендерит линейный степпер этапов с бейджами возврата внутри узлов."""
        if self._stepper_container is None:
            return
        traces: list[TraceEvent] = list(self.runtime.state.get("trace_log", []))
        completed = {
            _AGENT_TO_STAGE.get(ev.agent_name, ev.agent_name.lower())
            for ev in traces
        }
        active = self.runtime.current_node
        is_paused = self.runtime.status == "paused_hitl"
        is_running = self.runtime.status == "running"

        loop1_count = max(
            int(self.runtime.state.get("data_refilter_iteration") or 0),
            sum(1 for ev in traces if ev.verdict_or_routing == "refilter_data"),
        )
        loop2_count = sum(1 for ev in traces if ev.verdict_or_routing == "redesign_molecules")
        if loop2_count == 0 and int(self.runtime.state.get("design_iteration") or 0) > 1:
            loop2_count = int(self.runtime.state.get("design_iteration") or 0) - 1

        node_telemetry = self.runtime.state.get("node_telemetry", {})
        last_events: dict[str, TraceEvent] = {}
        for ev in traces:
            sk = _AGENT_TO_STAGE.get(ev.agent_name, ev.agent_name.lower())
            last_events[sk] = ev

        self._stepper_container.clear()
        with self._stepper_container, ui.row().classes("items-center gap-1 no-wrap"):
            for idx, (stage_key, stage_label) in enumerate(_PIPELINE_STAGES):
                if idx > 0:
                    ui.label("→").classes("text-xs text-[#94A3B8] select-none")
                ev = last_events.get(stage_key)
                tip_parts = [f"Этап: {stage_label}"]
                if ev is not None:
                    tip_parts.append(f"{ev.output_summary} ({ev.duration_ms:.0f} мс)")
                elif isinstance(node_telemetry, dict) and stage_key in node_telemetry:
                    tip_parts.append("Выполнен")
                else:
                    tip_parts.append("Ожидание")

                if is_paused and stage_key == active:
                    cls = "bg-[#FFFBEB] border border-[#D97706] text-[#78350F] font-semibold"
                elif stage_key == active and is_running:
                    cls = "bg-[#FBF5F7] border border-[#8B495C] text-[#5C2D3B] font-semibold animate-pulse"
                elif stage_key in completed:
                    cls = "bg-[#ECFDF5] border border-[#059669] text-[#065F46] font-medium"
                else:
                    cls = "bg-[#FAF8F9] border border-[#D5C8CD] text-[#594D52]"

                loop_badge_count = 0
                if stage_key == "research" and loop1_count > 0:
                    loop_badge_count = loop1_count
                elif stage_key == "design" and loop2_count > 0:
                    loop_badge_count = loop2_count

                def _on_node_click(sk: str = stage_key, sl: str = stage_label) -> None:
                    ev_local = last_events.get(sk)
                    msg = (
                        f"{sl}: {ev_local.output_summary}"
                        if ev_local is not None
                        else f"{sl}: этап ещё не выполнялся"
                    )
                    ui.notify(msg, type="info")

                with (
                    ui.button(on_click=_on_node_click)
                    .props("dense flat no-caps size=xs")
                    .classes(f"text-[11px] px-2 py-0.5 rounded-full gap-1 {cls}")
                    .tooltip(" — ".join(tip_parts))
                ):
                    ui.label(stage_label).classes("text-[11px] leading-none")
                    if loop_badge_count > 0:
                        ui.label(f"×{loop_badge_count}").classes(
                            "text-[10px] font-semibold px-1 py-0 rounded bg-[#FFFBEB] border border-[#F59E0B] text-[#78350F] leading-tight"
                        )

    def refresh(self) -> None:
        """Обновляет интерактивный степпер этапов и бейджи по текущему состоянию сессии."""
        self._build_stepper_nodes()

        if self._overrides_badge is not None:
            sources = self.runtime.state.get("param_sources", {})
            non_default = sum(1 for src in sources.values() if src != "default")
            self._overrides_badge.set_text(str(non_default))
            self._overrides_badge.set_visibility(non_default > 0)
        if self._depth_toggle is not None and self._depth_toggle.value != self.runtime.depth_mode:
            self._depth_toggle.set_value(self.runtime.depth_mode)


class AgentRail:
    """Левая панель 300px: скаффолды Bemis–Murcko и вкладки «Правила SAR» / «Журнал верификации»."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_scaffold_selected: Callable[[str], Any] | None = None,
        on_toggle_collapse: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_scaffold_selected = on_scaffold_selected
        self._on_toggle_collapse = on_toggle_collapse
        self._is_collapsed: bool = False

        self._content_col: ui.column | None = None
        self._collapse_btn: ui.button | None = None
        self._collapse_icon_box: ui.html | None = None
        self._scaffolds_section: ui.column | None = None
        self._scaffolds_box: ui.row | None = None
        self._sar_header_label: ui.label | None = None
        self._sar_box: ui.column | None = None
        self._timeline_box: ui.column | None = None
        self._state_spoiler_box: ui.column | None = None
        self._build()

    def set_collapsed(self, collapsed: bool) -> None:
        """Переключает свёрнутое состояние левой панели."""
        self._is_collapsed = collapsed
        if self._content_col is not None:
            self._content_col.set_visibility(not collapsed)
        if self._collapse_icon_box is not None:
            self._collapse_icon_box.set_content(
                svg_icon("chevron_right" if collapsed else "chevron_left", 14)
            )
        if self._collapse_btn is not None:
            self._collapse_btn._props["title"] = (
                "Развернуть панель" if collapsed else "Свернуть панель"
            )

    def _build(self) -> None:
        with ui.row().classes(
            "w-full h-full no-wrap gap-0 bg-[#FAF8F9] text-[#1F181B] border-r border-[#E2D9DD]"
        ):
            self._content_col = ui.column().classes(
                "flex-1 min-w-0 h-full flex flex-col overflow-hidden gap-2 p-2.5"
            )
            with self._content_col:
                # Базовые скаффолды Bemis–Murcko (скрываются если пусто)
                self._scaffolds_section = ui.column().classes("w-full shrink-0 gap-1")
                with self._scaffolds_section:
                    ui.label("Базовые скаффолды (Bemis–Murcko)").classes(
                        "text-xs font-semibold text-[#5C2D3B]"
                    )
                    self._scaffolds_box = ui.row().classes("w-full gap-1 flex-wrap")

                # Вкладки аналитики на всю доступную высоту левой панели
                with ui.column().classes(
                    "w-full flex-1 min-h-0 flex flex-col bg-white border border-[#E2D9DD] rounded-lg overflow-hidden"
                ):
                    with (
                        ui.tabs()
                        .props("dense align=justify no-caps mobile-breakpoint=0 outside-arrows=false active-color=primary indicator-color=primary")
                        .classes("w-full bg-[#F4EFF1] border-b border-[#E2D9DD] text-[#594D52] shrink-0 compact-tabs") as rail_tabs
                    ):
                        t_sar = ui.tab("sar", label="Правила SAR")
                        t_log = ui.tab("log", label="Журнал верификации")

                    with ui.tab_panels(rail_tabs, value=t_sar).props("animated=false").classes(
                        "w-full flex-1 min-h-0 bg-white"
                    ):
                        with ui.tab_panel(t_sar).classes("p-2 h-full overflow-y-auto wetlab-scroll"):
                            self._sar_header_label = ui.label("Правила SAR (MMP)").classes(
                                "text-[11px] font-semibold text-[#475569] mb-1.5"
                            )
                            self._sar_box = ui.column().classes("w-full gap-1.5")
                        with ui.tab_panel(t_log).classes("p-2 h-full overflow-y-auto wetlab-scroll"):
                            self._timeline_box = ui.column().classes("w-full gap-2")
                            self._state_spoiler_box = ui.column().classes("w-full mt-2")

            # Кнопка сворачивания на правой границе левой панели
            with ui.column().classes(
                "w-[22px] h-full shrink-0 items-center justify-center bg-[#F1F5F9] hover:bg-[#E2E8F0] border-l border-[#E2E8F0] transition-colors"
            ):
                self._collapse_btn = (
                    ui.button(on_click=self._handle_collapse_click)
                    .props("dense flat size=xs color=grey-8")
                    .classes("w-full h-12 min-w-0 p-0 text-[#475569]")
                    .tooltip("Свернуть панель")
                )
                self._collapse_btn._props["title"] = "Свернуть панель"
                with self._collapse_btn:
                    self._collapse_icon_box = ui.html(
                        svg_icon("chevron_left", 14), sanitize=False
                    )

        self.refresh()

    def _handle_collapse_click(self) -> None:
        if self._on_toggle_collapse is not None:
            self._on_toggle_collapse()

    def _select_scaffold_chip(self, scaffold_smiles: str) -> None:
        if self._on_scaffold_selected is not None:
            self._on_scaffold_selected(scaffold_smiles)

    def _render_sar_rules(self) -> None:
        if self._sar_box is None:
            return
        self._sar_box.clear()
        raw_rules = list(self.runtime.state.get("extracted_sar_rules", []))
        rules = sorted(
            raw_rules,
            key=lambda r: float(getattr(r, "mean_delta_pic50", 0.0)),
            reverse=True,
        )

        # Определяем общий код BAO для заголовка секции
        bao_counts: Counter[str] = Counter()
        for r in rules:
            for b in getattr(r, "participating_bao_formats", []) or []:
                if b and b != "USER_CONSTRAINT":
                    bao_counts[str(b)] += 1
        common_bao = bao_counts.most_common(1)[0][0] if bao_counts else ""
        if self._sar_header_label is not None:
            self._sar_header_label.set_text(
                f"Правила SAR (MMP) · {common_bao}" if common_bao else "Правила SAR (MMP)"
            )

        sid = self.runtime.session_id
        is_expert = self.runtime.depth_mode == "expert"
        with self._sar_box:
            if not rules:
                ui.label("Правила SAR ещё не извлечены").classes(
                    "text-xs text-[#475569]"
                )
                return
            for rule in rules[:8]:
                rule_id = str(getattr(rule, "rule_id", "R?"))
                delta = float(getattr(rule, "mean_delta_pic50", 0.0))
                pairs = int(getattr(rule, "support_pairs_count", 0))
                homog = bool(getattr(rule, "assay_homogeneous", False))
                from_readable = _format_readable_fragment(str(getattr(rule, "from_fragment", "")))
                to_readable = _format_readable_fragment(str(getattr(rule, "to_fragment", "")))
                smirks_raw = str(getattr(rule, "smirks", ""))

                delta_cls = (
                    "bg-[#ECFDF5] text-[#065F46] border-[#A7F3D0]"
                    if delta > 0
                    else "bg-[#FFF1F2] text-[#9F1239] border-[#FECDD3]"
                )

                with ui.card().classes(
                    "w-full bg-white border border-[#E2E8F0] border-l-[3px] border-l-[#2563EB] p-2 gap-1.5 shadow-none hover:bg-[#F8FAFC] transition-colors"
                ) as card:
                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.label(f"{rule_id}: {from_readable} → {to_readable}").classes(
                            "text-xs font-semibold text-[#0F172A]"
                        )
                        if not homog:
                            ui.label("Неоднородно").classes(
                                "text-[10px] font-medium px-1.5 py-0 rounded border bg-[#FFFBEB] text-[#78350F] border-[#FDE68A]"
                            )
                    if is_expert and smirks_raw:
                        ui.label(smirks_raw).classes(
                            "text-[10px] mono-code px-1.5 py-0.5 truncate w-full"
                        )
                    with ui.row().classes("w-full items-center gap-1 flex-wrap"):
                        ui.label(f"ΔpIC50: {delta:+.2f}").classes(
                            f"text-[10px] font-semibold px-1.5 py-0.5 rounded border {delta_cls}"
                        )
                        ui.label(f"Пар (N): {pairs}").classes(
                            "text-[10px] font-medium px-1.5 py-0.5 rounded border bg-[#EFF6FF] text-[#1E3A8A] border-[#BFDBFE]"
                        )

                    # Превью 2D-трансформации по наведению
                    with card, ui.tooltip().classes("bg-white border border-[#CBD5E1] shadow-md p-1.5"):
                        ui.html(
                            f'<img src="/api/sessions/{html.escape(sid)}/rule/{html.escape(rule_id)}/svg" '
                            f'style="width:300px;height:105px;display:block;background:#FFFFFF;" />',
                            sanitize=False,
                        )

    def _render_unified_timeline(self) -> None:
        if self._timeline_box is None:
            return
        self._timeline_box.clear()
        traces: list[TraceEvent] = list(self.runtime.state.get("trace_log", []))
        node_telemetry = self.runtime.state.get("node_telemetry", {})
        param_sources = self.runtime.state.get("param_sources", {})
        is_expert = self.runtime.depth_mode == "expert"

        with self._timeline_box:
            if not traces:
                ui.label("Ожидание запуска расчёта...").classes(
                    "text-xs text-[#475569]"
                )
                return

            iter_groups: list[tuple[str, list[TraceEvent]]] = []
            cur_iter_idx = 0
            cur_label = "Итерация 1 · Основной проход"
            cur_list: list[TraceEvent] = []
            for ev in traces:
                if ev.verdict_or_routing in ("refilter_data", "redesign_molecules"):
                    cur_list.append(ev)
                    iter_groups.append((cur_label, cur_list))
                    cur_iter_idx += 1
                    reason_lbl = (
                        "Перефильтрация BAO"
                        if ev.verdict_or_routing == "refilter_data"
                        else "Повторная генерация"
                    )
                    cur_label = f"Итерация {cur_iter_idx + 1} · {reason_lbl}"
                    cur_list = []
                else:
                    cur_list.append(ev)
            if cur_list:
                iter_groups.append((cur_label, cur_list))

            for group_title, group_events in reversed(iter_groups):
                ui.label(group_title).classes(
                    "text-[11px] font-semibold text-[#475569] bg-[#F1F5F9] px-2 py-1 rounded border border-[#E2E8F0]"
                )
                with ui.column().classes(
                    "w-full gap-1.5 pl-2.5 ml-1 border-l-2 border-[#CBD5E1]"
                ):
                    for ev in reversed(group_events):
                        is_critic = ev.agent_name == "CriticAgent"
                        is_loop_back = is_critic and ev.verdict_or_routing in (
                            "refilter_data",
                            "redesign_molecules",
                        )
                        is_approve = is_critic and ev.verdict_or_routing == "approve"

                        if is_loop_back:
                            card_cls = "w-full bg-[#FFFBEB] border border-[#F59E0B] border-l-[3px] border-l-[#D97706] p-2 gap-1 shadow-none"
                            text_cls = "text-xs text-[#78350F]"
                            badge_cls = "bg-[#F59E0B] text-white"
                            loop_ru = (
                                "Перефильтрация BAO"
                                if ev.verdict_or_routing == "refilter_data"
                                else "Повторная генерация"
                            )
                            title_txt = f"Возврат: {loop_ru}"
                        elif is_approve:
                            card_cls = "w-full bg-[#ECFDF5] border border-[#059669] border-l-[3px] border-l-[#059669] p-2 gap-1 shadow-none"
                            text_cls = "text-xs text-[#065F46]"
                            badge_cls = "bg-[#059669] text-white"
                            title_txt = "Утверждено (5/5)"
                        else:
                            card_cls = "w-full bg-white border border-[#E2D9DD] p-2 gap-1 shadow-none"
                            text_cls = "text-xs text-[#1F181B]"
                            badge_cls = (
                                "bg-[#D97706] text-white"
                                if ev.fallback_triggered
                                else "bg-[#8B495C] text-white"
                            )
                            stage_ru = _AGENT_TO_RU.get(ev.agent_name, ev.agent_name)
                            title_txt = stage_ru

                        with ui.card().classes(card_cls):
                            with ui.row().classes("w-full items-center justify-between gap-1"):
                                ui.label(title_txt).classes(
                                    f"text-[10px] font-semibold px-1.5 py-0.5 rounded {badge_cls}"
                                )
                                ui.label(f"{ev.duration_ms:.0f} мс").classes(
                                    "text-[10px] font-mono text-[#475569]"
                                )
                            ui.label(ev.output_summary or ev.action_taken).classes(text_cls)

                            if is_expert:
                                stage_key = _AGENT_TO_STAGE.get(
                                    ev.agent_name, ev.agent_name.lower()
                                )
                                telemetry = (
                                    node_telemetry.get(stage_key)
                                    if isinstance(node_telemetry, dict)
                                    else None
                                )
                                param_chips, stat_chips = format_node_telemetry_chips(
                                    telemetry, param_sources
                                )
                                if stat_chips:
                                    with ui.row().classes("w-full gap-1 flex-wrap"):
                                        for sk, sv in stat_chips[:6]:
                                            ui.label(f"{sk}={sv}").classes(
                                                "text-[10px] mono-code px-1 py-0"
                                            )
                                if param_chips:
                                    with ui.row().classes("w-full gap-1 flex-wrap"):
                                        for chip in param_chips[:10]:
                                            ui.label(chip["label"]).classes(
                                                "text-[10px] mono-code px-1 py-0"
                                            ).tooltip(f"Источник: {chip['source']}")

    def _render_state_json_spoiler(self) -> None:
        if self._state_spoiler_box is None:
            return
        self._state_spoiler_box.clear()
        if self.runtime.depth_mode != "expert":
            return
        tc = resolve_tool_config(self.runtime.state)
        state_flags = {
            "strict_assay_homogeneity": tc.strict_assay_homogeneity,
            "strict_property_thresholds": tc.strict_property_thresholds,
            "assay_filter_strict": self.runtime.state.get("assay_filter_strict"),
            "active_bao_filter": self.runtime.state.get("active_bao_filter"),
            "data_refilter_iteration": self.runtime.state.get("data_refilter_iteration"),
            "design_iteration": self.runtime.state.get("design_iteration"),
            "critic_verdict": self.runtime.state.get("critic_verdict"),
            "non_default_param_sources": {
                k: v
                for k, v in self.runtime.state.get("param_sources", {}).items()
                if v != "default"
            },
        }
        with self._state_spoiler_box, (
            ui.expansion("Диагностика состояния", value=False)
            .props("dense header-class='text-[11px] text-[#475569] bg-[#F1F5F9] rounded'")
            .classes("w-full bg-white border border-[#E2E8F0] rounded-lg")
        ):
            ui.html(
                f"<pre class='mono-code p-2 text-[10px] overflow-x-auto m-1'>{html.escape(json.dumps(state_flags, ensure_ascii=False, indent=2))}</pre>",
                sanitize=False,
            )

    def refresh(self) -> None:
        """Синхронизирует содержимое левой панели с состоянием `SessionRuntime`."""
        summary = self.runtime.state.get("curation_summary")
        scaffolds = getattr(summary, "top_scaffolds", []) if summary else []
        if self._scaffolds_section is not None:
            self._scaffolds_section.set_visibility(bool(scaffolds))
        if self._scaffolds_box is not None:
            self._scaffolds_box.clear()
            if scaffolds:
                with self._scaffolds_box:
                    for idx, scaf in enumerate(scaffolds[:6]):
                        smi = str(scaf.get("scaffold_smiles", ""))
                        cnt = scaf.get("count", 0)
                        label = f"S{idx + 1} (n={cnt})"
                        ui.button(
                            label,
                            on_click=lambda s=smi: self._select_scaffold_chip(s),
                        ).props("dense outline size=sm color=primary").classes(
                            "text-xs font-mono bg-[#FBF5F7]"
                        ).tooltip(smi)

        self._render_sar_rules()
        self._render_unified_timeline()
        self._render_state_json_spoiler()
