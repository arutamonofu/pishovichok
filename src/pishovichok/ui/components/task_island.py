"""Единый нейрочат-островок постановки и управления задачей (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from nicegui import ui

from pishovichok.config import settings
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import QueryConstraints
from pishovichok.ui.bridge import agent_bridge
from pishovichok.ui.components.agent_rail import (
    DEMO_SCENARIOS,
    format_pishovichok_journal_summary,
    render_verification_journal,
)
from pishovichok.ui.components.chem_widgets import create_sketcher
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import HITLDecision, SessionRuntime, session_registry

_IslandPanel = Literal["none", "demos", "csv", "sketcher", "docking", "journal"]


class TaskCommandIsland:
    """Единый островок управления задачей внизу экрана в стиле нейрочатов.

    Объединяет в одном месте:
    - живую реплику «Пишовичка» о текущем выполняемом действии;
    - ввод промпта (Enter — запуск, Shift+Enter — перенос строки) и кнопки Запуск / Стоп;
    - быстрый выбор демо-промптов и открытие готовых предпросчитанных сценариев;
    - загрузку лабораторного CSV, 2D-редактор структур/SMARTS;
    - настройки 3D-докинга и кнопку ручного запуска 3D-докинга;
    - кнопку глобальных настроек вычислительного ядра («Параметры»);
    - панель паузы контроля исследователя (HITL).
    """

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_tool_config_changed: Callable[[], Any] | None = None,
        on_toggle_settings: Callable[[], Any] | None = None,
        on_session_switched: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_tool_config_changed = on_tool_config_changed
        self._on_toggle_settings = on_toggle_settings
        self._on_session_switched = on_session_switched

        self.uploaded_csv_path: Path | None = None
        self._syncing_smarts: bool = False
        self._syncing_docking: bool = False
        self._last_sketcher_smiles: str = ""
        self._active_panel: _IslandPanel = "none"
        self._is_floating: bool = not self.runtime.is_clean_welcome
        self._composer_expanded: bool = self.runtime.is_clean_welcome

        # Внешние контейнеры для плавающего позиционирования и сворачивания
        self._root_col: ui.column | None = None
        self._inner_col: ui.column | None = None
        self._composer_card: ui.card | None = None

        # Живая реплика Пишовичка и кнопки компактной пилюли
        self._speech_bar: ui.row | None = None
        self._speech_spinner: ui.spinner | None = None
        self._speech_dot: ui.element | None = None
        self._speech_label: ui.label | None = None
        self._journal_toggle_btn: ui.button | None = None
        self._journal_toggle_label: ui.label | None = None
        self._journal_chevron_box: ui.html | None = None
        self._compact_stop_btn: ui.button | None = None
        self._expand_btn: ui.button | None = None
        self._collapse_btn: ui.button | None = None

        # Всплывающие секции над композером
        self._demos_box: ui.column | None = None
        self._csv_box: ui.column | None = None
        self._csv_status_label: ui.label | None = None
        self._sketcher_box: ui.column | None = None
        self._docking_box: ui.column | None = None
        self._journal_box: ui.column | None = None
        self._journal_timeline_box: ui.column | None = None
        self._journal_spoiler_box: ui.column | None = None

        # Поля композера и контекстные чипы
        self._context_chips_row: ui.row | None = None
        self._structure_chip: ui.label | None = None
        self._csv_chip: ui.label | None = None
        self._query_input: ui.textarea | None = None
        self._smiles_input: ui.input | None = None
        self._smarts_input: ui.input | None = None
        self._sketcher: Any = None

        # Настройки 3D-докинга
        self._docking_mode_select: ui.select | None = None
        self._docking_pdb_input: ui.input | None = None
        self._docking_box_input: ui.input | None = None
        self._docking_panel_run_btn: ui.button | None = None

        # Кнопки тулбара
        self._demos_btn: ui.button | None = None
        self._csv_btn: ui.button | None = None
        self._sketcher_btn: ui.button | None = None
        self._docking_cfg_btn: ui.button | None = None
        self._quick_dock_btn: ui.button | None = None
        self._settings_btn: ui.button | None = None
        self._overrides_badge: ui.badge | None = None
        self._run_btn: ui.button | None = None
        self._cancel_btn: ui.button | None = None

        # HITL и приветственные стартовые демо-чипы
        self._hitl_card: ui.card | None = None
        self._hitl_prompt_label: ui.label | None = None
        self._welcome_chips_row: ui.row | None = None

        self._build()

    def _build(self) -> None:
        self._root_col = ui.column().classes(
            "w-full items-center shrink-0 px-3 pb-2.5 pt-1 gap-2"
        )
        with self._root_col:  # noqa: SIM117
            self._inner_col = ui.column().classes(
                "relative w-full max-w-[760px] gap-1.5 transition-all duration-200"
            )
            with self._inner_col:
                # 1. Компактная полоса Пишовичка (и свёрнутая плавающая пилюля после старта)
                self._speech_bar = ui.row().classes(
                    "pointer-events-auto w-full items-center gap-2 px-3 py-1.5 rounded-2xl "
                    "bg-[#FBF5F7] border border-[#E5CFD6] shadow-sm no-wrap"
                )
                with self._speech_bar:
                    ui.html(
                        '<div class="w-6 h-6 rounded-full bg-white border border-[#E5CFD6] overflow-hidden flex items-center justify-center shrink-0">'
                        '<img src="/static/mascot_head.png" alt="Пишовичок" class="w-5 h-5 object-contain" />'
                        "</div>",
                        sanitize=False,
                    )
                    self._speech_spinner = ui.spinner("dots", size="xs", color="primary")
                    self._speech_dot = ui.element("span").classes(
                        "w-2 h-2 rounded-full bg-[#D97706] shrink-0"
                    )
                    ui.label("Пишовичок:").classes(
                        "text-xs font-semibold text-[#5C2D3B] shrink-0"
                    )
                    self._speech_label = ui.label("").classes(
                        "text-xs text-[#261F22] flex-1 min-w-0 truncate"
                    )
                    self._journal_toggle_btn = (
                        ui.button(on_click=lambda: self._toggle_panel("journal"))
                        .props("dense outline no-caps size=xs color=primary")
                        .classes(
                            "text-[11px] px-2 py-0.5 rounded-lg bg-white/90 border-[#D9BCC5] text-[#5C2D3B] gap-1 shrink-0"
                        )
                        .tooltip("Развернуть или свернуть подробный журнал верификации")
                    )
                    with self._journal_toggle_btn:
                        self._journal_toggle_label = ui.label("Журнал").classes(
                            "text-[11px] font-medium"
                        )
                        self._journal_chevron_box = ui.html(
                            svg_icon("chevron_up", 12), sanitize=False
                        )

                    self._compact_stop_btn = (
                        ui.button(on_click=self._handle_cancel)
                        .props("dense flat no-caps color=negative size=xs")
                        .classes(
                            "text-[11px] px-2 py-0.5 rounded-lg gap-1 bg-[#FFF1F2] border border-[#FECDD3] shrink-0"
                        )
                        .tooltip("Остановить текущий расчёт")
                    )
                    with self._compact_stop_btn:
                        ui.html(svg_icon("stop", 11), sanitize=False)
                        ui.label("Стоп").classes("text-[11px] font-medium")
                    self._compact_stop_btn.set_visibility(False)

                    self._expand_btn = (
                        ui.button(on_click=self._expand_composer)
                        .props("dense unelevated no-caps size=xs")
                        .style("background-color: #8B495C; color: #FFFFFF;")
                        .classes("btn-run-primary text-[11px] px-2.5 py-0.5 rounded-lg gap-1 shrink-0")
                        .tooltip("Развернуть форму запроса и инструменты")
                    )
                    with self._expand_btn:
                        ui.html(svg_icon("edit", 11), sanitize=False)
                        ui.label("Запрос").classes("text-[11px] font-medium")
                    self._expand_btn.set_visibility(False)

                    self._collapse_btn = (
                        ui.button(on_click=self._collapse_composer)
                        .props("dense flat no-caps size=xs color=grey-8")
                        .classes(
                            "text-[11px] px-2 py-0.5 rounded-lg bg-white/80 hover:bg-white border border-[#E5CFD6] text-[#594D52] gap-1 shrink-0"
                        )
                        .tooltip("Свернуть чат в компактную строку поверх таблицы")
                    )
                    with self._collapse_btn:
                        ui.html(svg_icon("chevron_down", 12), sanitize=False)
                        ui.label("Свернуть").classes("text-[11px]")
                    self._collapse_btn.set_visibility(False)
                self._speech_bar.set_visibility(False)

                # Панель паузы контроля исследователя (HITL) — всегда видна при паузе, даже если чат свёрнут
                self._hitl_card = ui.card().classes(
                    "pointer-events-auto w-full bg-[#FFFBEB]/95 backdrop-blur-md border border-[#F59E0B] p-2 gap-1.5 shadow-lg rounded-xl"
                )
                with self._hitl_card:
                    with ui.row().classes("w-full items-center justify-between gap-2"):
                        ui.label("Пауза контроля исследователя").classes(
                            "text-xs font-semibold text-[#78350F]"
                        )
                        with ui.row().classes("gap-1.5"):
                            ui.button(
                                "Утвердить",
                                on_click=self._handle_hitl_approve,
                            ).props("dense unelevated no-caps size=xs color=green-7").classes(
                                "text-[11px] px-2"
                            )
                            ui.button(
                                "Перезапустить ветвь",
                                on_click=self._handle_hitl_restart,
                            ).props("dense unelevated no-caps size=xs color=amber-8").classes(
                                "text-[11px] px-2"
                            )
                    self._hitl_prompt_label = ui.label("").classes(
                        "text-[11px] text-[#78350F]"
                    )
                self._hitl_card.set_visibility(False)

                # 2. Всплывающая вверх панель дополнительных настроек / демо / CSV / скетчера
                with ui.element("div").classes(
                    "pointer-events-none absolute bottom-full left-0 right-0 mb-2 z-40 flex flex-col items-stretch"
                ):
                    # 2a. Демо-промпты и готовые сценарии
                    self._demos_box = ui.column().classes(
                        "pointer-events-auto w-full gap-1.5 p-2.5 bg-white border border-[#D9BCC5] rounded-2xl shadow-xl"
                    )
                    with self._demos_box:
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Демо-промпты и предпросчитанные сценарии").classes(
                                "text-xs font-semibold text-[#5C2D3B]"
                            )
                            with ui.button(on_click=self._close_panels).props(
                                "dense flat size=xs color=grey-7"
                            ):
                                ui.html(svg_icon("close", 14), sanitize=False)
                        for sc_key, sc_meta in DEMO_SCENARIOS.items():
                            with ui.row().classes(
                                "w-full items-center justify-between gap-2 p-2 rounded-xl "
                                "bg-[#FAF8F9] border border-[#E2D9DD] hover:border-[#D9BCC5] no-wrap"
                            ):
                                with ui.column().classes("gap-0.5 min-w-0 flex-1"):
                                    ui.label(sc_meta["title"]).classes(
                                        "text-xs font-semibold text-[#1F181B] truncate"
                                    )
                                    ui.label(sc_meta["query"]).classes(
                                        "text-[11px] text-[#594D52] truncate"
                                    )
                                with ui.row().classes("items-center gap-1 shrink-0"):
                                    ui.button(
                                        "Подставить",
                                        on_click=lambda k=sc_key: self._apply_demo_prompt(k),
                                    ).props("dense flat no-caps size=xs color=primary").classes(
                                        "text-[11px] px-2 py-0.5 rounded-lg bg-[#FBF5F7] border border-[#E5CFD6]"
                                    )
                                    ui.button(
                                        "Открыть расчёт",
                                        on_click=lambda k=sc_key: self._open_saved_scenario(k),
                                    ).props("dense outline no-caps size=xs color=grey-8").classes(
                                        "text-[11px] px-2 py-0.5 rounded-lg bg-white border-[#D5C8CD]"
                                    )
                    self._demos_box.set_visibility(False)

                    # 2b. Загрузка лабораторного CSV
                    self._csv_box = ui.column().classes(
                        "pointer-events-auto w-full gap-1.5 p-2.5 bg-white border border-[#D9BCC5] rounded-2xl shadow-xl"
                    )
                    with self._csv_box:
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Лабораторная серия (.csv)").classes(
                                "text-xs font-semibold text-[#5C2D3B]"
                            )
                            with ui.button(on_click=self._close_panels).props(
                                "dense flat size=xs color=grey-7"
                            ):
                                ui.html(svg_icon("close", 14), sanitize=False)
                        self._csv_status_label = ui.label(
                            "Выберите лабораторный CSV-файл (SMILES, pIC50/IC50)"
                        ).classes("text-[11px] text-[#594D52]")
                        ui.upload(
                            on_upload=self._handle_csv_upload,
                            auto_upload=True,
                        ).props(
                            "accept=.csv flat bordered color=primary label='Выбрать CSV'"
                        ).classes("w-full")
                        with ui.row().classes("w-full items-center justify-between gap-1"):
                            ui.button(
                                "Подключить демо-серию (.csv)",
                                on_click=self._load_sample_wetlab_csv,
                            ).props("dense flat size=sm no-caps color=primary").classes(
                                "text-[11px]"
                            )
                            ui.button(
                                "Сбросить файл",
                                on_click=self._clear_csv,
                            ).props("dense flat size=sm no-caps color=grey-7").classes(
                                "text-[11px]"
                            )
                    self._csv_box.set_visibility(False)

                    # 2c. Редактор структур и SMARTS-ограничений
                    self._sketcher_box = ui.column().classes(
                        "pointer-events-auto w-full gap-1.5 p-2.5 bg-white border border-[#D9BCC5] rounded-2xl shadow-xl"
                    )
                    with self._sketcher_box:
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Редактор структур и фиксация кора (SMARTS)").classes(
                                "text-xs font-semibold text-[#5C2D3B]"
                            )
                            with ui.button(on_click=self._close_panels).props(
                                "dense flat size=xs color=grey-7"
                            ):
                                ui.html(svg_icon("close", 14), sanitize=False)
                        constraints_obj = self.runtime.state.get("query_constraints")
                        initial_lead_smiles = (
                            str(constraints_obj.lead_smiles)
                            if constraints_obj is not None and constraints_obj.lead_smiles
                            else ""
                        )
                        with ui.grid(columns=2).classes("w-full gap-2"):
                            self._smiles_input = (
                                ui.input(
                                    label="SMILES прототипа / каркаса",
                                    value=initial_lead_smiles,
                                    on_change=lambda _e: self._sync_context_chips(),
                                )
                                .props("dense outlined clearable")
                                .classes("w-full text-xs font-mono bg-white")
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
                                .props("dense outlined clearable")
                                .classes("w-full text-xs font-mono bg-white")
                            )
                        self._sketcher = create_sketcher(
                            backend=settings.sketcher_backend,
                            initial_smiles=self._smiles_input.value or "",
                            height="190px",
                            on_smiles_change=self._handle_sketcher_change,
                        )
                        ui.button(
                            "Перенести SMILES в редактор",
                            on_click=self._sync_smiles_to_sketcher,
                        ).props("dense flat no-caps size=xs color=primary").classes(
                            "text-[11px] self-start"
                        )
                    self._sketcher_box.set_visibility(False)

                    # 2d. Настройки и запуск 3D-докинга
                    tc_dock = resolve_tool_config(self.runtime.state)
                    self._docking_box = ui.column().classes(
                        "pointer-events-auto w-full gap-2 p-2.5 bg-white border border-[#BFDBFE] rounded-2xl shadow-xl"
                    )
                    with self._docking_box:
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("3D-докинг (AutoDock Vina / MMFF94)").classes(
                                "text-xs font-semibold text-[#1E3A8A]"
                            )
                            with ui.button(on_click=self._close_panels).props(
                                "dense flat size=xs color=grey-7"
                            ):
                                ui.html(svg_icon("close", 14), sanitize=False)
                        with ui.grid(columns=3).classes("w-full gap-2"):
                            self._docking_mode_select = (
                                ui.select(
                                    options={
                                        "auto_top_k": "Авто (топ-K)",
                                        "manual_only": "Только вручную",
                                        "skip": "Отключён",
                                    },
                                    value=tc_dock.docking_mode,
                                    label="Режим докинга",
                                    on_change=self._handle_quick_docking_change,
                                )
                                .props("dense outlined options-dense")
                                .classes("w-full text-xs bg-white")
                            )
                            self._docking_pdb_input = (
                                ui.input(
                                    label="PDB ID (опц.)",
                                    value=str(tc_dock.custom_pdb_id or ""),
                                    placeholder="Авто или 1M17",
                                    on_change=self._handle_quick_docking_change,
                                )
                                .props("dense outlined clearable")
                                .classes("w-full text-xs font-mono bg-white")
                            )
                            box_str = (
                                f"{tc_dock.custom_box_center[0]:.2f}, {tc_dock.custom_box_center[1]:.2f}, {tc_dock.custom_box_center[2]:.2f}"
                                if tc_dock.custom_box_center and len(tc_dock.custom_box_center) == 3
                                else ""
                            )
                            self._docking_box_input = (
                                ui.input(
                                    label="Центр X, Y, Z (опц.)",
                                    value=box_str,
                                    placeholder="Авто по ко-лиганду",
                                    on_change=self._handle_quick_docking_change,
                                )
                                .props("dense outlined clearable")
                                .classes("w-full text-xs font-mono bg-white")
                            )
                        with ui.row().classes("w-full items-center justify-between gap-2"):
                            ui.label(
                                "Для готовых кандидатов можно запустить 3D-докинг без перезапуска всего графа"
                            ).classes("text-[11px] text-[#475569]")
                            self._docking_panel_run_btn = (
                                ui.button(on_click=self._handle_run_docking)
                                .props("dense unelevated no-caps size=sm color=primary")
                                .classes("text-[11px] px-2.5 gap-1 rounded-lg shrink-0")
                            )
                            with self._docking_panel_run_btn:
                                ui.html(svg_icon("play", 12), sanitize=False)
                                ui.label("Запустить 3D-докинг").classes("text-[11px] font-medium")
                    self._docking_box.set_visibility(False)

                    # 2e. Раскрываемый журнал верификации и расчёта
                    self._journal_box = ui.column().classes(
                        "pointer-events-auto w-full gap-2 p-2.5 bg-white border border-[#D9BCC5] rounded-2xl shadow-xl"
                    )
                    with self._journal_box:
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Журнал верификации и расчёта").classes(
                                "text-xs font-semibold text-[#5C2D3B]"
                            )
                            with ui.button(on_click=self._close_panels).props(
                                "dense flat size=xs color=grey-7"
                            ):
                                ui.html(svg_icon("close", 14), sanitize=False)
                        with ui.column().classes(
                            "w-full max-h-[340px] overflow-y-auto wetlab-scroll gap-2 pr-1"
                        ):
                            self._journal_timeline_box = ui.column().classes("w-full gap-2")
                            self._journal_spoiler_box = ui.column().classes("w-full")
                    self._journal_box.set_visibility(False)

                # 3. Основной единый островок композера (как в нейрочатах)
                self._composer_card = ui.card().classes(
                    "pointer-events-auto w-full bg-white border border-[#D9BCC5] focus-within:border-[#8B495C] "
                    "rounded-2xl px-3 py-2 gap-1.5 shadow-lg transition-colors"
                )
                with self._composer_card:
                    # Активные контекстные чипы (SMILES / SMARTS / CSV)
                    self._context_chips_row = ui.row().classes("w-full items-center gap-1.5 flex-wrap")
                    with self._context_chips_row:
                        self._structure_chip = ui.label("").classes(
                            "text-[10px] mono-code px-1.5 py-0.5 truncate max-w-[260px]"
                        )
                        self._csv_chip = ui.label("").classes(
                            "text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#ECFDF5] border border-[#A7F3D0] text-[#065F46] truncate max-w-[240px]"
                        )
                    self._context_chips_row.set_visibility(False)

                    # Поле ввода промпта (Enter — отправить, Shift+Enter — новая строка)
                    initial_query = str(self.runtime.state.get("user_query") or "")
                    self._query_input = (
                        ui.textarea(
                            value=initial_query,
                            placeholder="Опишите задачу SAR-дизайна или оптимизации молекулы… (Enter — запуск)",
                        )
                        .props("borderless autogrow rows=1 max-rows=5 dense color=primary")
                        .classes("w-full text-sm composer-textarea")
                    )
                    self._query_input.on("keydown.enter.exact.prevent", self._handle_start_pipeline)

                    # Единая нижняя панель управления задачей внутри островка
                    with ui.row().classes("w-full items-center justify-between gap-1.5 pt-1 border-t border-[#F4EFF1] flex-wrap"):
                        with ui.row().classes("items-center gap-1 flex-wrap"):
                            self._demos_btn = (
                                ui.button(on_click=lambda: self._toggle_panel("demos"))
                                .props("dense flat no-caps size=xs color=grey-8")
                                .classes("text-[11px] px-2 py-1 rounded-lg bg-[#FAF8F9] hover:bg-[#F4EFF1] gap-1")
                                .tooltip("Готовые демо-промпты и предпросчитанные сценарии")
                            )
                            with self._demos_btn:
                                ui.html(svg_icon("sparkles", 12), sanitize=False)
                                ui.label("Демо").classes("text-[11px]")

                            self._csv_btn = (
                                ui.button(on_click=lambda: self._toggle_panel("csv"))
                                .props("dense flat no-caps size=xs color=grey-8")
                                .classes("text-[11px] px-2 py-1 rounded-lg bg-[#FAF8F9] hover:bg-[#F4EFF1] gap-1")
                                .tooltip("Загрузить лабораторную серию (.csv)")
                            )
                            with self._csv_btn:
                                ui.html(svg_icon("paperclip", 12), sanitize=False)
                                ui.label("CSV").classes("text-[11px]")

                            self._sketcher_btn = (
                                ui.button(on_click=lambda: self._toggle_panel("sketcher"))
                                .props("dense flat no-caps size=xs color=grey-8")
                                .classes("text-[11px] px-2 py-1 rounded-lg bg-[#FAF8F9] hover:bg-[#F4EFF1] gap-1")
                                .tooltip("2D-редактор структур и SMARTS-фиксация кора")
                            )
                            with self._sketcher_btn:
                                ui.html(svg_icon("edit", 12), sanitize=False)
                                ui.label("Структура").classes("text-[11px]")

                            self._docking_cfg_btn = (
                                ui.button(on_click=lambda: self._toggle_panel("docking"))
                                .props("dense flat no-caps size=xs color=grey-8")
                                .classes("text-[11px] px-2 py-1 rounded-lg bg-[#EFF6FF] border border-[#DBEAFE] text-[#1D4ED8] gap-1")
                                .tooltip("Настройки 3D-докинга и ручной запуск")
                            )
                            with self._docking_cfg_btn:
                                ui.html(svg_icon("cube", 12), sanitize=False)
                                ui.label("3D-докинг").classes("text-[11px]")

                            self._quick_dock_btn = (
                                ui.button(on_click=self._handle_run_docking)
                                .props("dense outline no-caps size=xs color=primary")
                                .classes("text-[11px] px-2 py-1 rounded-lg bg-white border-[#D9BCC5] text-[#73394B] gap-1")
                                .tooltip("Запустить физический 3D-докинг для топ-кандидатов")
                            )
                            with self._quick_dock_btn:
                                ui.html(svg_icon("play", 11), sanitize=False)
                                ui.label("Докинг").classes("text-[11px] font-medium")

                            self._settings_btn = (
                                ui.button(on_click=self._handle_toggle_settings)
                                .props("dense flat no-caps size=xs color=grey-8")
                                .classes("text-[11px] px-2 py-1 rounded-lg bg-[#FBF5F7] border border-[#E5CFD6] text-[#73394B] gap-1 relative")
                                .tooltip("Открыть настройки вычислительного ядра и оркестрации")
                            )
                            with self._settings_btn:
                                ui.html(svg_icon("sliders", 12), sanitize=False)
                                ui.label("Параметры").classes("text-[11px]")
                                self._overrides_badge = ui.badge("0", color="amber-8").props("floating")
                                self._overrides_badge.set_visibility(False)

                        with ui.row().classes("items-center gap-1.5 shrink-0"):
                            self._cancel_btn = (
                                ui.button(on_click=self._handle_cancel)
                                .props("dense flat no-caps color=negative size=sm")
                                .classes("text-xs px-2.5 py-1 rounded-xl gap-1 bg-[#FFF1F2] border border-[#FECDD3]")
                            )
                            with self._cancel_btn:
                                ui.html(svg_icon("stop", 12), sanitize=False)
                                ui.label("Стоп").classes("text-xs font-medium")

                            self._run_btn = (
                                ui.button(on_click=self._handle_start_pipeline)
                                .props("dense unelevated no-caps")
                                .style("background-color: #8B495C; color: #FFFFFF;")
                                .classes("btn-run-primary text-xs font-medium px-3 py-1.5 gap-1.5 rounded-xl")
                            )
                            with self._run_btn:
                                ui.html(svg_icon("send", 13), sanitize=False)
                                ui.label("Запустить").classes("text-xs font-medium whitespace-nowrap")

                # 4. Стартовые демо-чипы на чистом приветственном экране
                self._welcome_chips_row = ui.row().classes(
                    "w-full items-center justify-center gap-2 flex-wrap pt-1"
                )
                with self._welcome_chips_row:
                    for sc_key, sc_meta in DEMO_SCENARIOS.items():
                        short_title = sc_meta["title"].split(":", 1)[-1].strip()
                        with (
                            ui.button(on_click=lambda k=sc_key: self._apply_demo_prompt(k))
                            .props("dense outline no-caps size=sm color=grey-8")
                            .classes(
                                "text-xs px-3 py-1 rounded-full bg-white border-[#E2D9DD] "
                                "hover:border-[#8B495C] hover:bg-[#FBF5F7] text-[#334155] gap-1.5 shadow-xs"
                            )
                            .tooltip(sc_meta["query"])
                        ):
                            ui.html(svg_icon("sparkles", 12), sanitize=False)
                            ui.label(short_title).classes("text-xs")

        self.refresh()

    def set_floating_mode(self, floating: bool) -> None:
        """Переключает островок между стартовым потоковым видом и плавающим слоем поверх таблицы."""
        was_floating = self._is_floating
        self._is_floating = floating
        if not floating:
            self._composer_expanded = True
        elif not was_floating:
            self._composer_expanded = False
        self._sync_island_chrome()

    def _expand_composer(self) -> None:
        self._composer_expanded = True
        self._sync_island_chrome()

    def _collapse_composer(self) -> None:
        self._composer_expanded = False
        if self._active_panel != "journal":
            self._active_panel = "none"
        self._sync_panel_visibility()
        self._sync_island_chrome()

    def _sync_island_chrome(self) -> None:
        """Синхронизирует плавающее позиционирование, полупрозрачность и видимость тела композера."""
        is_clean = self.runtime.is_clean_welcome
        floating = self._is_floating and not is_clean
        expanded = is_clean or self._composer_expanded

        if self._root_col is not None:
            if floating:
                self._root_col.classes(
                    remove="shrink-0 px-3 pb-2.5 pt-1 gap-2",
                    add="pointer-events-none fixed bottom-3 left-1/2 -translate-x-1/2 z-40 w-[calc(100vw-24px)] gap-1.5",
                )
            else:
                self._root_col.classes(
                    remove="pointer-events-none fixed bottom-3 left-1/2 -translate-x-1/2 z-40 w-[calc(100vw-24px)] gap-1.5",
                    add="shrink-0 px-3 pb-2.5 pt-1 gap-2",
                )

        if self._inner_col is not None:
            if floating and not expanded:
                self._inner_col.classes(remove="max-w-[760px]", add="max-w-[620px]")
            else:
                self._inner_col.classes(remove="max-w-[620px]", add="max-w-[760px]")

        if self._composer_card is not None:
            self._composer_card.set_visibility(expanded)
            if floating:
                self._composer_card.classes(
                    remove="bg-white shadow-lg",
                    add="floating-island-surface shadow-2xl",
                )
            else:
                self._composer_card.classes(
                    remove="floating-island-surface shadow-2xl",
                    add="bg-white shadow-lg",
                )

        if self._speech_bar is not None:
            if floating and not expanded:
                self._speech_bar.classes(
                    remove="bg-[#FBF5F7] shadow-sm",
                    add="floating-island-pill shadow-lg",
                )
            else:
                self._speech_bar.classes(
                    remove="floating-island-pill shadow-lg",
                    add="bg-[#FBF5F7] shadow-sm",
                )

        if self._expand_btn is not None:
            self._expand_btn.set_visibility(floating and not expanded)
        if self._collapse_btn is not None:
            self._collapse_btn.set_visibility(floating and expanded)
        if self._compact_stop_btn is not None:
            self._compact_stop_btn.set_visibility(
                floating and not expanded and self.runtime.status == "running"
            )

    def _toggle_panel(self, panel: _IslandPanel) -> None:
        self._active_panel = "none" if self._active_panel == panel else panel
        if (
            self._active_panel in ("demos", "csv", "sketcher", "docking")
            and self._is_floating
            and not self._composer_expanded
        ):
            self._composer_expanded = True
            self._sync_island_chrome()
        self._sync_panel_visibility()

    def _close_panels(self) -> None:
        self._active_panel = "none"
        self._sync_panel_visibility()

    def _sync_panel_visibility(self) -> None:
        if self._demos_box is not None:
            self._demos_box.set_visibility(self._active_panel == "demos")
        if self._csv_box is not None:
            self._csv_box.set_visibility(self._active_panel == "csv")
        if self._sketcher_box is not None:
            self._sketcher_box.set_visibility(self._active_panel == "sketcher")
        if self._docking_box is not None:
            self._docking_box.set_visibility(self._active_panel == "docking")
        if self._journal_box is not None:
            self._journal_box.set_visibility(self._active_panel == "journal")
        if self._journal_chevron_box is not None:
            icon_name = "chevron_down" if self._active_panel == "journal" else "chevron_up"
            self._journal_chevron_box.set_content(svg_icon(icon_name, 12))

    def _handle_toggle_settings(self) -> None:
        self._close_panels()
        if self._on_toggle_settings is not None:
            self._on_toggle_settings()

    def _apply_demo_prompt(self, scenario_key: str) -> None:
        if self.runtime.status in ("running", "paused_hitl"):
            ui.notify("Сначала остановите или завершите текущий расчёт", type="warning")
            return
        self.runtime.active_scenario_key = scenario_key
        if self._is_floating:
            self._composer_expanded = True
        self.apply_scenario_context(scenario_key)
        self._close_panels()
        if scenario_key == "scenario3_wetlab":
            self._load_sample_wetlab_csv()

    async def _open_saved_scenario(self, scenario_key: str) -> None:
        restored = session_registry.get(scenario_key) or await session_registry.restore_from_workspace(
            scenario_key
        )
        if restored is None:
            ui.notify(f"Сохранённый сценарий '{scenario_key}' не найден", type="warning")
            return
        restored.active_scenario_key = scenario_key
        self._close_panels()
        if self._on_session_switched is not None:
            res = self._on_session_switched(scenario_key)
            if asyncio.iscoroutine(res):
                await res

    @staticmethod
    def _parse_box_coords(raw: str) -> list[float] | None:
        parts = [p.strip() for p in str(raw or "").replace(";", ",").split(",") if p.strip()]
        if len(parts) != 3:
            return None
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None

    def _handle_quick_docking_change(self, _e: Any = None) -> None:
        if self._syncing_docking:
            return
        overrides: dict[str, Any] = {}
        if self._docking_mode_select is not None and self._docking_mode_select.value:
            overrides["docking_mode"] = str(self._docking_mode_select.value)
        if self._docking_pdb_input is not None:
            pdb_val = str(self._docking_pdb_input.value or "").strip().upper()
            overrides["custom_pdb_id"] = pdb_val or None
        if self._docking_box_input is not None:
            box_raw = str(self._docking_box_input.value or "").strip()
            if not box_raw:
                overrides["custom_box_center"] = None
            else:
                parsed = self._parse_box_coords(box_raw)
                if parsed is not None:
                    overrides["custom_box_center"] = parsed
        if overrides:
            changed = agent_bridge.apply_ui_tool_overrides(self.runtime, overrides)
            if changed and self._on_tool_config_changed is not None:
                self._on_tool_config_changed()

    async def _handle_run_docking(self) -> None:
        cands = list(self.runtime.view_model.candidates_by_id.values())
        if not cands:
            ui.notify("Сначала сгенерируйте кандидатов", type="warning")
            return
        self._close_panels()
        if self._quick_dock_btn is not None:
            self._quick_dock_btn.disable()
        if self._docking_panel_run_btn is not None:
            self._docking_panel_run_btn.disable()
        try:
            summary = await agent_bridge.run_standalone_docking(self.runtime.session_id)
            status = summary.get("status")
            if status == "SUCCESS":
                dg = summary.get("best_score_kcal")
                dg_txt = f" (лучший ΔG = {dg:.2f} ккал/моль)" if dg is not None else ""
                ui.notify(f"3D-докинг завершён{dg_txt}", type="positive")
            else:
                ui.notify(
                    str(summary.get("message") or f"Статус докинга: {status}"),
                    type="warning",
                )
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Ошибка 3D-докинга: {exc}", type="negative")
        finally:
            self.refresh()

    def apply_scenario_context(self, scenario_key: str) -> None:
        """Обновляет текст запроса и SMILES прототипа при выборе демо-сценария."""
        preset = DEMO_SCENARIOS.get(scenario_key)
        if preset and self.runtime.status not in ("running", "paused_hitl"):
            if self._query_input is not None:
                self._query_input.set_value(preset["query"])
            lead_smi = preset.get("lead_smiles", "")
            if self._smiles_input is not None:
                self._smiles_input.set_value(lead_smi)
            if self._sketcher is not None:
                self._sketcher.set_smiles(lead_smi)
        self.refresh()

    async def _handle_csv_upload(self, e: Any) -> None:
        upload_dir = settings.resolved_workspace_dir / self.runtime.session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        target_path = upload_dir / (
            getattr(e, "name", "uploaded_series.csv") or "uploaded_series.csv"
        )
        content = getattr(e, "content", None)
        if content is not None:
            data = content.read()
            if asyncio.iscoroutine(data):
                data = await data
            target_path.write_bytes(data)
            self.uploaded_csv_path = target_path
            if self._csv_status_label is not None:
                self._csv_status_label.set_text(f"Активный файл: {target_path.name}")
            self._sync_context_chips()
            ui.notify(f"Загружен файл: {target_path.name}", type="positive")

    def _load_sample_wetlab_csv(self) -> None:
        sample = settings.samples_dir / "custom_wetlab_series.csv"
        if not sample.is_file():
            ui.notify("Демо-файл custom_wetlab_series.csv не найден", type="warning")
            return
        self.uploaded_csv_path = sample
        if self._csv_status_label is not None:
            self._csv_status_label.set_text(f"Активный файл: {sample.name}")
        self._sync_context_chips()
        ui.notify(f"Подключён файл: {sample.name}", type="info")

    def _clear_csv(self) -> None:
        self.uploaded_csv_path = None
        if self._csv_status_label is not None:
            self._csv_status_label.set_text(
                "Выберите лабораторный CSV-файл (SMILES, pIC50/IC50)"
            )
        self._sync_context_chips()

    def _handle_sketcher_change(self, smiles: str) -> None:
        self._last_sketcher_smiles = smiles
        if self._smiles_input is not None and smiles:
            self._smiles_input.set_value(smiles)
        self._sync_context_chips()

    def get_sketcher_smiles(self) -> str:
        """Возвращает текущий SMILES из редактора или поля SMILES прототипа."""
        if self._last_sketcher_smiles:
            return self._last_sketcher_smiles
        if self._smiles_input is not None:
            return str(self._smiles_input.value or "").strip()
        return ""

    def set_sketcher_smiles(self, smiles: str) -> None:
        """Устанавливает SMILES в 2D-редактор и поле SMILES."""
        self._last_sketcher_smiles = smiles
        if self._smiles_input is not None:
            self._smiles_input.set_value(smiles)
        if self._sketcher is not None:
            self._sketcher.set_smiles(smiles)
        self.refresh()

    def load_scaffold(self, scaffold_smiles: str) -> None:
        """Загружает выбранный скаффолд в островок и раскрывает редактор для правки."""
        self.set_sketcher_smiles(scaffold_smiles)
        self._composer_expanded = True
        self._active_panel = "sketcher"
        self.refresh()
        ui.notify("Скаффолд загружен в редактор структур", type="info")

    def set_protected_smarts(self, smarts: str) -> None:
        """Синхронизирует поле `_smarts_input` с настройками параметров."""
        if self._smarts_input is None:
            return
        self._syncing_smarts = True
        try:
            self._smarts_input.set_value(smarts)
        finally:
            self._syncing_smarts = False
        self._sync_context_chips()

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
        self._sync_context_chips()
        if changed and self._on_tool_config_changed is not None:
            self._on_tool_config_changed()

    def _sync_smiles_to_sketcher(self) -> None:
        if self._sketcher is not None and self._smiles_input is not None:
            smi = str(self._smiles_input.value or "")
            self._last_sketcher_smiles = smi
            self._sketcher.set_smiles(smi)

    def _sync_context_chips(self) -> None:
        smi_txt = str(self._smiles_input.value or "").strip() if self._smiles_input else ""
        smt_txt = str(self._smarts_input.value or "").strip() if self._smarts_input else ""
        struct_txt = smt_txt or smi_txt
        csv_txt = self.uploaded_csv_path.name if self.uploaded_csv_path else ""

        if self._structure_chip is not None:
            prefix = "SMARTS: " if smt_txt else "SMILES: "
            self._structure_chip.set_text(f"{prefix}{struct_txt}" if struct_txt else "")
            self._structure_chip.set_visibility(bool(struct_txt))
        if self._csv_chip is not None:
            self._csv_chip.set_text(f"CSV: {csv_txt}" if csv_txt else "")
            self._csv_chip.set_visibility(bool(csv_txt))
        if self._context_chips_row is not None:
            self._context_chips_row.set_visibility(bool(struct_txt or csv_txt))

    async def _handle_start_pipeline(self, _e: Any = None) -> None:
        if self.runtime.status == "running":
            return
        query = str(self._query_input.value or "").strip() if self._query_input else ""
        if not query:
            ui.notify("Введите исследовательский запрос", type="warning")
            return
        self._close_panels()
        self._is_floating = True
        self._composer_expanded = False
        self._sync_island_chrome()

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
        base_tc = resolve_tool_config(self.runtime.state)
        overrides: dict[str, Any] = {}
        if smarts_val and ">>" not in smarts_val:
            overrides["protected_smarts"] = smarts_val
        if self._docking_mode_select is not None and self._docking_mode_select.value:
            overrides["docking_mode"] = str(self._docking_mode_select.value)
        if self._docking_pdb_input is not None:
            pdb_val = str(self._docking_pdb_input.value or "").strip().upper()
            overrides["custom_pdb_id"] = pdb_val or None
        if self._docking_box_input is not None:
            box_raw = str(self._docking_box_input.value or "").strip()
            if not box_raw:
                overrides["custom_box_center"] = None
            else:
                parsed = self._parse_box_coords(box_raw)
                if parsed is not None:
                    overrides["custom_box_center"] = parsed

        tool_cfg, changed_keys = base_tc.apply_overrides(
            overrides,
            allow_none_keys={"protected_smarts", "custom_pdb_id", "custom_box_center"},
        )
        existing_sources = dict(self.runtime.state.get("param_sources") or {})
        for k in changed_keys:
            existing_sources[k] = "ui_manual"
        param_srcs: dict[str, Any] | None = existing_sources or None

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
        ui.notify("Выполнение расчёта остановлено", type="info")

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

    def refresh(self) -> None:
        """Синхронизирует состояние единого нижнего островка с `SessionRuntime`."""
        status = self.runtime.status
        is_paused = status == "paused_hitl"
        is_running = status == "running"
        is_docking = self.runtime.is_docking_running
        is_busy = is_running or is_docking
        has_candidates = len(self.runtime.view_model.candidates_by_id) > 0

        # Живая реплика Пишовичка или краткий итог верификации
        is_clean = self.runtime.is_clean_welcome
        speech_text = self.runtime.get_pishovichok_status_text()
        traces = list(self.runtime.state.get("trace_log", []))
        summary_text, summary_tone = format_pishovichok_journal_summary(self.runtime)
        display_text = speech_text or summary_text or ("Готов к новому запросу" if not is_clean else "")
        show_speech = (not is_clean) or bool(display_text) or bool(traces)
        if self._speech_bar is not None:
            self._speech_bar.set_visibility(show_speech)
        if self._speech_spinner is not None:
            self._speech_spinner.set_visibility(is_busy)
        if self._speech_dot is not None:
            self._speech_dot.set_visibility(not is_busy and show_speech)
            dot_cls = "bg-[#D97706]"
            if status == "error" or summary_tone == "error":
                dot_cls = "bg-[#E11D48]"
            elif not speech_text and summary_tone == "approved":
                dot_cls = "bg-[#059669]"
            elif not speech_text and not summary_text:
                dot_cls = "bg-[#8B495C]"
            self._speech_dot.classes(
                remove="bg-[#D97706] bg-[#E11D48] bg-[#059669] bg-[#8B495C]",
                add=dot_cls,
            )
        if self._speech_label is not None:
            self._speech_label.set_text(display_text)
        if self._journal_toggle_btn is not None:
            self._journal_toggle_btn.set_visibility(bool(traces) or is_busy)
        if self._journal_toggle_label is not None:
            self._journal_toggle_label.set_text(
                f"Журнал ({len(traces)})" if traces else "Журнал"
            )
        if self._journal_timeline_box is not None:
            render_verification_journal(
                self.runtime,
                timeline_box=self._journal_timeline_box,
                state_spoiler_box=self._journal_spoiler_box,
            )

        # Стартовые демо-чипы видны только на чистом приветственном экране
        if self._welcome_chips_row is not None:
            self._welcome_chips_row.set_visibility(self.runtime.is_clean_welcome)

        # Кнопка быстрого запуска докинга видна, когда есть кандидаты
        if self._quick_dock_btn is not None:
            self._quick_dock_btn.set_visibility(has_candidates)
            self._quick_dock_btn.set_enabled(not is_busy)
        if self._docking_panel_run_btn is not None:
            self._docking_panel_run_btn.set_enabled(has_candidates and not is_busy)

        # Бейдж изменённых параметров ядра
        if self._overrides_badge is not None:
            sources = self.runtime.state.get("param_sources", {})
            non_default = sum(1 for src in sources.values() if src != "default")
            self._overrides_badge.set_text(str(non_default))
            self._overrides_badge.set_visibility(non_default > 0)

        if self._hitl_card is not None:
            self._hitl_card.set_visibility(is_paused)
        if self._hitl_prompt_label is not None:
            self._hitl_prompt_label.set_text(self.runtime.hitl_pause_reason or "")

        if self._run_btn is not None:
            self._run_btn.set_enabled(not is_busy)
        if self._cancel_btn is not None:
            self._cancel_btn.set_visibility(is_running)

        tc = resolve_tool_config(self.runtime.state)
        self._syncing_docking = True
        try:
            if (
                self._docking_mode_select is not None
                and self._docking_mode_select.value != tc.docking_mode
            ):
                self._docking_mode_select.set_value(tc.docking_mode)
            if self._docking_pdb_input is not None:
                target_pdb = str(tc.custom_pdb_id or "")
                if str(self._docking_pdb_input.value or "").strip().upper() != target_pdb:
                    self._docking_pdb_input.set_value(target_pdb)
            if self._docking_box_input is not None:
                target_box = (
                    f"{tc.custom_box_center[0]:.2f}, {tc.custom_box_center[1]:.2f}, {tc.custom_box_center[2]:.2f}"
                    if tc.custom_box_center and len(tc.custom_box_center) == 3
                    else ""
                )
                if str(self._docking_box_input.value or "").strip() != target_box:
                    self._docking_box_input.set_value(target_box)
        finally:
            self._syncing_docking = False

        if self._smarts_input is not None:
            cur_input_val = str(self._smarts_input.value or "").strip()
            if ">>" not in cur_input_val:
                target_smarts = str(tc.protected_smarts or "")
                if cur_input_val != target_smarts:
                    self.set_protected_smarts(target_smarts)

        self._sync_panel_visibility()
        self._sync_context_chips()
        self._sync_island_chrome()
