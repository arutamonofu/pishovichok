"""Плавающий нижний островок постановки задачи поверх рабочей области (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from pishovichok.config import settings
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import QueryConstraints, ToolConfig
from pishovichok.ui.bridge import agent_bridge
from pishovichok.ui.components.agent_rail import DEMO_SCENARIOS
from pishovichok.ui.components.chem_widgets import create_sketcher
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import HITLDecision, SessionRuntime


class TaskCommandIsland:
    """Плавающий островок ввода задачи внизу центральной области.

    В состоянии `idle` отображается как заметная карточка ввода с быстрым доступом
    к редактору структур и загрузке CSV; после запуска/завершения автоматически
    сворачивается в полупрозрачную компактную плашку, не перекрывая таблицу.
    """

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_tool_config_changed: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_tool_config_changed = on_tool_config_changed
        self.uploaded_csv_path: Path | None = None
        self._syncing_smarts: bool = False
        self._last_sketcher_smiles: str = ""
        self._expanded: bool = self.runtime.status in ("idle", "paused_hitl")
        self._show_sketcher: bool = False
        self._show_csv: bool = False

        self._expanded_card: ui.card | None = None
        self._minimized_pill: ui.row | None = None
        self._pill_dot: ui.element | None = None
        self._pill_query_lbl: ui.label | None = None
        self._pill_stop_btn: ui.button | None = None

        self._csv_btn: ui.button | None = None
        self._csv_box: ui.column | None = None
        self._csv_status_label: ui.label | None = None
        self._sketcher_box: ui.column | None = None
        self._query_input: ui.textarea | None = None
        self._smiles_input: ui.input | None = None
        self._smarts_input: ui.input | None = None
        self._sketcher: Any = None
        self._run_btn: ui.button | None = None
        self._cancel_btn: ui.button | None = None
        self._hitl_card: ui.card | None = None
        self._hitl_prompt_label: ui.label | None = None
        self._structure_chip: ui.label | None = None
        self._show_docking: bool = False
        self._syncing_docking: bool = False
        self._docking_box: ui.column | None = None
        self._docking_mode_select: ui.select | None = None
        self._docking_pdb_input: ui.input | None = None
        self._docking_box_input: ui.input | None = None
        self._build()

    def _is_csv_scenario_active(self) -> bool:
        constraints = self.runtime.state.get("query_constraints")
        sc_type = getattr(constraints, "scenario_type", None)
        return (
            self.runtime.active_scenario_key == "scenario3_wetlab"
            or self.runtime.session_id == "scenario3_wetlab"
            or sc_type == "custom_csv_analysis"
            or self.uploaded_csv_path is not None
            or bool(self.runtime.state.get("uploaded_csv_path"))
        )

    def _resolve_active_query_text(self) -> str:
        sc_key = self.runtime.active_scenario_key or self.runtime.session_id
        sc_query = DEMO_SCENARIOS.get(sc_key, {}).get("query", "")
        return str(
            self.runtime.state.get("user_query")
            or (self._query_input.value if self._query_input is not None else "")
            or sc_query
            or "Постановка задачи"
        )

    def _build(self) -> None:
        with ui.element("div").classes(
            "pointer-events-none absolute inset-x-0 bottom-6 z-30 flex justify-center px-4"
        ):
            # 1. Компактная приглушённая плашка (после запуска / завершения)
            self._minimized_pill = ui.row().classes(
                "pointer-events-auto items-center gap-2.5 max-w-[560px] px-3 py-1.5 "
                "rounded-full bg-white/90 backdrop-blur-md border border-[#E5CFD6] "
                "shadow-md opacity-85 hover:opacity-100 hover:bg-white hover:shadow-lg "
                "hover:border-[#8B495C] transition-all"
            )
            with self._minimized_pill:
                with ui.row().classes("items-center gap-2 cursor-pointer min-w-0").on(
                    "click", self._expand
                ):
                    ui.html(
                        '<div class="w-6 h-6 rounded-full bg-[#FBF5F7] border border-[#E5CFD6] overflow-hidden flex items-center justify-center shrink-0">'
                        '<img src="/static/mascot_head.png" alt="Пишовичок" class="w-5 h-5 object-contain" />'
                        "</div>",
                        sanitize=False,
                    )
                    self._pill_dot = ui.element("span").classes(
                        "w-2 h-2 rounded-full bg-[#059669] shrink-0"
                    )
                    self._pill_query_lbl = ui.label("").classes(
                        "text-xs text-[#334155] truncate max-w-[330px]"
                    )
                self._pill_stop_btn = (
                    ui.button(on_click=self._handle_cancel)
                    .props("dense flat no-caps size=xs color=negative")
                    .classes("text-[11px] px-1.5 gap-1 shrink-0")
                )
                with self._pill_stop_btn:
                    ui.html(svg_icon("stop", 12), sanitize=False)
                    ui.label("Стоп").classes("text-[11px]")

                with ui.button(on_click=self._expand).props(
                    "dense flat no-caps size=xs color=grey-8"
                ).classes("text-[11px] px-1.5 gap-1 shrink-0 text-[#73394B]"):
                    ui.html(svg_icon("edit", 12), sanitize=False)
                    ui.label("Изменить").classes("text-[11px]")

            # 2. Полный заметный островок постановки задачи
            self._expanded_card = ui.card().classes(
                "pointer-events-auto w-full max-w-[660px] bg-white border border-[#D9BCC5] "
                "rounded-2xl p-2.5 gap-2 shadow-xl"
            )
            with self._expanded_card:
                # Верхняя строка: маскот + заголовок + чипы + переключатели скетчера / CSV / свернуть
                with ui.row().classes("w-full items-center justify-between gap-2 no-wrap"):
                    with ui.row().classes("items-center gap-2 min-w-0"):
                        ui.html(
                            '<div class="w-7 h-7 rounded-full bg-[#FBF5F7] border border-[#E5CFD6] overflow-hidden flex items-center justify-center shrink-0">'
                            '<img src="/static/mascot_head.png" alt="Пишовичок" class="w-6 h-6 object-contain" />'
                            "</div>",
                            sanitize=False,
                        )
                        ui.label("Постановка задачи").classes(
                            "text-xs font-semibold text-[#5C2D3B] shrink-0"
                        )
                        self._structure_chip = ui.label("").classes(
                            "text-[10px] mono-code px-1.5 py-0.5 truncate max-w-[200px]"
                        )
                        self._structure_chip.set_visibility(False)

                    with ui.row().classes("items-center gap-1 shrink-0"):
                        self._csv_btn = (
                            ui.button(on_click=self._toggle_csv_panel)
                            .props("dense flat no-caps size=xs color=grey-8")
                            .classes("text-[11px] px-2 py-0.5 rounded bg-[#F4F1F3] gap-1")
                            .tooltip("Загрузить лабораторную серию (.csv)")
                        )
                        with self._csv_btn:
                            ui.html(svg_icon("paperclip", 12), sanitize=False)
                            ui.label("CSV").classes("text-[11px]")

                        with ui.button(on_click=self._toggle_sketcher_panel).props(
                            "dense flat no-caps size=xs color=grey-8"
                        ).classes("text-[11px] px-2 py-0.5 rounded bg-[#FBF5F7] border border-[#E5CFD6] text-[#73394B] gap-1").tooltip(
                            "Редактор структур и SMARTS-ограничений"
                        ):
                            ui.html(svg_icon("edit", 12), sanitize=False)
                            ui.label("Редактор структур").classes("text-[11px]")

                        with ui.button(on_click=self._toggle_docking_panel).props(
                            "dense flat no-caps size=xs color=grey-8"
                        ).classes("text-[11px] px-2 py-0.5 rounded bg-[#EFF6FF] border border-[#BFDBFE] text-[#1D4ED8] gap-1").tooltip(
                            "Настройки 3D-докинга перед запуском"
                        ):
                            ui.label("3D-докинг").classes("text-[11px]")

                        min_btn = (
                            ui.button(on_click=self._minimize)
                            .props("dense flat size=xs color=grey-7")
                            .classes("p-1 min-w-0")
                            .tooltip("Свернуть островок")
                        )
                        min_btn._props["title"] = "Свернуть"
                        with min_btn:
                            ui.html(svg_icon("chevron_down", 14), sanitize=False)

                # Панель паузы контроля исследователя (HITL)
                self._hitl_card = ui.card().classes(
                    "w-full bg-[#FFFBEB] border border-[#F59E0B] p-2 gap-1.5 shadow-none rounded-lg"
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

                # Выдвижная секция загрузки CSV (для сценария 3 или по кнопке)
                self._csv_box = ui.column().classes(
                    "w-full gap-1.5 p-2 bg-[#FAF8F9] border border-[#E2D9DD] rounded-lg"
                )
                with self._csv_box:
                    self._csv_status_label = ui.label(
                        "Выберите лабораторный CSV-файл (SMILES, pIC50/IC50)"
                    ).classes("text-[11px] text-[#594D52]")
                    ui.upload(
                        on_upload=self._handle_csv_upload,
                        auto_upload=True,
                    ).props("accept=.csv flat bordered color=primary label='Выбрать CSV'").classes(
                        "w-full"
                    )
                    with ui.row().classes("w-full items-center justify-between gap-1"):
                        ui.button(
                            "Демо-серия (.csv)",
                            on_click=self._load_sample_wetlab_csv,
                        ).props("dense flat size=sm no-caps color=primary").classes("text-[11px]")
                        ui.button(
                            "Сброс",
                            on_click=self._clear_csv,
                        ).props("dense flat size=sm no-caps color=grey-7").classes("text-[11px]")
                self._csv_box.set_visibility(False)

                # Выдвижная секция редактора структур и ограничений
                self._sketcher_box = ui.column().classes(
                    "w-full gap-1.5 p-2 bg-[#FAF8F9] border border-[#E2D9DD] rounded-lg"
                )
                with self._sketcher_box:
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
                    ).props("dense flat no-caps size=xs color=primary").classes("text-[11px]")
                self._sketcher_box.set_visibility(False)

                # Компактные настройки 3D-докинга (управляемый опциональный этап)
                tc_dock = resolve_tool_config(self.runtime.state)
                self._docking_box = ui.column().classes(
                    "w-full gap-1.5 p-2 bg-[#F8FAFC] border border-[#CBD5E1] rounded-lg"
                )
                with self._docking_box:
                    ui.label("Настройки 3D-докинга").classes(
                        "text-[11px] font-semibold text-[#1E3A8A]"
                    )
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
                self._docking_box.set_visibility(False)

                # Основная строка ввода как в чате + кнопка запуска
                initial_query = str(self.runtime.state.get("user_query") or "")
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    self._query_input = (
                        ui.textarea(
                            value=initial_query,
                            placeholder="Опишите задачу SAR-анализа или оптимизации прототипа...",
                        )
                        .props("outlined autogrow rows=2 dense color=primary")
                        .classes("flex-1 text-xs bg-[#FAF8F9] rounded-xl")
                    )
                    with ui.column().classes("gap-1 shrink-0"):
                        self._run_btn = (
                            ui.button(on_click=self._handle_start_pipeline)
                            .props("dense unelevated no-caps")
                            .style("background-color: #8B495C; color: #FFFFFF;")
                            .classes("btn-run-primary text-xs font-medium px-3.5 py-2 gap-1.5 rounded-xl")
                        )
                        with self._run_btn:
                            ui.html(svg_icon("play", 13), sanitize=False)
                            ui.label("Запустить расчёт").classes("text-xs font-medium whitespace-nowrap")

                        self._cancel_btn = (
                            ui.button(on_click=self._handle_cancel)
                            .props("dense flat no-caps color=negative size=sm")
                            .classes("text-xs px-2 gap-1 w-full")
                        )
                        with self._cancel_btn:
                            ui.html(svg_icon("stop", 12), sanitize=False)
                            ui.label("Стоп").classes("text-xs")

        self.refresh()

    def _expand(self, _e: Any = None) -> None:
        self._expanded = True
        self.refresh()

    def _minimize(self, _e: Any = None) -> None:
        self._expanded = False
        self._show_sketcher = False
        self._show_csv = False
        self.refresh()

    def _toggle_sketcher_panel(self) -> None:
        self._show_sketcher = not self._show_sketcher
        if self._sketcher_box is not None:
            self._sketcher_box.set_visibility(self._show_sketcher)

    def _toggle_csv_panel(self) -> None:
        self._show_csv = not self._show_csv
        if self._csv_box is not None:
            self._csv_box.set_visibility(self._show_csv)

    def _toggle_docking_panel(self) -> None:
        self._show_docking = not self._show_docking
        if self._docking_box is not None:
            self._docking_box.set_visibility(self._show_docking)

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

    def apply_scenario_context(self, scenario_key: str) -> None:
        """Обновляет блок загрузки CSV и текст запроса при выборе сценария."""
        preset = DEMO_SCENARIOS.get(scenario_key)
        is_csv = scenario_key == "scenario3_wetlab"
        self._show_csv = is_csv
        if preset and self.runtime.status == "idle":
            self._expanded = True
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
        target_path = upload_dir / (getattr(e, "name", "uploaded_series.csv") or "uploaded_series.csv")
        content = getattr(e, "content", None)
        if content is not None:
            data = content.read()
            if asyncio.iscoroutine(data):
                data = await data
            target_path.write_bytes(data)
            self.uploaded_csv_path = target_path
            if self._csv_status_label is not None:
                self._csv_status_label.set_text(f"Активный файл: {target_path.name}")
            ui.notify(f"Загружен файл: {target_path.name}", type="positive")

    def _load_sample_wetlab_csv(self) -> None:
        sample = settings.samples_dir / "custom_wetlab_series.csv"
        if not sample.is_file():
            ui.notify("Демо-файл custom_wetlab_series.csv не найден", type="warning")
            return
        self.uploaded_csv_path = sample
        if self._csv_status_label is not None:
            self._csv_status_label.set_text(f"Активный файл: {sample.name}")
        ui.notify(f"Подключён файл: {sample.name}", type="info")

    def _clear_csv(self) -> None:
        self.uploaded_csv_path = None
        if self._csv_status_label is not None:
            self._csv_status_label.set_text(
                "Выберите лабораторный CSV-файл (SMILES, pIC50/IC50)"
            )

    def _handle_sketcher_change(self, smiles: str) -> None:
        self._last_sketcher_smiles = smiles
        if self._smiles_input is not None and smiles:
            self._smiles_input.set_value(smiles)

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
        """Загружает выбранный скаффолд в островок и раскрывает его для правки."""
        self.set_sketcher_smiles(scaffold_smiles)
        self._expanded = True
        self._show_sketcher = True
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
        if self._structure_chip is not None:
            smi_txt = str(self._smiles_input.value or "").strip() if self._smiles_input else ""
            chip = smarts.strip() or smi_txt
            self._structure_chip.set_text(chip)
            self._structure_chip.set_visibility(bool(chip))

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

    async def _handle_start_pipeline(self) -> None:
        query = str(self._query_input.value or "").strip() if self._query_input else ""
        if not query:
            ui.notify("Введите исследовательский запрос", type="warning")
            return
        self._expanded = False
        self._show_sketcher = False
        self._show_csv = False
        self.refresh()

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
        """Синхронизирует состояние плавающего островка с `SessionRuntime`."""
        status = self.runtime.status
        is_paused = status == "paused_hitl"
        is_running = status == "running"
        if is_paused:
            self._expanded = True

        if self._expanded_card is not None:
            self._expanded_card.set_visibility(self._expanded)
        if self._minimized_pill is not None:
            self._minimized_pill.set_visibility(not self._expanded)

        if self._pill_query_lbl is not None:
            self._pill_query_lbl.set_text(self._resolve_active_query_text())
        if self._pill_dot is not None:
            self._pill_dot.classes(
                remove="bg-[#059669] bg-[#8B495C] bg-[#64748B] animate-pulse",
                add=(
                    "bg-[#8B495C] animate-pulse"
                    if is_running
                    else ("bg-[#059669]" if status == "completed" else "bg-[#64748B]")
                ),
            )
        if self._pill_stop_btn is not None:
            self._pill_stop_btn.set_visibility(is_running)

        show_csv_btn = self._is_csv_scenario_active() or self.uploaded_csv_path is not None
        if self._csv_btn is not None:
            self._csv_btn.set_visibility(show_csv_btn)
        if self._csv_box is not None:
            self._csv_box.set_visibility(self._show_csv and self._expanded)
        if self._sketcher_box is not None:
            self._sketcher_box.set_visibility(self._show_sketcher and self._expanded)
        if self._docking_box is not None:
            self._docking_box.set_visibility(self._show_docking and self._expanded)

        if self._hitl_card is not None:
            self._hitl_card.set_visibility(is_paused)
        if self._hitl_prompt_label is not None:
            self._hitl_prompt_label.set_text(self.runtime.hitl_pause_reason or "")

        if self._run_btn is not None:
            self._run_btn.set_enabled(not is_running)
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

        if self._structure_chip is not None:
            smi_txt = str(self._smiles_input.value or "").strip() if self._smiles_input else ""
            smt_txt = str(self._smarts_input.value or "").strip() if self._smarts_input else ""
            chip = smt_txt or smi_txt
            self._structure_chip.set_text(chip)
            self._structure_chip.set_visibility(bool(chip))
