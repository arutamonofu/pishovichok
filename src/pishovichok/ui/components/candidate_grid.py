"""Виртуализированная таблица кандидатов с двухуровневой 4-категорийной шапкой (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import run, ui

from pishovichok.agents.workflow import (
    export_candidates_to_csv,
    export_candidates_to_sdf,
)
from pishovichok.config import settings
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import ToolConfig
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import SessionRuntime


def build_grid_telemetry_badges(tool_config: ToolConfig) -> list[tuple[str, str]]:
    """Формирует компактные плашки активных параметров расчёта для режима «Хемоинформатик»."""
    cats = (
        ", ".join(tool_config.active_filter_catalogs)
        if tool_config.active_filter_catalogs
        else "нет"
    )
    return [
        (
            f"FP: {tool_config.fp_type}/r{tool_config.fp_radius}/{tool_config.fp_nbits}",
            "Отпечатки молекул (fp_type / fp_radius / fp_nbits)",
        ),
        (
            f"Сходство: {tool_config.similarity_metric}",
            "Метрика сходства",
        ),
        (
            f"Перепады: S≥{tool_config.cliff_similarity_cutoff:.2f} Δ≥{tool_config.cliff_delta_pic50_cutoff:.1f}",
            "Пороги детекции перепадов активности",
        ),
        (
            f"ADMET: {tool_config.admet_backend}",
            "Выбранный движок прогноза ADMET",
        ),
        (
            f"Фильтры: {cats} [{tool_config.alert_penalty_mode}]",
            "Активные каталоги структурных фильтров и режим штрафа",
        ),
        (
            f"BRICS≥{tool_config.brics_min_fragment_size}",
            "Минимальный размер фрагмента BRICS",
        ),
    ]


def build_candidate_column_defs(
    *,
    include_docking: bool = False,
) -> list[dict[str, Any]]:
    """Формирует двухуровневые заголовки `columnDefs` с цветовыми категориями знаний."""
    groups: list[dict[str, Any]] = [
        {
            "headerName": "1. Эксперимент",
            "headerClass": "ag-cat-experiment",
            "children": [
                {
                    "headerName": "Структура",
                    "field": "preview_html",
                    "width": 176,
                    "minWidth": 170,
                    "pinned": "left",
                    "sortable": False,
                    "filter": False,
                    "resizable": True,
                    "cellClass": "ag-cell-structure",
                    ":cellRenderer": "(params) => params.value || ''",
                    "headerClass": "ag-cat-experiment-col",
                },
                {
                    "headerName": "Прототип",
                    "field": "parent_chembl_id",
                    "width": 152,
                    "minWidth": 140,
                    "filter": "agTextColumnFilter",
                    "headerClass": "ag-cat-experiment-col",
                },
                {
                    "headerName": "pIC50 (эксп.)",
                    "field": "parent_exp_pic50",
                    "width": 108,
                    "minWidth": 100,
                    "filter": "agNumberColumnFilter",
                    "tooltipField": "parent_bao_tooltip",
                    "headerTooltip": "Экспериментальный pIC50 прототипа (протокол BAO)",
                    "headerClass": "ag-cat-experiment-col",
                },
            ],
        },
        {
            "headerName": "2. Расчёт RDKit",
            "headerClass": "ag-cat-calc",
            "children": [
                {
                    "headerName": "cLogP (Δ)",
                    "field": "clogp_display",
                    "width": 116,
                    "minWidth": 106,
                    "filter": "agTextColumnFilter",
                    "headerClass": "ag-cat-calc-col",
                },
                {
                    "headerName": "SAscore",
                    "field": "sa_score",
                    "width": 88,
                    "minWidth": 80,
                    "filter": "agNumberColumnFilter",
                    "headerClass": "ag-cat-calc-col",
                },
                {
                    "headerName": "QED",
                    "field": "qed",
                    "width": 76,
                    "minWidth": 70,
                    "filter": "agNumberColumnFilter",
                    "headerClass": "ag-cat-calc-col",
                },
                {
                    "headerName": "Фильтры",
                    "field": "alerts_display",
                    "width": 98,
                    "minWidth": 90,
                    "filter": "agTextColumnFilter",
                    "headerClass": "ag-cat-calc-col",
                },
            ],
        },
        {
            "headerName": "3. Прогноз ADMET",
            "headerClass": "ag-cat-pred",
            "children": [
                {
                    "headerName": "pIC50 (прогноз)",
                    "field": "pred_pic50_display",
                    "width": 130,
                    "minWidth": 122,
                    "filter": "agTextColumnFilter",
                    "headerClass": "ag-cat-pred-col",
                },
                {
                    "headerName": "Риск hERG",
                    "field": "herg_display",
                    "width": 124,
                    "minWidth": 114,
                    "filter": "agTextColumnFilter",
                    "headerClass": "ag-cat-pred-col",
                },
                {
                    "headerName": "LogS",
                    "field": "log_s_aq",
                    "width": 82,
                    "minWidth": 74,
                    "filter": "agNumberColumnFilter",
                    "headerClass": "ag-cat-pred-col",
                },
            ],
        },
        {
            "headerName": "4. Гипотеза SAR",
            "headerClass": "ag-cat-hypothesis",
            "children": [
                {
                    "headerName": "Правило",
                    "field": "method_rule_display",
                    "width": 136,
                    "minWidth": 124,
                    "filter": "agTextColumnFilter",
                    "headerClass": "ag-cat-hypothesis-col",
                },
                {
                    "headerName": "Индекс MPO",
                    "field": "mpo_score",
                    "width": 106,
                    "minWidth": 98,
                    "sort": "desc",
                    "filter": "agNumberColumnFilter",
                    "tooltipField": "mpo_tooltip",
                    "headerClass": "ag-cat-hypothesis-col",
                },
            ],
        },
    ]
    if include_docking:
        groups.append(
            {
                "headerName": "3D-моделирование / Докинг",
                "headerClass": "ag-cat-hypothesis",
                "children": [
                    {
                        "headerName": "ΔG (ккал/моль)",
                        "field": "docking_dg_display",
                        "width": 134,
                        "minWidth": 120,
                        "filter": "agTextColumnFilter",
                        "headerTooltip": "Расчётная свободная энергия связывания (AutoDock Vina / RDKit-MMFF94-3D)",
                        "headerClass": "ag-cat-hypothesis-col",
                    },
                    {
                        "headerName": "PDB мишени",
                        "field": "docking_pdb_id",
                        "width": 98,
                        "minWidth": 90,
                        "filter": "agTextColumnFilter",
                        "headerTooltip": "Идентификатор 3D-структуры рецептора в RCSB PDB",
                        "headerClass": "ag-cat-hypothesis-col",
                    },
                ],
            }
        )
    return groups


class CandidateGrid:
    """Центральный нижний компонент: таблица кандидатов на базе виртуализированной сетки."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_candidate_selected: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_candidate_selected = on_candidate_selected
        self._grid: ui.aggrid | None = None
        self._title_label: ui.label | None = None
        self._telemetry_row: ui.row | None = None
        self._dock_btn: ui.button | None = None
        self._empty_overlay: ui.column | None = None
        self._build()

    def _build(self) -> None:
        vm = self.runtime.view_model
        initial_rows = vm.get_all_grid_rows()
        for r in initial_rows:
            vm.emitted_to_grid_ids.add(str(r["candidate_id"]))

        column_defs = build_candidate_column_defs(include_docking=True)

        grid_options: dict[str, Any] = {
            "columnDefs": column_defs,
            "rowData": initial_rows,
            "rowHeight": 104,
            "groupHeaderHeight": 30,
            "headerHeight": 34,
            "tooltipShowDelay": 150,
            "rowSelection": "single",
            "animateRows": False,
            "suppressCellFocus": False,
            "suppressNoRowsOverlay": True,
            "deferRender": True,
            ":getRowId": "(params) => params.data.candidate_id",
            "defaultColDef": {
                "sortable": True,
                "resizable": True,
            },
        }

        with ui.column().classes("w-full h-full flex-1 min-h-0 gap-1.5 px-2.5 pb-2 pt-1 bg-[#FAF8F9] text-[#1F181B]"):
            with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap shrink-0"):
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    self._title_label = ui.label(
                        f"Кандидаты и расчётные свойства ({len(vm.rows_by_id)})"
                    ).classes(
                        "text-xs font-semibold text-[#5C2D3B]"
                    )
                    self._telemetry_row = ui.row().classes("items-center gap-1 flex-wrap")

                with ui.row().classes("items-center gap-1.5"):
                    self._dock_btn = (
                        ui.button(on_click=self._run_manual_docking)
                        .props("dense unelevated no-caps size=sm color=primary")
                        .classes("text-[11px] px-2.5 gap-1")
                        .tooltip("Запустить физический 3D-докинг для топ-кандидатов без перезапуска пайплайна")
                    )
                    with self._dock_btn:
                        ui.html(svg_icon("play", 12), sanitize=False)
                        ui.label("Запустить 3D-докинг").classes("text-[11px] font-medium")

                    with ui.button(on_click=self._export_csv).props(
                        "dense outline no-caps size=sm color=grey-8"
                    ).classes("text-[11px] bg-white border-[#D5C8CD] px-2 gap-1").tooltip(
                        "Выгрузить таблицу кандидатов в формате CSV"
                    ):
                        ui.html(svg_icon("download", 13), sanitize=False)
                        ui.label(".csv").classes("font-mono text-[11px] text-[#261F22]")

                    with ui.button(on_click=self._export_sdf).props(
                        "dense outline no-caps size=sm color=grey-8"
                    ).classes("text-[11px] bg-white border-[#D5C8CD] px-2 gap-1").tooltip(
                        "Выгрузить структуры и свойства кандидатов в формате SDF"
                    ):
                        ui.html(svg_icon("download", 13), sanitize=False)
                        ui.label(".sdf").classes("font-mono text-[11px] text-[#261F22]")

            self.refresh_telemetry()
            with ui.card().classes(
                "relative w-full flex-1 min-h-0 bg-white border border-[#E2D9DD] rounded-lg p-0 overflow-hidden shadow-none"
            ):
                self._grid = ui.aggrid(
                    grid_options,
                    auto_size_columns=False,
                    theme="quartz",
                ).classes("w-full h-full")
                self._grid.on("cellClicked", self._handle_cell_clicked)

                # Приветственное пустое состояние с маскотом в полный рост (до появления кандидатов)
                self._empty_overlay = ui.column().classes(
                    "pointer-events-none absolute inset-0 z-10 items-center justify-center bg-white p-4 pb-32 gap-2"
                )
                with self._empty_overlay:
                    ui.html(
                        '<img src="/static/mascot_full.png" alt="Пишовичок" '
                        'style="height:134px;width:auto;object-fit:contain;display:block;" />',
                        sanitize=False,
                    )
                    ui.label("ПИШОВИЧОК готов к SAR-дизайну").classes(
                        "text-sm font-semibold text-[#5C2D3B]"
                    )
                    ui.label(
                        "Откройте готовый сценарий в шапке или опишите задачу в нижнем островке"
                    ).classes("text-xs text-[#64748B] text-center max-w-sm")
                self._empty_overlay.set_visibility(len(vm.rows_by_id) == 0)

    async def _run_manual_docking(self) -> None:
        from pishovichok.ui.bridge import agent_bridge

        cands = list(self.runtime.view_model.candidates_by_id.values())
        if not cands:
            ui.notify("Сначала сгенерируйте кандидатов", type="warning")
            return
        if self._dock_btn is not None:
            self._dock_btn.disable()
        ui.notify("Запущен расчёт 3D-докинга...", type="info")
        try:
            summary = await agent_bridge.run_standalone_docking(self.runtime.session_id)
            status = summary.get("status")
            if status == "SUCCESS":
                dg = summary.get("best_score_kcal")
                dg_txt = f" (лучший ΔG = {dg:.2f} ккал/моль)" if dg is not None else ""
                ui.notify(f"3D-докинг завершён{dg_txt}", type="positive")
            else:
                ui.notify(str(summary.get("message") or f"Статус докинга: {status}"), type="warning")
        except Exception as exc:
            ui.notify(f"Ошибка 3D-докинга: {exc}", type="negative")
        finally:
            if self._dock_btn is not None:
                self._dock_btn.enable()

    async def _export_csv(self) -> None:
        cands = list(self.runtime.view_model.candidates_by_id.values())
        if not cands:
            ui.notify("Нет кандидатов для выгрузки", type="warning")
            return
        target = settings.resolved_workspace_dir / self.runtime.session_id / "ui_export.csv"
        await run.io_bound(export_candidates_to_csv, cands, target)
        ui.download(target)

    async def _export_sdf(self) -> None:
        cands = list(self.runtime.view_model.candidates_by_id.values())
        if not cands:
            ui.notify("Нет кандидатов для выгрузки", type="warning")
            return
        target = settings.resolved_workspace_dir / self.runtime.session_id / "ui_export.sdf"
        await run.io_bound(export_candidates_to_sdf, cands, target)
        ui.download(target)

    def refresh_telemetry(self) -> None:
        """Обновляет заголовок счётчика и плашки параметров (скрыты в режиме «Синтетик»)."""
        count = len(self.runtime.view_model.rows_by_id)
        if self._title_label is not None:
            self._title_label.set_text(f"Кандидаты и расчётные свойства ({count})")
        if self._empty_overlay is not None:
            self._empty_overlay.set_visibility(count == 0)
        if self._dock_btn is not None:
            self._dock_btn.set_visibility(count > 0)
        if self._telemetry_row is None:
            return
        self._telemetry_row.clear()
        if self.runtime.depth_mode != "expert":
            self._telemetry_row.set_visibility(False)
            return
        self._telemetry_row.set_visibility(True)
        tc = resolve_tool_config(self.runtime.state)
        with self._telemetry_row:
            for label, tip in build_grid_telemetry_badges(tc):
                ui.label(label).classes(
                    "text-[10px] mono-code px-1.5 py-0.5"
                ).tooltip(tip)

    def _handle_cell_clicked(self, e: Any) -> None:
        args = getattr(e, "args", None)
        if isinstance(args, dict):
            data = args.get("data")
            if isinstance(data, dict):
                cid = data.get("candidate_id")
                if cid and self._on_candidate_selected is not None:
                    self._on_candidate_selected(str(cid))

    def reset_rows(self) -> None:
        """Очищает строки таблицы при новом запуске или перезапуске ветви."""
        self.refresh_telemetry()
        if self._grid is None:
            return
        self._grid.options["rowData"] = []
        self._grid.update()

    def push_new_rows(self) -> None:
        """Инкрементально добавляет новые строки в таблицу через `applyTransaction`."""
        self.refresh_telemetry()
        if self._grid is None:
            return
        new_rows = self.runtime.view_model.pop_new_grid_rows()
        if new_rows:
            self._grid.run_grid_method("applyTransaction", {"add": new_rows})

    def push_updated_rows(self, candidate_ids: list[str] | None = None) -> None:
        """Обновляет существующие строки без перерисовки DOM."""
        self.refresh_telemetry()
        if self._grid is None:
            return
        vm = self.runtime.view_model
        target_ids = candidate_ids if candidate_ids is not None else vm.row_order
        updates = [
            vm.rows_by_id[cid].to_aggrid_row(vm.session_id)
            for cid in target_ids
            if cid in vm.rows_by_id and cid in vm.emitted_to_grid_ids
        ]
        if updates:
            self._grid.run_grid_method("applyTransaction", {"update": updates})

    def select_row(self, candidate_id: str) -> None:
        """Выделяет строку в таблице при клике на точку графика."""
        if self._grid is None:
            return
        self._grid.run_row_method(candidate_id, "setSelected", True, True)
