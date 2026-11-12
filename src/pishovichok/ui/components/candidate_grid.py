"""Виртуализированная таблица кандидатов `ui.aggrid` с ленивой загрузкой SVG и `applyTransaction`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from pishovichok.ui.session import SessionRuntime


class CandidateGrid:
    """Центральный нижний компонент: таблица кандидатов на базе `ui.aggrid`."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_candidate_selected: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_candidate_selected = on_candidate_selected
        self._grid: ui.aggrid | None = None
        self._build()

    def _build(self) -> None:
        vm = self.runtime.view_model
        initial_rows = vm.get_all_grid_rows()
        for r in initial_rows:
            vm.emitted_to_grid_ids.add(str(r["candidate_id"]))

        column_defs = [
            {
                "headerName": "2D Diff",
                "field": "preview_html",
                "width": 120,
                "minWidth": 110,
                "sortable": False,
                "filter": False,
                "resizable": True,
            },
            {
                "headerName": "ID",
                "field": "candidate_id",
                "width": 130,
                "filter": "agTextColumnFilter",
            },
            {
                "headerName": "MPO",
                "field": "mpo_score",
                "width": 85,
                "sort": "desc",
                "filter": "agNumberColumnFilter",
            },
            {
                "headerName": "pIC50",
                "field": "predicted_pic50",
                "width": 85,
                "filter": "agNumberColumnFilter",
            },
            {
                "headerName": "ΔpIC50",
                "field": "delta_activity",
                "width": 85,
                "filter": "agNumberColumnFilter",
            },
            {
                "headerName": "cLogP",
                "field": "clogp",
                "width": 80,
                "filter": "agNumberColumnFilter",
            },
            {
                "headerName": "ΔcLogP",
                "field": "delta_clogp",
                "width": 85,
            },
            {
                "headerName": "TPSA",
                "field": "tpsa",
                "width": 78,
            },
            {
                "headerName": "SA",
                "field": "sa_score",
                "width": 72,
            },
            {
                "headerName": "hERG",
                "field": "herg_prob",
                "width": 78,
            },
            {
                "headerName": "Статус",
                "field": "critic_status",
                "width": 115,
                "filter": "agTextColumnFilter",
            },
            {
                "headerName": "Правило",
                "field": "rule_id",
                "width": 110,
                "filter": "agTextColumnFilter",
            },
        ]

        grid_options: dict[str, Any] = {
            "columnDefs": column_defs,
            "rowData": initial_rows,
            "rowHeight": 62,
            "headerHeight": 34,
            "rowSelection": "single",
            "animateRows": False,
            "suppressCellFocus": False,
            "deferRender": True,
            ":getRowId": "(params) => params.data.candidate_id",
            "defaultColDef": {
                "sortable": True,
                "resizable": True,
            },
        }

        with ui.column().classes("w-full h-full gap-1 p-2 bg-slate-950"):
            ui.label("Кандидаты и дескрипторы (AG Grid)").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400"
            )
            self._grid = ui.aggrid(
                grid_options,
                html_columns=[0],
                theme="quartz",
            ).classes("w-full flex-1 h-full min-h-[220px]")
            self._grid.on("cellClicked", self._handle_cell_clicked)

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
        if self._grid is None:
            return
        self._grid.options["rowData"] = []
        self._grid.update()

    def push_new_rows(self) -> None:
        """Инкрементально добавляет новые строки в таблицу через `applyTransaction`."""
        if self._grid is None:
            return
        new_rows = self.runtime.view_model.pop_new_grid_rows()
        if new_rows:
            self._grid.run_grid_method("applyTransaction", {"add": new_rows})

    def push_updated_rows(self, candidate_ids: list[str] | None = None) -> None:
        """Обновляет существующие строки (после пересчёта MPO или Pin/Exclude) без перерисовки DOM."""
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
