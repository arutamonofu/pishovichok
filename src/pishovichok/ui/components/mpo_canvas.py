"""Интерактивный MPO-холст (Pareto pIC50 vs cLogP и PCA-химпространство) на `ui.echart`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import run, ui

from pishovichok.core.rendering import compute_chemical_space_projection
from pishovichok.ui.session import SessionRuntime


class MPOCanvas:
    """Центральный верхний компонент: интерактивный график и слайдеры весов/порогов MPO."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_candidate_selected: Callable[[str], Any] | None = None,
        on_mpo_recalculated: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_candidate_selected = on_candidate_selected
        self._on_mpo_recalculated = on_mpo_recalculated
        self._mode: str = "pareto"
        self._pca_dirty: bool = True
        self._chart: ui.echart | None = None
        self._mode_toggle: ui.toggle | None = None
        self._build()

    def _build(self) -> None:
        with ui.column().classes("w-full h-full gap-2 p-2 bg-slate-950"):
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label("Многопараметрическая оптимизация (MPO)").classes(
                    "text-xs uppercase tracking-wider font-semibold text-slate-400"
                )
                self._mode_toggle = ui.toggle(
                    {"pareto": "Парето (pIC50 vs cLogP)", "pca": "Химпространство (PCA)"},
                    value=self._mode,
                    on_change=self._handle_mode_change,
                ).props("dense no-caps toggle-color=emerald color=slate-800 size=sm")

            self._chart = ui.echart(
                options=self._build_chart_options(),
                on_point_click=self._handle_point_click,
            ).classes("w-full flex-1 min-h-[240px]")

            # Компактная панель весов и порогов MPO
            with ui.expansion("Веса и пороги MPO (векторный пересчёт)", icon="tune").props(
                "dense dark header-class='text-xs text-slate-300 bg-slate-900 rounded'"
            ).classes("w-full border border-slate-800 rounded"):
                w = self.runtime.view_model.weights
                with ui.grid(columns=3).classes("w-full gap-x-4 gap-y-1 p-2"):
                    self._make_slider("Вес pIC50", w.w_activity, 0.0, 5.0, 0.1, lambda v: setattr(w, "w_activity", v))
                    self._make_slider("Вес cLogP", w.w_logp, 0.0, 5.0, 0.1, lambda v: setattr(w, "w_logp", v))
                    self._make_slider("Вес SAscore", w.w_sa, 0.0, 5.0, 0.1, lambda v: setattr(w, "w_sa", v))
                    self._make_slider("Вес QED", w.w_qed, 0.0, 5.0, 0.1, lambda v: setattr(w, "w_qed", v))
                    self._make_slider("Вес hERG", w.w_herg, 0.0, 5.0, 0.1, lambda v: setattr(w, "w_herg", v))
                    self._make_slider("Max cLogP", w.max_logp, 1.0, 7.0, 0.1, lambda v: setattr(w, "max_logp", v))
                    self._make_slider("Max SA", w.max_sa_score, 2.0, 7.0, 0.1, lambda v: setattr(w, "max_sa_score", v))
                    self._make_slider("Max hERG", w.max_herg_prob, 0.1, 0.9, 0.05, lambda v: setattr(w, "max_herg_prob", v))
                    self._make_slider("Max MW", w.max_mw, 300.0, 800.0, 10.0, lambda v: setattr(w, "max_mw", v))

    def _make_slider(
        self,
        label: str,
        value: float,
        min_v: float,
        max_v: float,
        step: float,
        setter: Callable[[float], None],
    ) -> None:
        with ui.column().classes("w-full gap-0"):
            val_lbl = ui.label(f"{label}: {value:.2f}").classes(
                "text-[11px] text-slate-300"
            )

            def _on_change(e: Any) -> None:
                v = float(e.value)
                setter(v)
                val_lbl.set_text(f"{label}: {v:.2f}")
                self.runtime.view_model.recalculate_mpo()
                self.refresh(mark_pca_dirty=False)
                if self._on_mpo_recalculated is not None:
                    self._on_mpo_recalculated()

            ui.slider(
                min=min_v,
                max=max_v,
                step=step,
                value=value,
                on_change=_on_change,
            ).props("dense color=emerald")

    async def _handle_mode_change(self, e: Any) -> None:
        self._mode = str(e.value)
        if self._mode == "pca" and self._pca_dirty:
            await self._compute_pca_coords()
        self.refresh(mark_pca_dirty=False)

    async def _compute_pca_coords(self) -> None:
        vm = self.runtime.view_model
        smiles_list: list[str] = []
        targets: list[tuple[str, Any]] = []

        for ref in vm.reference_points:
            smiles_list.append(ref.canonical_smiles)
            targets.append(("ref", ref))
        for cid in vm.row_order:
            row = vm.rows_by_id.get(cid)
            if row is not None:
                smiles_list.append(row.canonical_smiles)
                targets.append(("row", row))

        if len(smiles_list) < 2:
            self._pca_dirty = False
            return

        coords = await run.cpu_bound(compute_chemical_space_projection, smiles_list)
        for (_, obj), (x, y) in zip(targets, coords, strict=False):
            obj.pca_x = float(x)
            obj.pca_y = float(y)
        self._pca_dirty = False

    def _handle_point_click(self, e: Any) -> None:
        data = getattr(e, "data", None)
        if isinstance(data, dict):
            cid = data.get("candidate_id")
            if cid and self._on_candidate_selected is not None:
                self._on_candidate_selected(str(cid))

    def _build_chart_options(self) -> dict[str, Any]:
        vm = self.runtime.view_model
        is_pca = self._mode == "pca"

        chembl_pts: list[dict[str, Any]] = []
        wetlab_pts: list[dict[str, Any]] = []
        lead_pts: list[dict[str, Any]] = []
        passed_pts: list[dict[str, Any]] = []
        failed_pts: list[dict[str, Any]] = []
        selected_pts: list[dict[str, Any]] = []

        for ref in vm.reference_points:
            x_val = ref.pca_x if is_pca else ref.clogp
            y_val = ref.pca_y if is_pca else ref.pchembl_value
            pt = {"value": [round(x_val, 3), round(y_val, 3)], "name": ref.mol_id}
            if ref.source_group == "wetlab":
                wetlab_pts.append(pt)
            elif ref.source_group == "lead":
                lead_pts.append(pt)
            else:
                chembl_pts.append(pt)

        if not lead_pts and vm.lead_reference is not None and not is_pca:
            lead = vm.lead_reference
            lead_pts.append(
                {
                    "value": [round(lead.calc_rdkit.clogp, 3), round(lead.pred_admet.predicted_pic50, 3)],
                    "name": f"Lead ({lead.parent_chembl_id})",
                    "candidate_id": lead.candidate_id,
                }
            )

        for cid in vm.row_order:
            row = vm.rows_by_id.get(cid)
            if row is None or row.excluded:
                continue
            x_val = row.pca_x if is_pca else row.clogp
            y_val = row.pca_y if is_pca else row.predicted_pic50
            pt = {
                "value": [round(x_val, 3), round(y_val, 3), round(row.mpo_score, 3)],
                "name": row.candidate_id,
                "candidate_id": row.candidate_id,
            }
            if cid == vm.selected_candidate_id:
                selected_pts.append(pt)
            elif row.passes_all and row.pains_count == 0 and row.brenk_count == 0:
                passed_pts.append(pt)
            else:
                failed_pts.append(pt)

        return {
            "backgroundColor": "transparent",
            "tooltip": {
                "trigger": "item",
                "backgroundColor": "#0f172a",
                "borderColor": "#334155",
                "textStyle": {"color": "#e2e8f0", "fontSize": 12},
            },
            "legend": {
                "top": 4,
                "textStyle": {"color": "#94a3b8", "fontSize": 11},
            },
            "grid": {"left": 48, "right": 20, "top": 36, "bottom": 38},
            "xAxis": {
                "type": "value",
                "name": "PC1" if is_pca else "cLogP",
                "nameLocation": "middle",
                "nameGap": 22,
                "nameTextStyle": {"color": "#94a3b8", "fontSize": 11},
                "axisLabel": {"color": "#94a3b8", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#1e293b"}},
            },
            "yAxis": {
                "type": "value",
                "name": "PC2" if is_pca else "pIC50",
                "nameTextStyle": {"color": "#94a3b8", "fontSize": 11},
                "axisLabel": {"color": "#94a3b8", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#1e293b"}},
            },
            "series": [
                {
                    "name": "ChEMBL",
                    "type": "scatter",
                    "symbolSize": 7,
                    "itemStyle": {"color": "rgba(148, 163, 184, 0.45)"},
                    "data": chembl_pts,
                },
                {
                    "name": "Wet-Lab",
                    "type": "scatter",
                    "symbolSize": 8,
                    "itemStyle": {"color": "#38bdf8"},
                    "data": wetlab_pts,
                },
                {
                    "name": "Lead",
                    "type": "scatter",
                    "symbol": "diamond",
                    "symbolSize": 14,
                    "itemStyle": {"color": "#f59e0b"},
                    "data": lead_pts,
                },
                {
                    "name": "Кандидаты (OK)",
                    "type": "scatter",
                    "symbolSize": 10,
                    "itemStyle": {"color": "#10b981"},
                    "data": passed_pts,
                },
                {
                    "name": "С нарушениями",
                    "type": "scatter",
                    "symbolSize": 8,
                    "itemStyle": {"color": "rgba(244, 63, 94, 0.7)"},
                    "data": failed_pts,
                },
                {
                    "name": "Выбранный",
                    "type": "scatter",
                    "symbol": "circle",
                    "symbolSize": 16,
                    "itemStyle": {
                        "color": "#fde047",
                        "borderColor": "#ffffff",
                        "borderWidth": 2,
                    },
                    "data": selected_pts,
                },
            ],
        }

    def refresh(self, *, mark_pca_dirty: bool = True) -> None:
        """Обновляет опции графика без пересоздания DOM-элемента."""
        if mark_pca_dirty:
            self._pca_dirty = True
        if self._chart is not None:
            self._chart.options.update(self._build_chart_options())
            self._chart.update()
