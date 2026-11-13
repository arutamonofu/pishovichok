"""Интерактивный MPO-холст (Профиль Парето, PCA, SALI) (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import run, ui

from pishovichok.core.rendering import (
    compute_chemical_space_projection,
    render_molecule_svg,
)
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.ui.components.agent_rail import render_sar_rules_panel
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import SessionRuntime


class MPOCanvas:
    """Центральный верхний компонент: компактный график 250px, поповер порогов/весов и панель правил SAR."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_candidate_selected: Callable[[str], Any] | None = None,
        on_mpo_recalculated: Callable[[], Any] | None = None,
        on_tool_config_changed: Callable[[], Any] | None = None,
        on_scaffold_selected: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_candidate_selected = on_candidate_selected
        self._on_mpo_recalculated = on_mpo_recalculated
        self._on_tool_config_changed = on_tool_config_changed
        self._on_scaffold_selected = on_scaffold_selected
        self._mode: str = "pareto"
        self._pca_dirty: bool = True
        self._chart: ui.echart | None = None
        self._mode_toggle: ui.toggle | None = None
        self._scaffolds_section: ui.column | None = None
        self._scaffolds_box: ui.row | None = None
        self._sar_header_label: ui.label | None = None
        self._sar_box: ui.element | None = None
        self._sar_rules_pane: ui.column | None = None
        self._build()

    def _build(self) -> None:
        with ui.column().classes("w-full h-full gap-1.5 px-2.5 pt-2 pb-1 bg-[#FAF8F9] text-[#1F181B] flex flex-col overflow-hidden"):
            with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap shrink-0"):
                ui.label("Многопараметрическая оптимизация (MPO)").classes(
                    "text-xs font-semibold text-[#5C2D3B]"
                )
                self._mode_toggle = (
                    ui.toggle(
                        {
                            "pareto": "Профиль Парето",
                            "pca": "Химпространство (PCA)",
                            "sali": "Перепады активности (SALI)",
                            "sar_rules": "Правила SAR",
                        },
                        value=self._mode,
                        on_change=self._handle_mode_change,
                    )
                    .props("dense no-caps toggle-color=primary color=grey-2 text-color=grey-9 size=sm")
                    .classes("border border-[#D5C8CD] rounded compact-toggle")
                )

            with ui.card().classes(
                "w-full flex-1 min-h-0 bg-white border border-[#E2E8F0] rounded-lg p-1.5 shadow-none overflow-hidden flex flex-col"
            ):
                self._chart = ui.echart(
                    options=self._build_chart_options(),
                    on_point_click=self._handle_point_click,
                ).classes("w-full h-full")

                self._sar_rules_pane = ui.column().classes(
                    "w-full h-full overflow-y-auto wetlab-scroll p-1.5 gap-2"
                )
                with self._sar_rules_pane:
                    with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap shrink-0"):
                        self._sar_header_label = ui.label("Правила SAR (MMP)").classes(
                            "text-xs font-semibold text-[#5C2D3B]"
                        )
                        self._scaffolds_section = ui.row().classes("items-center gap-1.5 flex-wrap")
                        with self._scaffolds_section:
                            ui.label("Базовые скаффолды (Bemis–Murcko):").classes(
                                "text-[11px] font-medium text-[#594D52]"
                            )
                            self._scaffolds_box = ui.row().classes("items-center gap-1 flex-wrap")

                    self._sar_box = ui.element("div").classes(
                        "w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2"
                    )
                self._sar_rules_pane.set_visibility(False)

            self._refresh_sar_panel()

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
        if vm.lead_reference is not None:
            smiles_list.append(vm.lead_reference.canonical_smiles)
            targets.append(("lead", vm))

        if len(smiles_list) < 2:
            self._pca_dirty = False
            return

        coords = await run.cpu_bound(
            compute_chemical_space_projection,
            smiles_list,
            tool_config=resolve_tool_config(self.runtime.state),
        )
        for (kind, obj), (x, y) in zip(targets, coords, strict=False):
            if kind == "lead":
                obj.lead_pca_coords = (float(x), float(y))
            else:
                obj.pca_x = float(x)
                obj.pca_y = float(y)
        self._pca_dirty = False

    def _open_reference_point_dialog(self, data: dict[str, Any]) -> None:
        """Открывает карточку референсного соединения или пары перепада активности при клике по точке."""
        kind = str(data.get("kind") or "")
        if kind in ("sali_cliff", "mmp_pair"):
            mol_a = str(data.get("mol_id_active") or "")
            mol_b = str(data.get("mol_id_inactive") or "")
            smirks = str(data.get("smirks") or "")
            delta_v = data.get("abs_delta_pic50")
            sim_v = data.get("tanimoto")
            sali_v = data.get("sali_index")
            with ui.dialog() as dlg, ui.card().classes(
                "w-[460px] max-w-[92vw] bg-white border border-[#CBD5E1] rounded-lg p-3.5 gap-2 shadow-lg"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(f"Пара MMP: {mol_a} ↔ {mol_b}").classes(
                        "text-xs font-semibold text-[#0F172A]"
                    )
                    with ui.button(on_click=dlg.close).props("dense flat size=xs color=grey-8"):
                        ui.html(svg_icon("close", 14), sanitize=False)
                with ui.grid(columns=3).classes("w-full gap-1.5"):
                    for lbl, val in (
                        ("|ΔpIC50|", f"{float(delta_v):.2f}" if delta_v is not None else "—"),
                        ("Tanimoto", f"{float(sim_v):.3f}" if sim_v is not None else "—"),
                        ("SALI", f"{float(sali_v):.1f}" if sali_v is not None else "—"),
                    ):
                        with ui.column().classes("bg-[#F8FAFC] border border-[#E2E8F0] rounded p-1.5 gap-0"):
                            ui.label(lbl).classes("text-[10px] text-[#475569]")
                            ui.label(val).classes("text-xs font-mono font-semibold text-[#0F172A]")
                if smirks:
                    ui.label(smirks).classes("text-[11px] mono-code px-2 py-1 break-all w-full")
            dlg.open()
            return

        mol_id = str(data.get("mol_id") or data.get("name") or "Соединение")
        smiles = str(data.get("canonical_smiles") or "")
        pic50_v = data.get("pic50")
        clogp_v = data.get("clogp")
        svg_markup = (
            render_molecule_svg(smiles, width=380, height=210, legend=mol_id)
            if smiles
            else ""
        )
        with ui.dialog() as dlg, ui.card().classes(
            "w-[440px] max-w-[92vw] bg-white border border-[#CBD5E1] rounded-lg p-3.5 gap-2 shadow-lg"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"Референсное соединение · {mol_id}").classes(
                    "text-xs font-semibold text-[#0F172A]"
                )
                with ui.button(on_click=dlg.close).props("dense flat size=xs color=grey-8"):
                    ui.html(svg_icon("close", 14), sanitize=False)
            if svg_markup:
                ui.html(svg_markup, sanitize=False).classes(
                    "w-full flex justify-center bg-white border border-[#E2E8F0] rounded p-1"
                )
            with ui.row().classes("w-full gap-2"):
                if pic50_v is not None:
                    ui.label(f"pIC50: {float(pic50_v):.2f}").classes(
                        "text-xs font-mono bg-[#F5F3FF] text-[#4C1D95] px-2 py-0.5 rounded"
                    )
                if clogp_v is not None:
                    ui.label(f"cLogP: {float(clogp_v):.2f}").classes(
                        "text-xs font-mono bg-[#ECFDF5] text-[#065F46] px-2 py-0.5 rounded"
                    )
            if smiles:
                ui.label(smiles).classes("text-[11px] mono-code px-2 py-1 break-all w-full")
        dlg.open()

    def _handle_point_click(self, e: Any) -> None:
        data = getattr(e, "data", None)
        if not isinstance(data, dict):
            return
        cid = data.get("candidate_id")
        if cid and self._on_candidate_selected is not None:
            self._on_candidate_selected(str(cid))
            return

        kind = str(data.get("kind") or "")
        mol_id = str(data.get("mol_id") or "")
        if kind == "lead" and mol_id and self._on_candidate_selected is not None:
            # Находим первого допустимого кандидата от этого прототипа
            vm = self.runtime.view_model
            for cand_id in vm.row_order:
                row = vm.rows_by_id.get(cand_id)
                if row is not None and not row.excluded and row.parent_chembl_id == mol_id:
                    self._on_candidate_selected(cand_id)
                    ui.notify(f"Прототип {mol_id} → выбран кандидат {cand_id}", type="info")
                    return

        self._open_reference_point_dialog(data)

    def _build_sali_chart_options(self) -> dict[str, Any]:
        """Строит опции графика перепадов активности (SALI): Сходство vs |ΔpIC50|."""
        tc = resolve_tool_config(self.runtime.state)
        cliffs = list(self.runtime.state.get("detected_activity_cliffs", []))
        mmp_pairs = list(getattr(self.runtime.view_model, "mmp_pairs_preview", []))

        sim_cutoff = float(tc.cliff_similarity_cutoff)
        delta_cutoff = float(tc.cliff_delta_pic50_cutoff)

        bg_pairs: list[dict[str, Any]] = []
        for p in mmp_pairs:
            if p.get("is_cliff"):
                continue
            bg_pairs.append(
                {
                    "value": [float(p.get("tanimoto", 0.0)), float(p.get("abs_delta_pic50", 0.0)), float(p.get("sali_index", 0.0))],
                    "name": f"{p.get('mol_id_lhs', '')} ↔ {p.get('mol_id_rhs', '')} (SALI {p.get('sali_index', 0):.1f})",
                    "kind": "mmp_pair",
                    "mol_id_active": str(p.get("mol_id_lhs", "")),
                    "mol_id_inactive": str(p.get("mol_id_rhs", "")),
                    "abs_delta_pic50": float(p.get("abs_delta_pic50", 0.0)),
                    "tanimoto": float(p.get("tanimoto", 0.0)),
                    "sali_index": float(p.get("sali_index", 0.0)),
                    "smirks": str(p.get("smirks", "")),
                }
            )

        cliff_pts: list[dict[str, Any]] = []
        for cl in cliffs:
            sali_val = float(getattr(cl, "sali_index", 1.0))
            sz = max(12, min(24, 10 + int(sali_val ** 0.5 * 2.0)))
            cliff_pts.append(
                {
                    "value": [
                        round(float(cl.tanimoto_similarity), 3),
                        round(float(cl.abs_delta_pic50), 2),
                        round(sali_val, 1),
                    ],
                    "symbolSize": sz,
                    "name": f"{cl.cliff_id}: {cl.mol_id_active} / {cl.mol_id_inactive} (SALI={sali_val:.1f}, {cl.smirks_difference})",
                    "kind": "sali_cliff",
                    "cliff_id": str(cl.cliff_id),
                    "mol_id_active": str(cl.mol_id_active),
                    "mol_id_inactive": str(cl.mol_id_inactive),
                    "abs_delta_pic50": float(cl.abs_delta_pic50),
                    "tanimoto": float(cl.tanimoto_similarity),
                    "sali_index": sali_val,
                    "smirks": str(cl.smirks_difference),
                }
            )

        return {
            "backgroundColor": "#FFFFFF",
            "tooltip": {
                "trigger": "item",
                "backgroundColor": "#FFFFFF",
                "borderColor": "#CBD5E1",
                "textStyle": {"color": "#0F172A", "fontSize": 12},
            },
            "legend": {
                "orient": "vertical",
                "right": 16,
                "top": "middle",
                "itemGap": 10,
                "textStyle": {"color": "#475569", "fontSize": 11},
            },
            "grid": {"left": 54, "right": 200, "top": 18, "bottom": 38},
            "xAxis": {
                "type": "value",
                "name": "Сходство (Tanimoto)",
                "min": 0.0,
                "max": 1.02,
                "nameLocation": "middle",
                "nameGap": 20,
                "nameTextStyle": {"color": "#475569", "fontSize": 11},
                "axisLabel": {"color": "#475569", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
            },
            "yAxis": {
                "type": "value",
                "name": "|ΔpIC50|",
                "nameLocation": "middle",
                "nameGap": 34,
                "nameTextStyle": {"color": "#475569", "fontSize": 11},
                "axisLabel": {"color": "#475569", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
            },
            "series": [
                {
                    "name": "Пары MMP",
                    "type": "scatter",
                    "symbolSize": 8,
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.35},
                    "itemStyle": {"color": "#94A3B8", "opacity": 0.45},
                    "data": bg_pairs,
                    "z": 2,
                },
                {
                    "name": "Перепады активности (SALI)",
                    "type": "scatter",
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.3},
                    "itemStyle": {
                        "color": "#7C3AED",
                        "borderColor": "#FFFFFF",
                        "borderWidth": 1.5,
                    },
                    "data": cliff_pts,
                    "z": 6,
                    "markLine": {
                        "silent": True,
                        "symbol": "none",
                        "lineStyle": {"type": "dashed", "color": "#E11D48", "width": 1.2},
                        "data": [
                            {
                                "xAxis": sim_cutoff,
                                "label": {
                                    "formatter": f"S ≥ {sim_cutoff:.2f}",
                                    "position": "insideEndTop",
                                    "rotate": 90,
                                    "distance": 8,
                                    "color": "#9F1239",
                                    "fontSize": 10,
                                },
                            },
                            {
                                "yAxis": delta_cutoff,
                                "label": {
                                    "formatter": f"|Δ| ≥ {delta_cutoff:.1f}",
                                    "position": "insideEndTop",
                                    "color": "#9F1239",
                                    "fontSize": 10,
                                },
                            },
                        ],
                    },
                    "markArea": {
                        "silent": True,
                        "itemStyle": {"color": "rgba(255, 241, 242, 0.55)"},
                        "data": [
                            [
                                {"xAxis": 0.0, "yAxis": 0.0},
                                {"xAxis": sim_cutoff, "yAxis": delta_cutoff},
                            ]
                        ],
                    },
                },
            ],
        }

    def _build_chart_options(self) -> dict[str, Any]:
        if self._mode == "sali":
            return self._build_sali_chart_options()

        vm = self.runtime.view_model
        is_pca = self._mode == "pca"

        chembl_pts: list[dict[str, Any]] = []
        wetlab_pts: list[dict[str, Any]] = []
        lead_pts: list[dict[str, Any]] = []
        passed_pts: list[dict[str, Any]] = []
        failed_pts: list[dict[str, Any]] = []
        selected_pts: list[dict[str, Any]] = []

        lead_ids: set[str] = set()
        lead_smiles_set: set[str] = set()
        if vm.lead_reference is not None:
            lead_ids.add(vm.lead_reference.parent_chembl_id)
            lead_smiles_set.add(vm.lead_reference.canonical_smiles)
        for cand in vm.candidates_by_id.values():
            lead_ids.add(cand.parent_chembl_id)
            lead_smiles_set.add(cand.parent_smiles)

        for ref in vm.reference_points:
            x_val = ref.pca_x if is_pca else ref.clogp
            y_val = ref.pca_y if is_pca else ref.pchembl_value
            is_lead_ref = (
                ref.source_group == "lead"
                or ref.mol_id in lead_ids
                or ref.canonical_smiles in lead_smiles_set
            )
            pt = {
                "value": [round(x_val, 3), round(y_val, 3)],
                "name": f"Прототип ({ref.mol_id})" if is_lead_ref else ref.mol_id,
                "kind": "lead" if is_lead_ref else ref.source_group,
                "mol_id": ref.mol_id,
                "canonical_smiles": ref.canonical_smiles,
                "pic50": round(float(ref.pchembl_value), 2),
                "clogp": round(float(ref.clogp), 2),
            }
            if is_lead_ref:
                lead_pts.append(pt)
            elif ref.source_group == "wetlab":
                wetlab_pts.append(pt)
            else:
                chembl_pts.append(pt)

        if not lead_pts and vm.lead_reference is not None:
            lead = vm.lead_reference
            if not is_pca:
                lead_pts.append(
                    {
                        "value": [
                            round(lead.calc_rdkit.clogp, 3),
                            round(lead.pred_admet.predicted_pic50, 3),
                        ],
                        "name": f"Прототип ({lead.parent_chembl_id})",
                        "kind": "lead",
                        "mol_id": lead.parent_chembl_id,
                        "canonical_smiles": lead.canonical_smiles,
                        "candidate_id": lead.candidate_id,
                    }
                )
            elif vm.lead_pca_coords is not None:
                lead_pts.append(
                    {
                        "value": [
                            round(vm.lead_pca_coords[0], 3),
                            round(vm.lead_pca_coords[1], 3),
                        ],
                        "name": f"Прототип ({lead.parent_chembl_id})",
                        "kind": "lead",
                        "mol_id": lead.parent_chembl_id,
                        "canonical_smiles": lead.canonical_smiles,
                        "candidate_id": lead.candidate_id,
                    }
                )
        elif not lead_pts and vm.candidates_by_id and not is_pca:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            seen_parents: set[str] = set()
            for cand in vm.candidates_by_id.values():
                if cand.parent_chembl_id in seen_parents:
                    continue
                seen_parents.add(cand.parent_chembl_id)
                pmol = Chem.MolFromSmiles(cand.parent_smiles)
                if pmol is None:
                    continue
                clogp_p = float(Descriptors.MolLogP(pmol))
                lead_pts.append(
                    {
                        "value": [
                            round(clogp_p, 3),
                            round(float(cand.parent_exp_pic50), 3),
                        ],
                        "name": f"Прототип ({cand.parent_chembl_id})",
                        "kind": "lead",
                        "mol_id": cand.parent_chembl_id,
                        "canonical_smiles": cand.parent_smiles,
                        "pic50": round(float(cand.parent_exp_pic50), 2),
                        "clogp": round(clogp_p, 2),
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

        pareto_mark_lines: list[dict[str, Any]] = []
        pareto_mark_areas: list[list[dict[str, Any]]] = []
        if not is_pca:
            constraints = self.runtime.state.get("query_constraints")
            max_logp = float(
                vm.weights.max_logp
                if vm.weights is not None
                else (getattr(constraints, "max_logp", 4.5) if constraints else 4.5)
            )
            min_logp = (
                vm.weights.min_logp
                if vm.weights is not None
                else getattr(constraints, "min_logp", None)
            )
            pareto_mark_lines.append(
                {
                    "xAxis": max_logp,
                    "label": {
                        "formatter": f"max cLogP={max_logp:.1f}",
                        "position": "insideEndTop",
                        "rotate": 90,
                        "distance": 8,
                        "color": "#9F1239",
                        "fontSize": 10,
                    },
                }
            )
            pareto_mark_areas.append(
                [
                    {"xAxis": max_logp},
                    {"xAxis": 99.0},
                ]
            )
            if min_logp is not None:
                pareto_mark_lines.append(
                    {
                        "xAxis": float(min_logp),
                        "label": {
                            "formatter": f"min cLogP={float(min_logp):.1f}",
                            "position": "insideEndTop",
                            "rotate": 90,
                            "distance": 8,
                            "color": "#9F1239",
                            "fontSize": 10,
                        },
                    }
                )
                pareto_mark_areas.append(
                    [
                        {"xAxis": -99.0},
                        {"xAxis": float(min_logp)},
                    ]
                )
            parent_pic50s = [
                c.parent_exp_pic50 for c in vm.candidates_by_id.values()
            ]
            if parent_pic50s:
                baseline_pic50 = round(min(parent_pic50s), 2)
                pareto_mark_lines.append(
                    {
                        "yAxis": baseline_pic50,
                        "label": {
                            "formatter": f"Прототип pIC50={baseline_pic50:.2f}",
                            "position": "insideEndTop",
                            "color": "#4C1D95",
                            "fontSize": 10,
                        },
                    }
                )

        passed_series: dict[str, Any] = {
            "name": "Кандидаты (допустимые)",
            "type": "scatter",
            "symbolSize": 11,
            "cursor": "pointer",
            "emphasis": {"scale": 1.35},
            "itemStyle": {"color": "#10B981"},
            "data": passed_pts,
            "z": 6,
        }
        if not is_pca and pareto_mark_lines:
            passed_series["markLine"] = {
                "silent": True,
                "symbol": "none",
                "lineStyle": {"type": "dashed", "color": "#E11D48", "width": 1.3},
                "data": pareto_mark_lines,
            }
            passed_series["markArea"] = {
                "silent": True,
                "itemStyle": {"color": "rgba(255, 241, 242, 0.72)"},
                "data": pareto_mark_areas,
            }
        else:
            passed_series["markLine"] = {"silent": True, "data": []}
            passed_series["markArea"] = {"silent": True, "data": []}

        return {
            "backgroundColor": "#FFFFFF",
            "tooltip": {
                "trigger": "item",
                "backgroundColor": "#FFFFFF",
                "borderColor": "#CBD5E1",
                "textStyle": {"color": "#0F172A", "fontSize": 12},
            },
            "legend": {
                "orient": "vertical",
                "right": 16,
                "top": "middle",
                "itemGap": 10,
                "textStyle": {"color": "#475569", "fontSize": 11},
            },
            "grid": {"left": 54, "right": 235, "top": 18, "bottom": 38},
            "xAxis": {
                "type": "value",
                "name": "PC1" if is_pca else "cLogP",
                "nameLocation": "middle",
                "nameGap": 20,
                "nameTextStyle": {"color": "#475569", "fontSize": 11},
                "axisLabel": {"color": "#475569", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
            },
            "yAxis": {
                "type": "value",
                "name": "PC2" if is_pca else "pIC50",
                "nameLocation": "middle",
                "nameGap": 34,
                "nameTextStyle": {"color": "#475569", "fontSize": 11},
                "axisLabel": {"color": "#475569", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
            },
            "series": [
                {
                    "name": "ChEMBL",
                    "type": "scatter",
                    "symbolSize": 8,
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.35},
                    "itemStyle": {"color": "#94A3B8", "opacity": 0.45},
                    "data": chembl_pts,
                    "z": 2,
                },
                {
                    "name": "Лаб. серия",
                    "type": "scatter",
                    "symbolSize": 9,
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.35},
                    "itemStyle": {"color": "#7C3AED"},
                    "data": wetlab_pts,
                    "z": 4,
                },
                {
                    "name": "Прототип",
                    "type": "scatter",
                    "symbol": "diamond",
                    "symbolSize": 15,
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.3},
                    "itemStyle": {"color": "#D97706"},
                    "data": lead_pts,
                    "z": 8,
                },
                passed_series,
                {
                    "name": "С нарушениями",
                    "type": "scatter",
                    "symbolSize": 9,
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.35},
                    "itemStyle": {"color": "rgba(239, 68, 68, 0.72)"},
                    "data": failed_pts,
                    "z": 5,
                },
                {
                    "name": "Выбранный",
                    "type": "scatter",
                    "symbol": "circle",
                    "symbolSize": 17,
                    "cursor": "pointer",
                    "emphasis": {"scale": 1.25},
                    "itemStyle": {
                        "color": "#8B495C",
                        "borderColor": "#FFFFFF",
                        "borderWidth": 2.5,
                        "shadowBlur": 8,
                        "shadowColor": "rgba(139, 73, 92, 0.72)",
                    },
                    "data": selected_pts,
                    "z": 10,
                },
            ],
        }

    def _refresh_sar_panel(self) -> None:
        if getattr(self, "_sar_box", None) is None:
            return
        render_sar_rules_panel(
            self.runtime,
            scaffolds_section=self._scaffolds_section,
            scaffolds_box=self._scaffolds_box,
            sar_header_label=self._sar_header_label,
            sar_box=self._sar_box,
            on_scaffold_selected=self._on_scaffold_selected,
        )

    def refresh(self, *, mark_pca_dirty: bool = True) -> None:
        """Обновляет опции графика и панель правил SAR без пересоздания DOM-элемента."""
        if mark_pca_dirty:
            self._pca_dirty = True
        is_sar_tab = self._mode == "sar_rules"
        if self._chart is not None:
            self._chart.set_visibility(not is_sar_tab)
            if not is_sar_tab:
                self._chart.options.update(self._build_chart_options())
                self._chart.update()
        if getattr(self, "_sar_rules_pane", None) is not None:
            self._sar_rules_pane.set_visibility(is_sar_tab)
        self._refresh_sar_panel()
