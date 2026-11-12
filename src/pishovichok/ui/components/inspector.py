"""Правый инспектор кандидата: 2D diff, 3D ETKDGv3, Desirability, Provenance и экспорт."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from nicegui import run, ui

from pishovichok.agents.workflow import (
    export_candidates_to_csv,
    export_candidates_to_sdf,
)
from pishovichok.config import settings
from pishovichok.core.rendering import generate_3d_conformer_molblock
from pishovichok.ui.components.chem_widgets import MoleculeViewer3D
from pishovichok.ui.session import SessionRuntime


class InspectorPane:
    """Правая панель Chemical IDE: детальный разбор выбранной молекулы и экспорт."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_row_updated: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_row_updated = on_row_updated
        self._header_title: ui.label | None = None
        self._smiles_label: ui.label | None = None
        self._pin_btn: ui.button | None = None
        self._exclude_btn: ui.button | None = None
        self._pairwise_box: ui.row | None = None
        self._viewer3d: MoleculeViewer3D | None = None
        self._desirability_box: ui.column | None = None
        self._provenance_box: ui.column | None = None
        self._mol3d_cache: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        with ui.column().classes("w-full h-full gap-3 p-3 overflow-y-auto bg-slate-950"):
            with ui.row().classes("w-full items-center justify-between gap-2"):
                self._header_title = ui.label("Инспектор кандидата").classes(
                    "text-xs uppercase tracking-wider font-semibold text-slate-400"
                )
                with ui.row().classes("items-center gap-1"):
                    self._pin_btn = ui.button(
                        "Pin",
                        icon="push_pin",
                        on_click=self._toggle_pin,
                    ).props("dense flat size=sm color=amber").classes("text-xs")
                    self._exclude_btn = ui.button(
                        "Exclude",
                        icon="block",
                        on_click=self._toggle_exclude,
                    ).props("dense flat size=sm color=rose").classes("text-xs")

            self._smiles_label = ui.label("").classes(
                "text-[11px] font-mono text-slate-400 break-all bg-slate-900 p-2 rounded border border-slate-800"
            )

            # Попарное 2D-сравнение (Parent -> SMIRKS -> Candidate) через ленивые HTTP-URL
            ui.label("Попарный дифф (Parent → Candidate)").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400"
            )
            self._pairwise_box = ui.row().classes(
                "w-full items-center justify-between gap-2 no-wrap"
            )

            # 3D-конформер ETKDGv3
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("3D-конформер (ETKDGv3 + UFF)").classes(
                    "text-xs uppercase tracking-wider font-semibold text-slate-400"
                )
                ui.button(
                    "Построить 3D",
                    icon="view_in_ar",
                    on_click=self._load_3d_conformer,
                ).props("dense outline size=sm color=sky").classes("text-xs")

            self._viewer3d = MoleculeViewer3D(height="240px")

            # Профиль желательности d1..d5
            ui.label("Профиль желательности (Harrington MPO)").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400"
            )
            self._desirability_box = ui.column().classes("w-full gap-1.5")

            # 4-категорийный Provenance
            ui.label("Provenance (4 категории)").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400"
            )
            self._provenance_box = ui.column().classes("w-full gap-2")

            # Экспорт артефактов
            ui.label("Экспорт артефактов").classes(
                "text-xs uppercase tracking-wider font-semibold text-slate-400 mt-1"
            )
            with ui.row().classes("w-full gap-2 flex-wrap"):
                ui.button("CSV", icon="download", on_click=self._export_csv).props(
                    "dense outline size=sm color=emerald"
                ).classes("text-xs")
                ui.button("SDF", icon="download", on_click=self._export_sdf).props(
                    "dense outline size=sm color=emerald"
                ).classes("text-xs")
                ui.button("Trace JSON", icon="data_object", on_click=self._export_trace).props(
                    "dense outline size=sm color=sky"
                ).classes("text-xs")
                ui.button("Report MD", icon="description", on_click=self._export_report).props(
                    "dense outline size=sm color=sky"
                ).classes("text-xs")

        self.refresh()

    def _get_selected(self) -> tuple[str | None, Any, Any]:
        vm = self.runtime.view_model
        cid = vm.selected_candidate_id
        if not cid and vm.row_order:
            cid = vm.row_order[0]
            vm.selected_candidate_id = cid
        if not cid:
            return None, None, None
        return cid, vm.candidates_by_id.get(cid), vm.rows_by_id.get(cid)

    def _toggle_pin(self) -> None:
        cid, _, row = self._get_selected()
        if not cid or row is None:
            return
        self.runtime.view_model.set_pinned(cid, not row.pinned)
        self.refresh()
        if self._on_row_updated is not None:
            self._on_row_updated(cid)

    def _toggle_exclude(self) -> None:
        cid, _, row = self._get_selected()
        if not cid or row is None:
            return
        self.runtime.view_model.set_excluded(cid, not row.excluded)
        self.refresh()
        if self._on_row_updated is not None:
            self._on_row_updated(cid)

    async def _load_3d_conformer(self) -> None:
        cid, cand, _ = self._get_selected()
        if cand is None or self._viewer3d is None:
            ui.notify("Сначала выберите кандидата", type="warning")
            return
        molblock = self._mol3d_cache.get(cid or "")
        if molblock is None:
            molblock, _ = await run.cpu_bound(
                generate_3d_conformer_molblock, cand.canonical_smiles
            )
            if cid and molblock:
                self._mol3d_cache[cid] = molblock
        if not molblock:
            ui.notify("Не удалось сгенерировать 3D-конформер", type="warning")
            return
        self._viewer3d.load_molblock(molblock, list(cand.highlighted_atom_indices))

    async def _export_csv(self) -> None:
        cands = list(self.runtime.view_model.candidates_by_id.values())
        if not cands:
            ui.notify("Нет кандидатов для экспорта", type="warning")
            return
        target = settings.resolved_workspace_dir / self.runtime.session_id / "ui_export.csv"
        await run.io_bound(export_candidates_to_csv, cands, target)
        ui.download(target)

    async def _export_sdf(self) -> None:
        cands = list(self.runtime.view_model.candidates_by_id.values())
        if not cands:
            ui.notify("Нет кандидатов для экспорта", type="warning")
            return
        target = settings.resolved_workspace_dir / self.runtime.session_id / "ui_export.sdf"
        await run.io_bound(export_candidates_to_sdf, cands, target)
        ui.download(target)

    async def _export_trace(self) -> None:
        traces = [ev.model_dump(mode="json") for ev in self.runtime.state.get("trace_log", [])]
        payload = json.dumps(traces, ensure_ascii=False, indent=2).encode("utf-8")
        ui.download(payload, filename=f"{self.runtime.session_id}_trace_log.json")

    async def _export_report(self) -> None:
        report_md = str(
            self.runtime.state.get("final_report_markdown") or "# Отчёт ещё не сформирован"
        )
        ui.download(
            report_md.encode("utf-8"),
            filename=f"{self.runtime.session_id}_report.md",
        )

    def refresh(self) -> None:
        """Обновляет содержимое инспектора под текущего выбранного кандидата."""
        cid, cand, row = self._get_selected()
        if cid is None or cand is None or row is None:
            if self._header_title is not None:
                self._header_title.set_text("Инспектор: нет выбранной молекулы")
            if self._smiles_label is not None:
                self._smiles_label.set_text("Выберите строку в таблице или точку на графике")
            if self._pairwise_box is not None:
                self._pairwise_box.clear()
            if self._viewer3d is not None:
                self._viewer3d.clear_viewer()
            if self._desirability_box is not None:
                self._desirability_box.clear()
            if self._provenance_box is not None:
                self._provenance_box.clear()
            return

        if self._header_title is not None:
            self._header_title.set_text(f"Инспектор · {cid} (MPO {row.mpo_score:.3f})")
        if self._smiles_label is not None:
            self._smiles_label.set_text(cand.canonical_smiles)
        if self._pin_btn is not None:
            self._pin_btn.props("color=amber" if row.pinned else "flat color=slate")
        if self._exclude_btn is not None:
            self._exclude_btn.props("color=rose" if row.excluded else "flat color=slate")

        sid = self.runtime.session_id
        if self._pairwise_box is not None:
            self._pairwise_box.clear()
            with self._pairwise_box:
                with ui.column().classes("items-center flex-1 gap-1"):
                    ui.label(f"Parent ({cand.parent_chembl_id})").classes(
                        "text-[11px] text-slate-400"
                    )
                    ui.image(f"/api/sessions/{sid}/mol/{cid}/svg?variant=parent").classes(
                        "w-full max-w-[170px] bg-white rounded border border-slate-700"
                    )
                ui.icon("arrow_forward", color="emerald-400", size="sm")
                with ui.column().classes("items-center flex-1 gap-1"):
                    ui.label(f"Candidate (ΔpIC50 {row.delta_activity:+.2f})").classes(
                        "text-[11px] text-emerald-300"
                    )
                    ui.image(f"/api/sessions/{sid}/mol/{cid}/svg?variant=diff").classes(
                        "w-full max-w-[170px] bg-white rounded border border-slate-700"
                    )

        if self._viewer3d is not None:
            cached_3d = self._mol3d_cache.get(cid)
            if cached_3d:
                self._viewer3d.load_molblock(cached_3d, list(cand.highlighted_atom_indices))
            else:
                self._viewer3d.clear_viewer()

        if self._desirability_box is not None:
            self._desirability_box.clear()
            metrics = [
                ("d1 Активность (pIC50)", row.desirability_d1_activity),
                ("d2 Липофильность (cLogP)", row.desirability_d2_clogp),
                ("d3 Синтез (SAscore)", row.desirability_d3_sa),
                ("d4 Лекарственность (QED)", row.desirability_d4_qed),
                ("d5 Безопасность (hERG)", row.desirability_d5_herg),
            ]
            with self._desirability_box:
                for lbl, val in metrics:
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        ui.label(lbl).classes("text-[11px] text-slate-300 w-40")
                        ui.linear_progress(value=max(0.0, min(1.0, val)), show_value=False).props(
                            "color=emerald track-color=slate-800 rounded"
                        ).classes("flex-1")
                        ui.label(f"{val:.2f}").classes(
                            "text-[11px] font-mono text-slate-300 w-9 text-right"
                        )

        if self._provenance_box is not None:
            self._provenance_box.clear()
            with self._provenance_box:
                self._prov_card(
                    "1. Экспериментальный факт (ChEMBL / Wet-Lab)",
                    f"Parent: {cand.parent_chembl_id} | Exp pIC50: {cand.parent_exp_pic50:.2f} | "
                    f"Assay: {cand.parent_assay_type} ({cand.parent_bao_format}) | Source: {cand.parent_source_dataset}",
                )
                self._prov_card(
                    "2. Расчёт RDKit",
                    f"MW: {cand.calc_rdkit.mw:.1f} | cLogP: {cand.calc_rdkit.clogp:.2f} | "
                    f"TPSA: {cand.calc_rdkit.tpsa:.1f} | SA: {cand.calc_rdkit.sa_score:.2f} | "
                    f"QED: {cand.calc_rdkit.qed:.2f} | PAINS: {cand.calc_rdkit.pains_alerts_count} | Brenk: {cand.calc_rdkit.brenk_alerts_count}",
                )
                self._prov_card(
                    "3. Прогноз ADMET",
                    f"Pred pIC50: {cand.pred_admet.predicted_pic50:.2f} | "
                    f"hERG: {cand.pred_admet.herg_liability_prob:.2f} ({cand.pred_admet.herg_risk_level}) | "
                    f"Source: {cand.pred_admet.model_source}",
                )
                self._prov_card(
                    "4. Гипотеза SAR / Critic",
                    f"Origin: {cand.hypothesis_rationale.generation_origin} | "
                    f"Rule: {cand.hypothesis_rationale.rule_id} | "
                    f"SMIRKS: {cand.hypothesis_rationale.applied_smirks}\n"
                    f"Нарушения: {', '.join(row.violations) if row.violations else 'нет'}",
                )

    def _prov_card(self, title: str, body: str) -> None:
        with ui.card().classes("w-full bg-slate-900 border border-slate-800 p-2 gap-0.5"):
            ui.label(title).classes("text-[11px] font-semibold text-sky-300")
            ui.label(body).classes("text-[11px] text-slate-300 whitespace-pre-line")
