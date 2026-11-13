"""Правый инспектор кандидата: Сравнение 2D / BRICS / ленивый 3D, Харрингтон и верификация (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import run, ui

from pishovichok.core.rendering import generate_3d_conformer_molblock
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import ToolConfig
from pishovichok.ui.components.chem_widgets import MoleculeViewer3D
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import SessionRuntime


def build_inspector_telemetry_summary(tool_config: ToolConfig) -> dict[str, str]:
    """Формирует компактные строки телеметрии активных параметров для режима «Хемоинформатик»."""
    cats = (
        ", ".join(tool_config.active_filter_catalogs)
        if tool_config.active_filter_catalogs
        else "нет"
    )
    return {
        "conformer_3d": (
            f"{tool_config.etkdg_variant} + {tool_config.force_field} "
            f"(n={tool_config.num_conformers}, ит={tool_config.conformer_opt_max_iters})"
        ),
        "rdkit_filters": (
            f"Фильтры: {cats} [{tool_config.alert_penalty_mode}] | "
            f"BRICS≥{tool_config.brics_min_fragment_size} | "
            f"Доп. дескр.: {len(tool_config.extra_rdkit_descriptors)}"
        ),
        "admet": f"Модель: {tool_config.admet_backend}",
        "sar_fp": (
            f"FP: {tool_config.fp_type}/r{tool_config.fp_radius}/{tool_config.fp_nbits} | "
            f"{tool_config.similarity_metric} | "
            f"Перепады: S≥{tool_config.cliff_similarity_cutoff:.2f}, Δ≥{tool_config.cliff_delta_pic50_cutoff:.1f}"
        ),
    }


def _desirability_bar_hex(val: float) -> str:
    """Возвращает цвет полосы желательности Харрингтона по значению d_i."""
    if val >= 0.7:
        return "#10B981"
    if val >= 0.4:
        return "#F59E0B"
    return "#EF4444"


_ORIGIN_LABELS: dict[str, str] = {
    "mmp_smirks_rule": "Правило MMP",
    "crem_fragment_replacement": "Замена CReM",
    "bioisostere_catalog": "Каталог биоизостеров",
    "graph_ga": "Эволюция GraphGA (BRICS)",
}

_ADMET_MODEL_LABELS: dict[str, str] = {
    "admet_ai_dmpnn": "ADMET-AI (D-MPNN)",
    "catboost_local_qsar": "CatBoost (локальный QSAR)",
    "rdkit_analytical_qsar": "Аналитика RDKit",
}

_HERG_RISK_LABELS: dict[str, str] = {
    "low": "низкий",
    "moderate": "средний",
    "high": "высокий",
}

_SOURCE_DATASET_LABELS: dict[str, str] = {
    "ChEMBL_Parquet": "ChEMBL (локальный)",
    "ChEMBL_API": "ChEMBL API",
    "Custom_WetLab_CSV": "Лабораторный CSV",
}


class InspectorPane:
    """Правая панель: карточка кандидата, 3-вкладочный визуализатор, Харрингтон и верификация."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_row_updated: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_row_updated = on_row_updated
        self._active_viz_tab: str = "diff"
        self._header_title: ui.label | None = None
        self._smiles_label: ui.label | None = None
        self._pin_btn: ui.button | None = None
        self._exclude_btn: ui.button | None = None
        self._pairwise_box: ui.column | None = None
        self._brics_box: ui.column | None = None
        self._conformer_container: ui.column | None = None
        self._conformer_title: ui.label | None = None
        self._viewer3d: MoleculeViewer3D | None = None
        self._desirability_box: ui.column | None = None
        self._docking_box: ui.column | None = None
        self._provenance_box: ui.column | None = None
        self._telemetry_bar: ui.row | None = None
        self._mol3d_cache: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        with ui.column().classes(
            "w-full h-full flex flex-col overflow-hidden bg-[#FAF8F9] text-[#1F181B] border-l border-[#E2D9DD]"
        ):
            # Фиксированная шапка карточки кандидата
            with ui.column().classes(
                "w-full shrink-0 gap-1.5 px-3 pt-2.5 pb-2 bg-[#FAF8F9] border-b border-[#E2D9DD]"
            ):
                with ui.row().classes("w-full items-center justify-between gap-2 no-wrap"):
                    self._header_title = ui.label("Карточка кандидата").classes(
                        "text-xs font-semibold text-[#5C2D3B] truncate"
                    )
                    with ui.row().classes("items-center gap-1 shrink-0"):
                        expand_btn = (
                            ui.button(on_click=self._open_enlarged_modal)
                            .props("dense flat size=sm color=grey-7")
                            .classes("p-1 min-w-0")
                            .tooltip("Открыть 2D и 3D в крупном окне")
                        )
                        expand_btn._props["title"] = "Открыть в крупном окне"
                        with expand_btn:
                            ui.html(svg_icon("maximize", 14), sanitize=False)

                        self._pin_btn = (
                            ui.button(on_click=self._toggle_pin)
                            .props("dense flat size=sm color=grey-7")
                            .classes("p-1 min-w-0")
                            .tooltip("Закрепить")
                        )
                        self._pin_btn._props["title"] = "Закрепить"
                        with self._pin_btn:
                            ui.html(svg_icon("pin", 14), sanitize=False)

                        self._exclude_btn = (
                            ui.button(on_click=self._toggle_exclude)
                            .props("dense flat size=sm color=grey-7")
                            .classes("p-1 min-w-0")
                            .tooltip("Исключить")
                        )
                        self._exclude_btn._props["title"] = "Исключить"
                        with self._exclude_btn:
                            ui.html(svg_icon("exclude", 14), sanitize=False)

                self._smiles_label = ui.label("").classes(
                    "text-[11px] mono-code break-all px-2 py-1.5 w-full max-h-14 overflow-y-auto wetlab-scroll"
                )
                self._telemetry_bar = ui.row().classes("w-full gap-1 flex-wrap")

            # Прокручиваемое тело инспектора с явным скроллбаром
            with ui.column().classes(
                "w-full flex-1 min-h-0 overflow-y-auto wetlab-scroll gap-2.5 p-3"
            ):
                # Табовый контейнер визуализации (Сравнение 2D | BRICS-блоки | 3D-конформер)
                with ui.card().classes(
                    "w-full shrink-0 bg-white border border-[#E2D9DD] rounded-lg p-1.5 gap-1 shadow-none"
                ):
                    with (
                        ui.tabs(on_change=self._handle_viz_tab_change)
                        .props(
                            "dense align=justify no-caps mobile-breakpoint=0 outside-arrows=false "
                            "active-color=primary indicator-color=primary"
                        )
                        .classes("w-full border-b border-[#E2D9DD] text-[#594D52] compact-tabs") as viz_tabs
                    ):
                        t_diff = ui.tab("diff", label="Сравнение 2D")
                        t_brics = ui.tab("brics", label="BRICS-блоки")
                        t_3d = ui.tab("3d", label="3D-конформер")

                    with ui.tab_panels(viz_tabs, value=t_diff).props("animated=false").classes(
                        "w-full bg-white p-1"
                    ):
                        with ui.tab_panel(t_diff).classes("p-0.5"):
                            self._pairwise_box = ui.column().classes("w-full gap-2")
                        with ui.tab_panel(t_brics).classes("p-0.5"):
                            self._brics_box = ui.column().classes("w-full gap-2")
                        with ui.tab_panel(t_3d).classes("p-0.5"):
                            with ui.row().classes("w-full items-center justify-between mb-1"):
                                self._conformer_title = ui.label("3D-конформер (ETKDGv3)").classes(
                                    "text-[11px] font-medium text-[#594D52]"
                                )
                                with ui.row().classes("items-center gap-1"):
                                    with ui.button(on_click=self._load_3d_conformer).props(
                                        "dense flat size=sm no-caps color=primary"
                                    ).classes("text-[11px] gap-1"):
                                        ui.html(svg_icon("refresh", 13), sanitize=False)
                                        ui.label("Пересчитать").classes("text-[11px]")
                                    fullscreen_3d_btn = (
                                        ui.button(on_click=self._open_3d_fullscreen_modal)
                                        .props("dense flat size=sm color=grey-7")
                                        .classes("p-1 min-w-0")
                                        .tooltip("Развернуть 3D-конформер на весь экран")
                                    )
                                    fullscreen_3d_btn._props["title"] = "На весь экран"
                                    with fullscreen_3d_btn:
                                        ui.html(svg_icon("maximize", 13), sanitize=False)
                            self._conformer_container = ui.column().classes("w-full gap-1")
                            ui.label("ЛКМ — вращение · Колесо — зум · ПКМ — сдвиг").classes(
                                "text-[10px] text-[#64748B] text-center w-full"
                            )

                # Профиль желательности (Харрингтон) и Источники данных и верификация
                ui.label("Профиль желательности (Харрингтон)").classes(
                    "text-xs font-semibold text-[#334155] shrink-0"
                )
                with ui.card().classes("w-full shrink-0 bg-white border border-[#E2E8F0] rounded-lg p-2.5 shadow-none"):
                    self._desirability_box = ui.column().classes("w-full gap-1.5")

                with ui.row().classes("w-full items-center justify-between shrink-0"):
                    ui.label("Профиль связывания в активном центре").classes(
                        "text-xs font-semibold text-[#334155]"
                    )
                    ui.label("3D-моделирование / Докинг").classes(
                        "text-[10px] font-medium px-1.5 py-0.5 rounded bg-[#EFF6FF] text-[#1D4ED8] border border-[#BFDBFE]"
                    )
                with ui.card().classes("w-full shrink-0 bg-white border border-[#E2E8F0] rounded-lg p-2.5 shadow-none"):
                    self._docking_box = ui.column().classes("w-full gap-1.5")

                ui.label("Источники данных и верификация").classes(
                    "text-xs font-semibold text-[#334155] shrink-0"
                )
                self._provenance_box = ui.column().classes("w-full shrink-0 gap-1.5 pb-2")

        self.refresh()

    async def _handle_viz_tab_change(self, e: Any) -> None:
        self._active_viz_tab = str(e.value or "diff")
        if self._active_viz_tab == "3d":
            self._ensure_3d_viewer()
            await self._load_3d_conformer()

    def _ensure_3d_viewer(self) -> None:
        """Ленивая инициализация WebGL-холста только при переключении на вкладку 3D-конформера."""
        if self._viewer3d is None and self._conformer_container is not None:
            with self._conformer_container:
                self._viewer3d = MoleculeViewer3D(height="300px")

    async def _open_enlarged_modal(self) -> None:
        """Открывает увеличенный 2D/3D-просмотр выбранного кандидата."""
        cid, cand, row = self._get_selected()
        if cid is None or cand is None or row is None:
            ui.notify("Сначала выберите кандидата", type="info")
            return
        sid = self.runtime.session_id
        tool_config = resolve_tool_config(self.runtime.state)
        cache_key = (
            f"{cid}_{tool_config.etkdg_variant}_{tool_config.force_field}_"
            f"{tool_config.num_conformers}_{tool_config.conformer_opt_max_iters}_"
            f"{tool_config.conformer_random_seed}"
        )
        molblock = self._mol3d_cache.get(cache_key)
        if molblock is None:
            molblock, _ = await run.cpu_bound(
                generate_3d_conformer_molblock,
                cand.canonical_smiles,
                num_conformers=tool_config.num_conformers,
                random_seed=tool_config.conformer_random_seed,
                tool_config=tool_config,
            )
            if molblock:
                self._mol3d_cache[cache_key] = molblock

        with ui.dialog() as dlg, ui.card().classes(
            "w-[920px] max-w-[95vw] bg-white border border-[#CBD5E1] rounded-lg p-4 gap-3 shadow-xl"
        ):
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label(f"Детальный просмотр структур · {cid}").classes(
                    "text-sm font-semibold text-[#0F172A]"
                )
                with ui.button(on_click=dlg.close).props("dense flat size=sm color=grey-8").classes(
                    "p-1 min-w-0"
                ):
                    ui.html(svg_icon("close", 16), sanitize=False)

            ui.label(cand.canonical_smiles).classes(
                "text-xs mono-code px-2.5 py-1.5 w-full break-all"
            )

            with ui.grid(columns=2).classes("w-full gap-3"):
                with ui.card().classes("bg-white border border-[#E2E8F0] p-2 gap-1 shadow-none"):
                    ui.label(f"Прототип: {cand.parent_chembl_id} (pIC50: {cand.parent_exp_pic50:.2f})").classes(
                        "text-xs font-semibold text-[#4C1D95] bg-[#F5F3FF] px-2 py-1 rounded"
                    )
                    ui.image(
                        f"/api/sessions/{sid}/mol/{cid}/svg?variant=parent&width=520&height=300"
                    ).classes("w-full h-[230px] object-contain bg-white")

                with ui.card().classes("bg-white border border-[#E2E8F0] p-2 gap-1 shadow-none"):
                    delta_cls = (
                        "text-[#065F46] bg-[#ECFDF5]"
                        if row.delta_activity >= 0
                        else "text-[#9F1239] bg-[#FFF1F2]"
                    )
                    ui.label(f"Кандидат {cid} (ΔpIC50: {row.delta_activity:+.2f}, MPO: {row.mpo_score:.3f})").classes(
                        f"text-xs font-semibold {delta_cls} px-2 py-1 rounded"
                    )
                    ui.image(
                        f"/api/sessions/{sid}/mol/{cid}/svg?variant=diff&width=520&height=300"
                    ).classes("w-full h-[230px] object-contain bg-white")

            ui.label(
                f"3D-конформер ({tool_config.etkdg_variant} + {tool_config.force_field})"
            ).classes("text-xs font-semibold text-[#334155]")
            modal_3d = MoleculeViewer3D(height="320px")
            if molblock:
                modal_3d.load_molblock(molblock, list(cand.highlighted_atom_indices))
        dlg.open()

    async def _open_3d_fullscreen_modal(self) -> None:
        """Открывает 3D-конформер или докинг-комплекс выбранного кандидата в полноэкранном модальном окне."""
        cid, cand, _ = self._get_selected()
        if cid is None or cand is None:
            ui.notify("Сначала выберите кандидата", type="info")
            return
        tool_config = resolve_tool_config(self.runtime.state)

        has_docking = (
            cand.docking is not None
            and cand.docking.status == "SUCCESS"
            and bool(cand.docking.pose_sdf_block)
        )

        modal_title = (
            f"3D-комплекс связывания (Докинг): {cid}"
            if has_docking
            else f"3D-конформер: {cid}"
        )
        if has_docking:
            score_str = (
                f" · ΔG = {cand.docking.docking_score_kcal:.2f} ккал/моль"
                if cand.docking.docking_score_kcal is not None
                else ""
            )
            modal_subtitle = f"(PDB: {cand.docking.pdb_id or '—'}{score_str})"
        else:
            modal_subtitle = f"({tool_config.etkdg_variant} + {tool_config.force_field})"

        molblock: str | None = None
        receptor_pdb = ""
        if has_docking:
            if cand.docking.receptor_pdb_path:
                rec_path = Path(cand.docking.receptor_pdb_path)
                if rec_path.is_file():
                    receptor_pdb = await run.io_bound(
                        rec_path.read_text, encoding="utf-8"
                    )
        else:
            cache_key = (
                f"{cid}_{tool_config.etkdg_variant}_{tool_config.force_field}_"
                f"{tool_config.num_conformers}_{tool_config.conformer_opt_max_iters}_"
                f"{tool_config.conformer_random_seed}"
            )
            molblock = self._mol3d_cache.get(cache_key)
            if molblock is None:
                molblock, _ = await run.cpu_bound(
                    generate_3d_conformer_molblock,
                    cand.canonical_smiles,
                    num_conformers=tool_config.num_conformers,
                    random_seed=tool_config.conformer_random_seed,
                    tool_config=tool_config,
                )
                if molblock:
                    self._mol3d_cache[cache_key] = molblock

        with ui.dialog().classes("w-full h-full p-2 md:p-6") as dlg, ui.card().classes(
            "w-[94vw] h-[88vh] max-w-6xl bg-white border border-[#CBD5E1] rounded-2xl p-4 gap-2.5 shadow-2xl flex flex-col overflow-hidden"
        ):
            with ui.row().classes("w-full items-center justify-between gap-2 shrink-0 pb-1 border-b border-[#E2D9DD]"):
                with ui.row().classes("items-center gap-2 min-w-0"):
                    ui.label(modal_title).classes(
                        "text-sm md:text-base font-semibold text-[#5C2D3B] truncate"
                    )
                    ui.label(modal_subtitle).classes(
                        "text-xs text-[#594D52]"
                    )
                with ui.row().classes("items-center gap-2"):
                    ui.label("ЛКМ — вращение · Колесо — зум · ПКМ — сдвиг").classes(
                        "text-[11px] text-[#64748B] hidden sm:block"
                    )
                    with ui.button(on_click=dlg.close).props("dense flat size=sm color=grey-8").classes(
                        "p-1 min-w-0"
                    ):
                        ui.html(svg_icon("close", 18), sanitize=False)

            ui.label(cand.canonical_smiles).classes(
                "text-xs mono-code px-3 py-1.5 w-full break-all bg-[#FAF8F9] rounded-lg shrink-0 border border-[#E2D9DD]"
            )

            with ui.column().classes("w-full flex-1 min-h-0 relative"):
                fs_viewer = MoleculeViewer3D(height="100%")
                if has_docking:
                    fs_viewer.load_docking_complex(
                        cand.docking.pose_sdf_block,
                        receptor_pdb,
                        list(cand.docking.key_residues),
                    )
                elif molblock:
                    fs_viewer.load_molblock(molblock, list(cand.highlighted_atom_indices))

        dlg.open()

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

    async def _dock_selected_candidate(self) -> None:
        from pishovichok.ui.bridge import agent_bridge

        cid, _, _ = self._get_selected()
        if not cid:
            return
        ui.notify(f"Запуск 3D-докинга для {cid}...", type="info")
        try:
            summary = await agent_bridge.run_standalone_docking(
                self.runtime.session_id,
                candidate_ids=[cid],
            )
            if summary.get("status") == "SUCCESS":
                dg = summary.get("best_score_kcal")
                dg_txt = f" (ΔG = {dg:.2f} ккал/моль)" if dg is not None else ""
                ui.notify(f"3D-докинг для {cid} завершён{dg_txt}", type="positive")
            else:
                ui.notify(
                    str(summary.get("message") or f"Статус: {summary.get('status')}"),
                    type="warning",
                )
        except Exception as exc:
            ui.notify(f"Ошибка докинга: {exc}", type="negative")

    async def _load_3d_conformer(self) -> None:
        cid, cand, _ = self._get_selected()
        if cand is None:
            return
        self._ensure_3d_viewer()
        if self._viewer3d is None:
            return

        if (
            cand.docking is not None
            and cand.docking.status == "SUCCESS"
            and cand.docking.pose_sdf_block
        ):
            receptor_pdb = ""
            if cand.docking.receptor_pdb_path:
                rec_path = Path(cand.docking.receptor_pdb_path)
                if rec_path.is_file():
                    receptor_pdb = await run.io_bound(
                        rec_path.read_text, encoding="utf-8"
                    )
            self._viewer3d.load_docking_complex(
                cand.docking.pose_sdf_block,
                receptor_pdb,
                list(cand.docking.key_residues),
            )
            return

        tool_config = resolve_tool_config(self.runtime.state)
        cache_key = (
            f"{cid or ''}_{tool_config.etkdg_variant}_{tool_config.force_field}_"
            f"{tool_config.num_conformers}_{tool_config.conformer_opt_max_iters}_"
            f"{tool_config.conformer_random_seed}"
        )
        molblock = self._mol3d_cache.get(cache_key)
        if molblock is None:
            molblock, _ = await run.cpu_bound(
                generate_3d_conformer_molblock,
                cand.canonical_smiles,
                num_conformers=tool_config.num_conformers,
                random_seed=tool_config.conformer_random_seed,
                tool_config=tool_config,
            )
            if cid and molblock:
                self._mol3d_cache[cache_key] = molblock
        if not molblock:
            ui.notify("Не удалось сгенерировать 3D-конформер", type="warning")
            return
        self._viewer3d.load_molblock(molblock, list(cand.highlighted_atom_indices))

    def invalidate_3d_cache(self) -> None:
        """Сбрасывает кэш 3D-конформеров при изменении параметров ETKDG/силового поля."""
        self._mol3d_cache.clear()

    def refresh(self) -> None:
        """Обновляет содержимое инспектора под текущего выбранного кандидата."""
        tool_config = resolve_tool_config(self.runtime.state)
        telem = build_inspector_telemetry_summary(tool_config)
        is_expert = self.runtime.depth_mode == "expert"

        if self._conformer_title is not None:
            self._conformer_title.set_text(f"3D-конформер ({telem['conformer_3d']})")

        if self._telemetry_bar is not None:
            self._telemetry_bar.clear()
            self._telemetry_bar.set_visibility(is_expert)
            if is_expert:
                with self._telemetry_bar:
                    for chip_text in (
                        telem["sar_fp"],
                        telem["rdkit_filters"],
                        telem["admet"],
                    ):
                        ui.label(chip_text).classes(
                            "text-[10px] mono-code px-1.5 py-0.5"
                        )

        cid, cand, row = self._get_selected()
        if cid is None or cand is None or row is None:
            is_running = self.runtime.status == "running"
            if self._header_title is not None:
                self._header_title.set_text("Карточка кандидата")
            if self._smiles_label is not None:
                self._smiles_label.set_text(
                    "Ожидание кандидатов…"
                    if is_running
                    else "Выберите строку в таблице или точку на графике"
                )
            if self._pairwise_box is not None:
                self._pairwise_box.clear()
                with self._pairwise_box:
                    with ui.column().classes(
                        "w-full h-[170px] items-center justify-center gap-2 border border-dashed border-[#D5C8CD] rounded-md bg-[#FAF8F9]/70 px-4 text-center"
                    ):
                        if is_running:
                            ui.spinner("dots", size="sm", color="primary")
                            ui.label(
                                "После завершения оценки здесь появится 2D-сравнение прототипа и выбранного кандидата"
                            ).classes("text-[11px] text-[#594D52] max-w-[280px]")
                        else:
                            ui.html(svg_icon("flask", 20, "text-[#8B495C]/60"), sanitize=False)
                            ui.label(
                                "Здесь появится 2D-сравнение прототипа и кандидата с подсветкой изменённого фрагмента"
                            ).classes("text-[11px] text-[#594D52] max-w-[280px]")
            if self._brics_box is not None:
                self._brics_box.clear()
                with self._brics_box:
                    with ui.column().classes(
                        "w-full h-[140px] items-center justify-center gap-1.5 border border-dashed border-[#D5C8CD] rounded-md bg-[#FAF8F9]/70 px-4 text-center"
                    ):
                        ui.label(
                            "После выбора кандидата здесь появятся синтетические блоки (BRICS)"
                        ).classes("text-[11px] text-[#594D52] max-w-[260px]")
            if self._viewer3d is not None:
                self._viewer3d.clear_viewer()
            if self._desirability_box is not None:
                self._desirability_box.clear()
                with self._desirability_box:
                    if is_running:
                        for lbl_txt, bar_w in (
                            ("pIC50", "w-3/4"),
                            ("cLogP", "w-2/3"),
                            ("SAscore", "w-4/5"),
                            ("QED", "w-1/2"),
                        ):
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.label(lbl_txt).classes("text-[10px] text-[#94A3B8] w-12 shrink-0")
                                ui.element("div").classes(
                                    f"h-2 {bar_w} rounded bg-[#E2E8F0] animate-pulse"
                                )
                    ui.label(
                        "Профиль желательности Харрингтона появится после оценки кандидатов"
                        if is_running
                        else "Здесь отобразятся частные функции желательности Харрингтона (pIC50, cLogP, SAscore, QED, LogS, hERG)"
                    ).classes("text-[11px] text-[#64748B]")
            if self._docking_box is not None:
                self._docking_box.clear()
                with self._docking_box:
                    if is_running:
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.element("div").classes("h-5 w-24 rounded bg-[#EFF6FF] border border-[#DBEAFE] animate-pulse")
                            ui.element("div").classes("h-5 flex-1 rounded bg-[#F1F5F9] animate-pulse")
                    ui.label(
                        "Ожидание результатов 3D-докинга (ΔG, PDB мишени, ключевые остатки)…"
                        if is_running
                        else "Здесь появятся расчётная аффинность ΔG (ккал/моль), PDB-структура мишени и ключевые остатки связывания"
                    ).classes("text-[11px] text-[#64748B]")
            if self._provenance_box is not None:
                self._provenance_box.clear()
                with self._provenance_box:
                    with ui.card().classes(
                        "w-full bg-white border border-[#E2E8F0] rounded-lg p-2.5 gap-1.5 shadow-none"
                    ):
                        if is_running:
                            with ui.row().classes("w-full gap-1.5"):
                                for bg_c in ("bg-[#F5F3FF]", "bg-[#ECFDF5]", "bg-[#FFFBEB]", "bg-[#EFF6FF]"):
                                    ui.element("div").classes(f"h-3 flex-1 rounded {bg_c} animate-pulse")
                        ui.label(
                            "Сбор источников данных: Эксперимент, Расчёт RDKit, Прогноз ADMET и Гипотеза SAR…"
                            if is_running
                            else "Здесь появятся источники значений по 4 категориям: Эксперимент, Расчёт RDKit, Прогноз ADMET и Гипотеза SAR"
                        ).classes("text-[11px] text-[#64748B]")
            return

        if self._header_title is not None:
            self._header_title.set_text(f"Карточка кандидата · {cid}")
        if self._smiles_label is not None:
            self._smiles_label.set_text(cand.canonical_smiles)
        if self._pin_btn is not None:
            self._pin_btn.props("color=amber-8" if row.pinned else "flat color=grey-7")
        if self._exclude_btn is not None:
            self._exclude_btn.props("color=negative" if row.excluded else "flat color=grey-7")

        sid = self.runtime.session_id
        # 1. Вкладка Сравнение 2D (крупные вертикальные структуры на #FFFFFF без тесного сжатия)
        if self._pairwise_box is not None:
            self._pairwise_box.clear()
            with self._pairwise_box:
                with ui.column().classes(
                    "w-full items-center gap-1 border border-[#E2E8F0] rounded-md p-1.5 bg-white"
                ):
                    with ui.row().classes(
                        "w-full items-center justify-between gap-2 bg-[#F5F3FF] px-2 py-0.5 rounded"
                    ):
                        ui.label(f"Прототип: {cand.parent_chembl_id}").classes(
                            "text-[11px] font-semibold text-[#4C1D95] truncate"
                        ).tooltip(cand.parent_chembl_id)
                        ui.label(f"pIC50: {cand.parent_exp_pic50:.2f}").classes(
                            "text-[11px] font-mono text-[#4C1D95] whitespace-nowrap shrink-0"
                        )
                    ui.image(
                        f"/api/sessions/{sid}/mol/{cid}/svg?variant=parent&width=480&height=220"
                    ).classes("w-full h-[150px] object-contain bg-white")

                with ui.row().classes("w-full items-center justify-center gap-1 text-[#64748B] py-0"):
                    ui.html(svg_icon("arrow_down", 13), sanitize=False)
                    ui.label(f"{row.method_rule_label}").classes("text-[10px] font-mono")

                delta_cls = (
                    "text-[#065F46] bg-[#ECFDF5]"
                    if row.delta_activity >= 0
                    else "text-[#9F1239] bg-[#FFF1F2]"
                )
                with ui.column().classes(
                    "w-full items-center gap-1 border border-[#E2E8F0] rounded-md p-1.5 bg-white"
                ):
                    with ui.row().classes(
                        f"w-full items-center justify-between gap-2 {delta_cls} px-2 py-0.5 rounded"
                    ):
                        ui.label(f"Кандидат · {cid}").classes(
                            "text-[11px] font-semibold truncate"
                        )
                        ui.label(f"ΔpIC50: {row.delta_activity:+.2f}").classes(
                            "text-[11px] font-mono font-semibold whitespace-nowrap shrink-0"
                        )
                    ui.image(
                        f"/api/sessions/{sid}/mol/{cid}/svg?variant=diff&width=480&height=220"
                    ).classes("w-full h-[155px] object-contain bg-white")

        # 2. Вкладка BRICS-блоки
        if self._brics_box is not None:
            self._brics_box.clear()
            frags = list(cand.calc_rdkit.brics_fragments or [])
            with self._brics_box:
                ui.label(
                    f"Разорвано связей BRICS: {cand.calc_rdkit.brics_bonds_broken} · Строительных блоков: {len(frags)}"
                ).classes("text-[11px] font-medium text-[#065F46]")
                if not frags:
                    ui.label("Фрагменты BRICS не выделены").classes("text-xs text-[#475569]")
                else:
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        for idx, frag_smi in enumerate(frags[:6]):
                            with ui.card().classes(
                                "w-full bg-white border border-[#E2E8F0] p-1.5 gap-1 items-center shadow-none"
                            ):
                                ui.image(
                                    f"/api/sessions/{sid}/mol/{cid}/brics/{idx}/svg?width=240&height=160"
                                ).classes("w-full h-[105px] object-contain bg-white")
                                ui.label(frag_smi).classes(
                                    "text-[10px] mono-code px-1.5 py-0.5 break-all w-full text-center"
                                )

        # 3. Ленивый 3D-конформер (только если открыта вкладка 3D-конформер)
        if self._active_viz_tab == "3d":
            self._ensure_3d_viewer()
            if self._viewer3d is not None:
                cache_key = (
                    f"{cid}_{tool_config.etkdg_variant}_{tool_config.force_field}_"
                    f"{tool_config.num_conformers}_{tool_config.conformer_opt_max_iters}_"
                    f"{tool_config.conformer_random_seed}"
                )
                cached_3d = self._mol3d_cache.get(cache_key) or self._mol3d_cache.get(cid)
                if cached_3d:
                    self._viewer3d.load_molblock(cached_3d, list(cand.highlighted_atom_indices))
                else:
                    self._viewer3d.clear_viewer()
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._load_3d_conformer())
                    except RuntimeError:
                        pass

        # Профиль желательности (Харрингтон)
        if self._desirability_box is not None:
            self._desirability_box.clear()
            metrics = [
                ("Активность (pIC50)", row.desirability_d1_activity),
                ("Липофильность (cLogP)", row.desirability_d2_clogp),
                ("Синтез (SAscore)", row.desirability_d3_sa),
                ("Лекарственность (QED)", row.desirability_d4_qed),
                ("Безопасность (hERG)", row.desirability_d5_herg),
            ]
            with self._desirability_box:
                for lbl, val in metrics:
                    bar_hex = _desirability_bar_hex(val)
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        ui.label(lbl).classes("text-[11px] text-[#0F172A] w-36 truncate")
                        ui.linear_progress(
                            value=max(0.0, min(1.0, val)),
                            show_value=False,
                        ).props("track-color=grey-3 rounded").style(
                            f"color: {bar_hex};"
                        ).classes("flex-1")
                        ui.label(f"{val:.2f}").classes(
                            "text-[11px] font-mono font-semibold text-[#0F172A] w-9 text-right"
                        )

        # Профиль связывания в активном центре (3D-моделирование / Докинг)
        if self._docking_box is not None:
            self._docking_box.clear()
            with self._docking_box:
                dock = cand.docking
                if dock is None or dock.status != "SUCCESS":
                    status_msg = (
                        dock.message
                        if dock is not None and dock.message
                        else "3D-докинг ещё не выполнялся для данного кандидата."
                    )
                    ui.label(status_msg).classes("text-[11px] text-[#475569]")
                    with ui.button(on_click=self._dock_selected_candidate).props(
                        "dense unelevated no-caps size=sm color=primary"
                    ).classes("text-[11px] px-2.5 gap-1 self-start"):
                        ui.html(svg_icon("play", 12), sanitize=False)
                        ui.label("Запустить 3D-докинг").classes("text-[11px] font-medium")
                else:
                    src_ru = (
                        "Кэш демо-мишени"
                        if dock.source == "DEMO_CACHE"
                        else (
                            "RCSB PDB автозагрузка"
                            if dock.source in ("RCSB_LIVE", "RCSB_CACHE")
                            else (dock.source or "Локальный кэш")
                        )
                    )
                    dg_val = (
                        f"{dock.docking_score_kcal:.2f} ккал/моль"
                        if dock.docking_score_kcal is not None
                        else "-"
                    )
                    with ui.grid(columns=2).classes("w-full gap-1.5"):
                        for k_lbl, v_txt in (
                            ("PDB мишени", f"{dock.pdb_id or '-'} ({src_ru})"),
                            ("Энергия ΔG", dg_val),
                            ("Движок 3D", dock.backend or "-"),
                            ("Статус", "Успешно"),
                        ):
                            with ui.column().classes(
                                "bg-[#F8FAFC] border border-[#E2E8F0] rounded px-1.5 py-1 gap-0 min-w-0"
                            ):
                                ui.label(k_lbl).classes("text-[10px] text-[#64748B]")
                                ui.label(v_txt).classes(
                                    "text-[11px] font-semibold text-[#0F172A] truncate"
                                ).tooltip(v_txt)

                    interaction_groups: list[tuple[str, list[str], str]] = [
                        (
                            "Водородные связи",
                            sorted(
                                set(dock.interactions.get("HBDonor", []))
                                | set(dock.interactions.get("HBAcceptor", []))
                            ),
                            "bg-[#ECFDF5] text-[#065F46] border-[#A7F3D0]",
                        ),
                        (
                            "Гидрофобные контакты",
                            list(dock.interactions.get("Hydrophobic", [])),
                            "bg-[#FFFBEB] text-[#92400E] border-[#FDE68A]",
                        ),
                        (
                            "π-стекинг",
                            list(dock.interactions.get("PiStacking", [])),
                            "bg-[#F3E8FF] text-[#6B21A8] border-[#DDD6FE]",
                        ),
                        (
                            "Ионные контакты",
                            sorted(
                                set(dock.interactions.get("Anionic", []))
                                | set(dock.interactions.get("Cationic", []))
                                | set(dock.interactions.get("CationPi", []))
                            ),
                            "bg-[#EFF6FF] text-[#1E40AF] border-[#BFDBFE]",
                        ),
                    ]
                    has_any_badges = any(res_list for _, res_list, _ in interaction_groups)
                    if has_any_badges:
                        with ui.column().classes("w-full gap-1 pt-0.5"):
                            for grp_name, res_list, badge_cls in interaction_groups:
                                if not res_list:
                                    continue
                                with ui.row().classes("w-full items-center gap-1 flex-wrap"):
                                    ui.label(f"{grp_name}:").classes(
                                        "text-[10px] font-medium text-[#475569]"
                                    )
                                    for rname in res_list[:6]:
                                        ui.label(rname).classes(
                                            f"text-[10px] font-mono px-1.5 py-0.5 rounded border {badge_cls}"
                                        )
                    else:
                        ui.label("Ключевые остатки кармана: " + (", ".join(dock.key_residues[:5]) or "-")).classes(
                            "text-[10px] text-[#64748B]"
                        )

                    with ui.button(on_click=self._dock_selected_candidate).props(
                        "dense flat no-caps size=xs color=primary"
                    ).classes("text-[11px] px-1.5 gap-1 self-start"):
                        ui.html(svg_icon("refresh", 12), sanitize=False)
                        ui.label("Пересчитать 3D-докинг").classes("text-[11px]")

        # Источники данных и верификация (4 категории в 2 колонки без технических дампов)
        if self._provenance_box is not None:
            self._provenance_box.clear()
            with self._provenance_box:
                src_label = _SOURCE_DATASET_LABELS.get(
                    cand.parent_source_dataset, cand.parent_source_dataset
                )
                self._prov_card_badges(
                    "cat-experiment-left",
                    "1. Эксперимент",
                    [
                        ("Прототип", cand.parent_chembl_id),
                        ("pIC50 (эксп.)", f"{cand.parent_exp_pic50:.2f}"),
                        ("Протокол BAO", cand.parent_bao_format),
                        ("Источник", src_label),
                    ],
                )

                rdkit_items: list[tuple[str, str]] = [
                    ("MW", f"{cand.calc_rdkit.mw:.1f}"),
                    ("cLogP", f"{cand.calc_rdkit.clogp:.2f}"),
                    ("TPSA", f"{cand.calc_rdkit.tpsa:.1f}"),
                    ("SAscore", f"{cand.calc_rdkit.sa_score:.2f}"),
                    ("QED", f"{cand.calc_rdkit.qed:.2f}"),
                    (
                        "Фильтры",
                        f"PAINS: {cand.calc_rdkit.pains_alerts_count}, Brenk: {cand.calc_rdkit.brenk_alerts_count}",
                    ),
                ]
                if is_expert and cand.calc_rdkit.extra_descriptors:
                    for dk, dv in list(cand.calc_rdkit.extra_descriptors.items())[:4]:
                        rdkit_items.append((dk, f"{dv:.2f}"))
                self._prov_card_badges("cat-calc-left", "2. Расчёт RDKit", rdkit_items)

                herg_ru = _HERG_RISK_LABELS.get(
                    cand.pred_admet.herg_risk_level, cand.pred_admet.herg_risk_level
                )
                model_ru = _ADMET_MODEL_LABELS.get(
                    cand.pred_admet.model_source, cand.pred_admet.model_source
                )
                self._prov_card_badges(
                    "cat-pred-left",
                    "3. Прогноз ADMET",
                    [
                        ("pIC50 (прогноз)", f"{cand.pred_admet.predicted_pic50:.2f}"),
                        ("Риск hERG", f"{cand.pred_admet.herg_liability_prob:.2f} ({herg_ru})"),
                        ("LogS", f"{cand.pred_admet.log_s_aq:.2f}"),
                        ("Модель", model_ru),
                    ],
                )

                if cand.hypothesis_rationale.rule_id == "CREM-GROW":
                    origin_ru = "Наращивание CReM (Grow)"
                elif cand.hypothesis_rationale.rule_id == "CREM-MUT":
                    origin_ru = "Мутация CReM (Mutate)"
                else:
                    origin_ru = _ORIGIN_LABELS.get(
                        cand.hypothesis_rationale.generation_origin,
                        cand.hypothesis_rationale.generation_origin,
                    )
                violations_txt = (
                    ", ".join(row.violations) if row.violations else "Соответствует"
                )
                sar_items: list[tuple[str, str]] = [
                    ("Метод", origin_ru),
                    ("Правило", cand.hypothesis_rationale.rule_id),
                    ("Ограничения", violations_txt),
                    ("Индекс MPO", f"{row.mpo_score:.3f}"),
                ]
                raw_smirks = str(cand.hypothesis_rationale.applied_smirks or "")
                compact_smirks = (
                    f"{raw_smirks[:69]}..." if len(raw_smirks) > 72 else raw_smirks
                )
                self._prov_card_badges(
                    "cat-hypothesis-left",
                    "4. Гипотеза SAR",
                    sar_items,
                    mono_footer=compact_smirks if is_expert else None,
                )

    def _prov_card_badges(
        self,
        cat_class: str,
        title: str,
        items: list[tuple[str, str]],
        *,
        mono_footer: str | None = None,
    ) -> None:
        with ui.card().classes(f"w-full {cat_class} rounded-md p-2 gap-1.5 shadow-none"):
            ui.label(title).classes("text-[11px] font-semibold")
            with ui.grid(columns=2).classes("w-full gap-1"):
                for k, v in items:
                    with ui.column().classes(
                        "bg-white/85 border border-[#E2E8F0] rounded px-1.5 py-1 gap-0 min-w-0"
                    ):
                        ui.label(k).classes("text-[10px] text-[#475569] leading-tight")
                        ui.label(v).classes(
                            "text-[11px] font-medium text-[#0F172A] leading-tight truncate"
                        ).tooltip(v)
            if mono_footer:
                ui.label(mono_footer).classes(
                    "text-[10px] mono-code px-1.5 py-0.5 truncate w-full"
                ).tooltip(mono_footer)
