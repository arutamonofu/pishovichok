"""Выдвижная панель конфигурации 83 параметров вычислительного ядра (`ToolConfig`) для Chemical IDE."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from nicegui import ui
from rdkit import Chem
from rdkit.Chem import Descriptors

from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import ParamSourceLiteral, ToolConfig
from pishovichok.ui.bridge import RestartStage, agent_bridge
from pishovichok.ui.session import SessionRuntime
from pishovichok.ui.state_vm import (
    PARAM_SOURCE_BADGE_META,
    ApplyModeLiteral,
    classify_tool_config_change_impact,
)

ALL_RDKIT_DESCRIPTOR_NAMES: list[str] = [name for name, _fn in Descriptors._descList]

_BAO_FORMAT_OPTIONS: list[str] = [
    "BAO_0000357",
    "BAO_0000219",
    "BAO_0000218",
    "BAO_0000019",
    "BAO_0000366",
]

_STANDARD_TYPE_OPTIONS: list[str] = ["IC50", "Ki", "Kd", "EC50", "AC50"]

_FILTER_CATALOG_OPTIONS: list[str] = [
    "PAINS_A",
    "PAINS_B",
    "PAINS_C",
    "Brenk",
    "Glaxo",
    "SureChEMBL",
    "NIH",
    "ZINC",
]

_MPO_WEIGHT_FIELDS: list[tuple[str, str]] = [
    ("pic50", "Вес pIC50"),
    ("logp", "Вес cLogP"),
    ("sascore", "Вес SAscore"),
    ("qed", "Вес QED"),
    ("tpsa", "Вес TPSA"),
    ("logs", "Вес LogS"),
    ("herg", "Вес hERG"),
]

_APPLY_MODE_LABELS: dict[ApplyModeLiteral, str] = {
    "instant_mpo": "Мгновенный пересчёт MPO (< 50 мс)",
    "evaluation": "Требуется пересчёт оценки (evaluation_node)",
    "design": "Требуется перезапуск генерации (design_node)",
    "sar": "Требуется пересчёт SAR (sar_node)",
    "research": "Требуется полный перезапуск (research_node)",
}


def validate_smarts_pattern(smarts: str) -> tuple[bool, str]:
    """Проверяет синтаксис SMARTS через `Chem.MolFromSmarts`."""
    cleaned = smarts.strip()
    if not cleaned:
        return True, ""
    mol = Chem.MolFromSmarts(cleaned)
    if mol is None:
        return False, "Невалидный SMARTS-паттерн (Chem.MolFromSmarts вернул None)"
    return True, cleaned


def parse_modifiable_atom_indices(raw_text: str) -> tuple[bool, list[int] | None, str]:
    """Парсит список индексов модифицируемых атомов из строки вида '0, 1, 5'."""
    cleaned = raw_text.strip()
    if not cleaned:
        return True, None, ""
    tokens = [tok.strip() for tok in cleaned.replace(";", ",").replace(" ", ",").split(",") if tok.strip()]
    parsed: list[int] = []
    for tok in tokens:
        if not tok.isdigit():
            return False, None, f"Некорректный индекс атома: '{tok}' (ожидаются целые >= 0)"
        idx = int(tok)
        if idx not in parsed:
            parsed.append(idx)
    return True, parsed, ""


def format_modifiable_atom_indices(indices: list[int] | None) -> str:
    """Форматирует список индексов атомов в компактную строку."""
    if not indices:
        return ""
    return ", ".join(str(i) for i in indices)


class ToolSettingsDrawer:
    """Выдвижная панель настройки всех параметров `ToolConfig` с бейджами источников и 4 режимами пересчёта."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_local_mpo_updated: Callable[[], Any] | None = None,
        on_3d_pca_params_updated: Callable[[], Any] | None = None,
        on_protected_smarts_changed: Callable[[str], Any] | None = None,
        get_sketcher_smiles: Callable[[], str] | None = None,
        set_sketcher_smiles: Callable[[str], Any] | None = None,
        on_tool_config_changed: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_local_mpo_updated = on_local_mpo_updated
        self._on_3d_pca_params_updated = on_3d_pca_params_updated
        self._on_protected_smarts_changed = on_protected_smarts_changed
        self._get_sketcher_smiles = get_sketcher_smiles
        self._set_sketcher_smiles = set_sketcher_smiles
        self._on_tool_config_changed = on_tool_config_changed

        self._drawer: ui.right_drawer | None = None
        self._syncing: bool = False
        self._pending_keys: set[str] = set()
        self._impact_label: ui.label | None = None
        self._badges: dict[str, ui.badge] = {}
        self._control_updaters: list[Callable[[ToolConfig], None]] = []

        # Локальные виджеты с особыми правилами валидации/синхронизации
        self._protected_smarts_input: ui.input | None = None
        self._protected_smarts_error: ui.label | None = None
        self._mod_atoms_input: ui.input | None = None
        self._mod_atoms_error: ui.label | None = None
        self._ban_smarts_input: ui.input | None = None
        self._ban_smarts_error: ui.label | None = None
        self._ban_smarts_chips_row: ui.row | None = None
        self._mpo_weight_sliders: dict[str, tuple[ui.slider, ui.label]] = {}

        self._build()

    @property
    def is_open(self) -> bool:
        return bool(self._drawer.value) if self._drawer is not None else False

    def toggle(self) -> None:
        """Открывает или закрывает боковую панель настроек."""
        if self._drawer is not None:
            self._drawer.toggle()

    def open(self) -> None:
        """Открывает панель настроек."""
        if self._drawer is not None:
            self._drawer.show()

    def close(self) -> None:
        """Закрывает панель настроек."""
        if self._drawer is not None:
            self._drawer.hide()

    def _build(self) -> None:
        total_params = len(ToolConfig.model_fields)
        self._drawer = (
            ui.right_drawer(value=False, fixed=False, bordered=True, elevated=True)
            .props("width=560 overlay behavior=mobile")
            .classes("bg-slate-950 border-l border-slate-800 p-0 text-slate-100")
        )
        with self._drawer, ui.column().classes("w-full h-full no-wrap gap-0"):
            # Шапка панели
            with ui.column().classes(
                "w-full gap-1.5 px-4 py-3 bg-slate-900 border-b border-slate-800"
            ):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("tune", color="emerald-400", size="sm")
                        ui.label(
                            f"Конфигурация вычислительного ядра ({total_params} параметров)"
                        ).classes("text-sm font-semibold text-slate-100")
                    ui.button(
                        icon="close",
                        on_click=self.close,
                    ).props("dense flat round size=sm color=slate")

                self._impact_label = ui.label(
                    "Все изменения синхронизируются с state['tool_config']"
                ).classes("text-[11px] text-slate-400")

                # Кнопки быстрого применения / сброса
                with ui.row().classes("w-full gap-1.5 flex-wrap pt-1"):
                    ui.button(
                        "Пересчитать MPO (<50мс)",
                        icon="bolt",
                        on_click=self._handle_apply_instant_mpo,
                    ).props("dense outline size=sm color=emerald").classes("text-[11px]")
                    ui.button(
                        "Пересчитать оценку",
                        icon="fact_check",
                        on_click=lambda: self._handle_restart_stage("evaluation"),
                    ).props("dense outline size=sm color=sky").classes("text-[11px]")
                    ui.button(
                        "Перезапуск генерации",
                        icon="auto_fix_high",
                        on_click=lambda: self._handle_restart_stage("design"),
                    ).props("dense outline size=sm color=purple").classes("text-[11px]")
                    ui.button(
                        "Пересчёт SAR",
                        icon="hub",
                        on_click=lambda: self._handle_restart_stage("sar"),
                    ).props("dense outline size=sm color=amber").classes("text-[11px]")
                    ui.button(
                        "Полный перезапуск",
                        icon="restart_alt",
                        on_click=lambda: self._handle_restart_stage("research"),
                    ).props("dense outline size=sm color=rose").classes("text-[11px]")
                    ui.button(
                        "Сбросить по умолчанию",
                        icon="settings_backup_restore",
                        on_click=self._handle_reset_defaults,
                    ).props("dense flat size=sm color=slate").classes("text-[11px]")

            # Вкладки (5 подсистем)
            with ui.tabs().props("dense align=left no-caps inline-label").classes(
                "w-full bg-slate-900/70 border-b border-slate-800 text-xs"
            ) as tabs:
                t1 = ui.tab("1. Курация")
                t2 = ui.tab("2. SAR/SALI")
                t3 = ui.tab("3. CReM/Дизайн")
                t4 = ui.tab("4. Фильтры/ADMET")
                t5 = ui.tab("5. MPO/3D/Критик")

            with (
                ui.scroll_area().classes("w-full flex-1"),
                ui.tab_panels(tabs, value=t1).classes("w-full bg-slate-950 p-3"),
            ):
                with ui.tab_panel(t1).classes("p-0 gap-2.5"):
                    self._build_tab_curation()
                with ui.tab_panel(t2).classes("p-0 gap-2.5"):
                    self._build_tab_sar()
                with ui.tab_panel(t3).classes("p-0 gap-2.5"):
                    self._build_tab_design()
                with ui.tab_panel(t4).classes("p-0 gap-2.5"):
                    self._build_tab_evaluation()
                with ui.tab_panel(t5).classes("p-0 gap-2.5"):
                    self._build_tab_scoring_3d_critic()

        self.refresh()

    # =========================================================================
    # Декларативные строители контролов и бейджей источников
    # =========================================================================

    def _render_param_header(self, field_name: str, label: str) -> None:
        with ui.row().classes("w-full items-center justify-between gap-2"):
            ui.label(label).classes("text-xs font-medium text-slate-200")
            badge = ui.badge("default", color="slate").props("dense outline")
            badge.classes("text-[10px] px-1.5 py-0")
            self._badges[field_name] = badge

    def _add_select(
        self,
        field_name: str,
        label: str,
        options: list[Any] | dict[Any, str],
    ) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = getattr(tc, field_name)
        with ui.column().classes("w-full gap-1 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
            self._render_param_header(field_name, label)
            sel = (
                ui.select(
                    options=options,
                    value=cur_val,
                    on_change=lambda e, fn=field_name: self._on_field_changed(fn, e.value),
                )
                .props("dense outlined dark options-dense")
                .classes("w-full text-xs")
            )

        def _update(cfg: ToolConfig) -> None:
            sel.set_value(getattr(cfg, field_name))

        self._control_updaters.append(_update)

    def _add_multiselect(
        self,
        field_name: str,
        label: str,
        base_options: list[str],
        *,
        with_input: bool = False,
        allow_custom: bool = False,
    ) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_vals = list(getattr(tc, field_name) or [])
        opts = list(dict.fromkeys([*base_options, *cur_vals]))
        props_str = "dense outlined dark use-chips options-dense"
        with ui.column().classes("w-full gap-1 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
            self._render_param_header(field_name, label)
            sel = (
                ui.select(
                    options=opts,
                    value=cur_vals,
                    multiple=True,
                    with_input=with_input or allow_custom,
                    new_value_mode="add-unique" if allow_custom else None,
                    on_change=lambda e, fn=field_name: self._on_field_changed(
                        fn, list(e.value or [])
                    ),
                )
                .props(props_str)
                .classes("w-full text-xs")
            )

        def _update(cfg: ToolConfig) -> None:
            vals = list(getattr(cfg, field_name) or [])
            merged_opts = list(dict.fromkeys([*base_options, *vals]))
            if sel.options != merged_opts:
                sel.set_options(merged_opts, value=vals)
            else:
                sel.set_value(vals)

        self._control_updaters.append(_update)

    def _add_number(
        self,
        field_name: str,
        label: str,
        *,
        min_val: float | None = None,
        max_val: float | None = None,
        step: float = 1.0,
        is_int: bool = False,
    ) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = getattr(tc, field_name)
        with ui.column().classes("w-full gap-1 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
            self._render_param_header(field_name, label)

            def _handle(e: Any, fn: str = field_name) -> None:
                if e.value is None or e.value == "":
                    return
                raw = float(e.value)
                parsed: int | float = round(raw) if is_int else raw
                if min_val is not None and parsed < min_val:
                    return
                if max_val is not None and parsed > max_val:
                    return
                self._on_field_changed(fn, parsed)

            num = (
                ui.number(
                    value=cur_val,
                    min=min_val,
                    max=max_val,
                    step=step,
                    on_change=_handle,
                )
                .props("dense outlined dark")
                .classes("w-full text-xs font-mono")
            )

        def _update(cfg: ToolConfig) -> None:
            num.set_value(getattr(cfg, field_name))

        self._control_updaters.append(_update)

    def _add_slider(
        self,
        field_name: str,
        label: str,
        *,
        min_val: float,
        max_val: float,
        step: float,
    ) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = float(getattr(tc, field_name))
        with ui.column().classes("w-full gap-0.5 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
            self._render_param_header(field_name, label)
            val_lbl = ui.label(f"{cur_val:.2f}").classes(
                "text-[11px] font-mono text-emerald-300"
            )

            def _handle(e: Any, fn: str = field_name) -> None:
                val = float(e.value)
                val_lbl.set_text(f"{val:.2f}")
                self._on_field_changed(fn, val)

            sld = ui.slider(
                min=min_val,
                max=max_val,
                step=step,
                value=cur_val,
                on_change=_handle,
            ).props("dense color=emerald")

        def _update(cfg: ToolConfig) -> None:
            val = float(getattr(cfg, field_name))
            sld.set_value(val)
            val_lbl.set_text(f"{val:.2f}")

        self._control_updaters.append(_update)

    def _add_checkbox(self, field_name: str, label: str) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = bool(getattr(tc, field_name))
        with ui.row().classes(
            "w-full items-center justify-between bg-slate-900/60 p-2 rounded border border-slate-800/80"
        ):
            chk = ui.checkbox(
                label,
                value=cur_val,
                on_change=lambda e, fn=field_name: self._on_field_changed(fn, bool(e.value)),
            ).props("dense dark color=emerald").classes("text-xs text-slate-200")
            badge = ui.badge("default", color="slate").props("dense outline")
            badge.classes("text-[10px] px-1.5 py-0")
            self._badges[field_name] = badge

        def _update(cfg: ToolConfig) -> None:
            chk.set_value(bool(getattr(cfg, field_name)))

        self._control_updaters.append(_update)

    def _add_text(self, field_name: str, label: str, placeholder: str = "") -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = str(getattr(tc, field_name) or "")
        with ui.column().classes("w-full gap-1 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
            self._render_param_header(field_name, label)
            inp = (
                ui.input(
                    value=cur_val,
                    placeholder=placeholder,
                    on_change=lambda e, fn=field_name: self._on_field_changed(
                        fn, str(e.value or "").strip()
                    ),
                )
                .props("dense outlined dark")
                .classes("w-full text-xs font-mono")
            )

        def _update(cfg: ToolConfig) -> None:
            inp.set_value(str(getattr(cfg, field_name) or ""))

        self._control_updaters.append(_update)

    # =========================================================================
    # Вкладки параметров (83 поля ToolConfig)
    # =========================================================================

    def _build_tab_curation(self) -> None:
        with ui.column().classes("w-full gap-2"):
            self._add_multiselect(
                "allowed_bao_formats",
                "Допустимые BAO-форматы",
                _BAO_FORMAT_OPTIONS,
                allow_custom=True,
            )
            self._add_multiselect(
                "standard_types",
                "Типы биоактивности (standard_types)",
                _STANDARD_TYPE_OPTIONS,
                allow_custom=True,
            )
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "relation_mode",
                    "Цензурированные значения",
                    {"include_censored": "include_censored", "exact_only": "exact_only (=)"},
                )
                self._add_select(
                    "duplicate_agg_method",
                    "Агрегация дубликатов",
                    ["median", "mean", "max"],
                )
                self._add_select(
                    "scaffold_mode",
                    "Режим скаффолда",
                    {"bemis_murcko": "Bemis-Murcko", "generic_framework": "Generic framework"},
                )
                self._add_number("pchembl_min_cutoff", "Мин. порог pChEMBL", min_val=0.0, max_val=15.0, step=0.1)
                self._add_number("max_duplicate_std", "Макс. STD дубликатов", min_val=0.0, step=0.05)
                self._add_number("chembl_api_limit", "Лимит ChEMBL API", min_val=10, max_val=5000, step=10, is_int=True)
                self._add_number("min_neighborhood_rows_fallback", "Мин. строк окрестности", min_val=1, step=1, is_int=True)
                self._add_number("top_scaffolds_limit", "Лимит топ-скаффолдов", min_val=1, max_val=50, step=1, is_int=True)

            self._add_checkbox("strip_salts", "Удалять соли и противоионы (strip_salts)")
            self._add_checkbox("neutralize_charges", "Нейтрализовать заряды (neutralize_charges)")
            self._add_checkbox("canonicalize_tautomers", "Канонизировать таутомеры (canonicalize_tautomers)")
            self._add_checkbox("keep_chirality", "Сохранять хиральность (keep_chirality)")

    def _build_tab_sar(self) -> None:
        with ui.column().classes("w-full gap-2"):
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select("mmpa_max_cuts", "rdMMPA макс. разрезов", [1, 2, 3])
                self._add_select("fp_type", "Тип отпечатков (fp_type)", ["ecfp", "fcfp", "rdkit", "maccs"])
                self._add_select("fp_radius", "Радиус фингерпринта", [1, 2, 3])
                self._add_select("fp_nbits", "Размерность битов (fp_nbits)", [1024, 2048, 4096])
                self._add_select("similarity_metric", "Метрика сходства", ["tanimoto", "dice", "cosine"])
                self._add_number("mmpa_max_cut_bonds", "Макс. разрезаемых связей", min_val=1, step=1, is_int=True)
                self._add_number("mmpa_min_core_heavy_atoms", "Мин. тяж. атомов кора", min_val=1, step=1, is_int=True)
                self._add_number("min_support_pairs", "Мин. пар поддержки (Support)", min_val=1, step=1, is_int=True)
                self._add_number("min_mean_delta_pic50", "Мин. сдвиг ΔpIC50", step=0.05)
                self._add_number("max_rgroup_heavy_atoms", "Макс. тяж. атомов R-группы", min_val=1, step=1, is_int=True)
                self._add_number("sali_epsilon", "SALI epsilon", min_val=0.001, step=0.01)
                self._add_number("min_rules_before_enrichment", "Мин. правил до обогащения", min_val=0, step=1, is_int=True)

            self._add_slider("cliff_similarity_cutoff", "Порог сходства Activity Cliff", min_val=0.0, max_val=1.0, step=0.01)
            self._add_slider("cliff_delta_pic50_cutoff", "Порог ΔpIC50 Activity Cliff", min_val=0.0, max_val=5.0, step=0.1)
            self._add_text("background_target_id", "Фоновая мишень ChEMBL (background_target_id)", "CHEMBL203")
            self._add_checkbox("prioritize_positive_rules", "Приоритизировать правила с ΔpIC50 > 0")

    def _build_tab_design(self) -> None:
        with ui.column().classes("w-full gap-2"):
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "generation_engine",
                    "Движок генерации",
                    ["mmp_and_crem", "mmp_only", "crem_only"],
                )
                self._add_select("crem_mode", "Режим CReM", ["mutate", "grow"])
                self._add_select("crem_context_radius", "Контекстный радиус CReM", [1, 2, 3])
                self._add_number("crem_min_rel_atoms", "CReM min_rel_atoms", step=1, is_int=True)
                self._add_number("crem_max_rel_atoms", "CReM max_rel_atoms", step=1, is_int=True)
                self._add_number("crem_max_replacements", "CReM макс. замен", min_val=1, step=1, is_int=True)
                self._add_number("crem_max_fragment_heavy_atoms", "Макс. тяж. атомов фрагмента", min_val=1, step=1, is_int=True)
                self._add_number("bioisostere_fallback_threshold", "Порог фолбэка биоизостеров", min_val=0, step=1, is_int=True)
                self._add_number("max_parent_seeds", "Макс. родительских сидов", min_val=1, max_val=50, step=1, is_int=True)
                self._add_number("max_candidates_pool", "Макс. пул кандидатов", min_val=1, step=1, is_int=True)
                self._add_number("top_k_output", "Топ-K на выходе (top_k_output)", min_val=1, max_val=50, step=1, is_int=True)

            self._add_slider("neighborhood_tanimoto", "Порог окрестности Танимото", min_val=0.0, max_val=1.0, step=0.01)
            self._add_checkbox("freeze_murcko_core", "Фиксировать каркас Bemis-Murcko (freeze_murcko_core)")
            self._add_checkbox("use_bioisostere_catalog", "Использовать каталог биоизостеров")

            # protected_smarts с валидацией Chem.MolFromSmarts и синхронизацией со скетчером / _smarts_input
            tc = resolve_tool_config(self.runtime.state)
            with ui.column().classes("w-full gap-1 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
                self._render_param_header("protected_smarts", "Защищённый SMARTS кора (protected_smarts)")
                self._protected_smarts_input = (
                    ui.input(
                        value=str(tc.protected_smarts or ""),
                        placeholder="Например: c1ncnc2ccccc12",
                        on_change=self._handle_protected_smarts_change,
                    )
                    .props("dense outlined dark clearable")
                    .classes("w-full text-xs font-mono")
                )
                self._protected_smarts_error = ui.label("").classes("text-[11px] text-rose-400")
                self._protected_smarts_error.set_visibility(False)
                with ui.row().classes("w-full gap-1.5"):
                    ui.button(
                        "Взять из скетчера",
                        icon="download",
                        on_click=self._pull_smarts_from_sketcher,
                    ).props("dense flat size=sm color=sky").classes("text-[11px]")
                    ui.button(
                        "Загрузить в скетчер",
                        icon="upload",
                        on_click=self._push_smarts_to_sketcher,
                    ).props("dense flat size=sm color=sky").classes("text-[11px]")

            # modifiable_atom_indices
            with ui.column().classes("w-full gap-1 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
                self._render_param_header("modifiable_atom_indices", "Модифицируемые индексы атомов (через запятую)")
                self._mod_atoms_input = (
                    ui.input(
                        value=format_modifiable_atom_indices(tc.modifiable_atom_indices),
                        placeholder="Например: 0, 1, 5 (пусто = без ограничений)",
                        on_change=self._handle_mod_atoms_change,
                    )
                    .props("dense outlined dark clearable")
                    .classes("w-full text-xs font-mono")
                )
                self._mod_atoms_error = ui.label("").classes("text-[11px] text-rose-400")
                self._mod_atoms_error.set_visibility(False)

    def _build_tab_evaluation(self) -> None:
        with ui.column().classes("w-full gap-2"):
            self._add_multiselect(
                "active_filter_catalogs",
                "Каталоги структурных фильтров",
                _FILTER_CATALOG_OPTIONS,
            )
            self._add_checkbox("brenk_ignore_long_chain", "Brenk: игнорировать Aliphatic_long_chain")

            # Редактор custom_ban_smarts
            with ui.column().classes("w-full gap-1.5 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
                self._render_param_header("custom_ban_smarts", "Пользовательские запрещённые SMARTS")
                with ui.row().classes("w-full items-center gap-1.5"):
                    self._ban_smarts_input = (
                        ui.input(placeholder="Введите SMARTS, напр. [N+](=O)[O-]")
                        .props("dense outlined dark")
                        .classes("flex-1 text-xs font-mono")
                    )
                    ui.button(
                        "Добавить",
                        icon="add",
                        on_click=self._handle_add_ban_smarts,
                    ).props("dense outline size=sm color=emerald").classes("text-xs")
                self._ban_smarts_error = ui.label("").classes("text-[11px] text-rose-400")
                self._ban_smarts_error.set_visibility(False)
                self._ban_smarts_chips_row = ui.row().classes("w-full gap-1 flex-wrap")

            self._add_multiselect(
                "extra_rdkit_descriptors",
                f"Доп. дескрипторы RDKit ({len(ALL_RDKIT_DESCRIPTOR_NAMES)} доступно)",
                ALL_RDKIT_DESCRIPTOR_NAMES,
                with_input=True,
            )

            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "admet_backend",
                    "Бэкенд прогноза ADMET",
                    [
                        "auto",
                        "auto_catboost",
                        "admet_ai_dmpnn",
                        "catboost_local_qsar",
                        "rdkit_analytical_qsar",
                        "empirical_only",
                    ],
                )
                self._add_number("brics_min_fragment_size", "BRICS мин. размер фрагмента", min_val=1, step=1, is_int=True)
            self._add_checkbox("brics_only_leaf_nodes", "BRICS только терминальные листья (only_leaf_nodes)")

            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_number("lipinski_mw_cutoff", "Липински MW <= ", min_val=1.0, step=10.0)
                self._add_number("lipinski_logp_cutoff", "Липински cLogP <= ", step=0.1)
                self._add_number("lipinski_hbd_cutoff", "Липински HBD <= ", min_val=0, step=1, is_int=True)
                self._add_number("lipinski_hba_cutoff", "Липински HBA <= ", min_val=0, step=1, is_int=True)
                self._add_number("herg_low_cutoff", "hERG low <=", min_val=0.0, max_val=1.0, step=0.05)
                self._add_number("herg_high_cutoff", "hERG high >=", min_val=0.0, max_val=1.0, step=0.05)
                self._add_number("logs_soluble_cutoff", "LogS soluble >=", step=0.5)
                self._add_number("logs_moderate_cutoff", "LogS moderate >=", step=0.5)
                self._add_number("logs_poor_cutoff", "LogS poor >=", step=0.5)

    def _build_tab_scoring_3d_critic(self) -> None:
        tc = resolve_tool_config(self.runtime.state)
        with ui.column().classes("w-full gap-2"):
            # Веса MPO (синхронизируются с MPOCanvas)
            with ui.column().classes("w-full gap-1.5 bg-slate-900/60 p-2 rounded border border-slate-800/80"):
                self._render_param_header("mpo_weights", "Веса Харрингтона MPO (синхр. с графиком)")
                with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-1"):
                    for w_key, w_label in _MPO_WEIGHT_FIELDS:
                        w_val = float(tc.mpo_weights.get(w_key, 1.0))
                        with ui.column().classes("w-full gap-0"):
                            lbl = ui.label(f"{w_label}: {w_val:.2f}").classes(
                                "text-[11px] text-slate-300"
                            )

                            def _on_w(e: Any, wk: str = w_key, lb: ui.label = lbl, wl: str = w_label) -> None:
                                v = float(e.value)
                                lb.set_text(f"{wl}: {v:.2f}")
                                self._on_field_changed("mpo_weights", {wk: v})

                            sld = ui.slider(
                                min=0.0,
                                max=5.0,
                                step=0.1,
                                value=w_val,
                                on_change=_on_w,
                            ).props("dense color=emerald")
                            self._mpo_weight_sliders[w_key] = (sld, lbl)

            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "cns_profile_mode",
                    "Профиль CNS / TPSA",
                    ["peripheral_low_bbb", "cns_high_bbb", "neutral"],
                )
                self._add_select(
                    "alert_penalty_mode",
                    "Штраф за алерты",
                    ["veto", "soft_penalty", "ignore"],
                )
            self._add_slider(
                "alert_soft_multiplier",
                "Множитель soft_penalty",
                min_val=0.0,
                max_val=1.0,
                step=0.05,
            )

            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_number("harrington_activity_low_delta", "Harrington pIC50 low Δ", step=0.05)
                self._add_number("harrington_activity_high_delta", "Harrington pIC50 high Δ", step=0.05)
                self._add_number("harrington_qed_min", "Harrington QED min", min_val=0.0, max_val=1.0, step=0.05)
                self._add_number("harrington_qed_target", "Harrington QED target", min_val=0.0, max_val=1.0, step=0.05)
                self._add_number("harrington_sa_min", "Harrington SA min", min_val=1.0, max_val=10.0, step=0.1)

                self._add_select("etkdg_variant", "3D генератор ETKDG", ["ETKDGv3", "ETKDGv2"])
                self._add_select("force_field", "Силовое поле 3D", ["MMFF94s", "MMFF94", "UFF", "none"])
                self._add_number("num_conformers", "Число 3D-конформеров", min_val=1, max_val=50, step=1, is_int=True)
                self._add_number("conformer_opt_max_iters", "Макс. итераций 3D-опт.", min_val=0, max_val=5000, step=25, is_int=True)
                self._add_number("conformer_random_seed", "Seed 3D-конформеров", step=1, is_int=True)
                self._add_number("pca_random_seed", "Seed PCA-проекции", step=1, is_int=True)

                self._add_number("max_iterations", "Макс. итераций дизайна (Критик)", min_val=0, max_val=10, step=1, is_int=True)
                self._add_number("max_refilter_iterations", "Макс. итераций перефильтрации BAO", min_val=0, max_val=5, step=1, is_int=True)
                self._add_number("critic_fallback_min_tpsa", "Critic fallback min TPSA", min_val=0.0, step=1.0)
                self._add_number("critic_logp_step_reduction", "Critic шаг снижения logP", min_val=0.0, step=0.1)

            self._add_checkbox("strict_assay_homogeneity", "Строгий аудит однородности BAO (Петля 1)")
            self._add_checkbox("strict_property_thresholds", "Строгая проверка порогов свойств (Петля 2)")

    # =========================================================================
    # Обработка изменений, валидация, синхронизация и 4 режима применения
    # =========================================================================

    def _on_field_changed(self, field_name: str, value: Any) -> None:
        if self._syncing:
            return
        changed = agent_bridge.apply_ui_tool_overrides(self.runtime, {field_name: value})
        if not changed:
            return
        self._pending_keys.update(changed)
        impact = classify_tool_config_change_impact(changed)
        self._refresh_badges_and_impact()

        if impact == "instant_mpo":
            # Мгновенный локальный пересчёт MPO (<50мс) или сброс 3D/PCA кэша
            self.runtime.view_model.recalculate_mpo()
            if self._on_local_mpo_updated is not None:
                self._on_local_mpo_updated()
            if field_name in {
                "etkdg_variant",
                "force_field",
                "num_conformers",
                "conformer_opt_max_iters",
                "conformer_random_seed",
                "pca_random_seed",
            } and self._on_3d_pca_params_updated is not None:
                self._on_3d_pca_params_updated()
        if self._on_tool_config_changed is not None:
            self._on_tool_config_changed()

    def _handle_protected_smarts_change(self, e: Any) -> None:
        if self._syncing:
            return
        raw = str(e.value or "")
        ok, cleaned_or_err = validate_smarts_pattern(raw)
        if not ok:
            if self._protected_smarts_error is not None:
                self._protected_smarts_error.set_text(cleaned_or_err)
                self._protected_smarts_error.set_visibility(True)
            return
        if self._protected_smarts_error is not None:
            self._protected_smarts_error.set_visibility(False)
        target_val: str | None = cleaned_or_err or None
        changed = agent_bridge.apply_ui_tool_overrides(
            self.runtime, {"protected_smarts": target_val}
        )
        if changed:
            self._pending_keys.update(changed)
            self._refresh_badges_and_impact()
        if self._on_protected_smarts_changed is not None:
            self._on_protected_smarts_changed(cleaned_or_err)
        if self._on_tool_config_changed is not None:
            self._on_tool_config_changed()

    def _handle_mod_atoms_change(self, e: Any) -> None:
        if self._syncing:
            return
        raw = str(e.value or "")
        ok, parsed, err = parse_modifiable_atom_indices(raw)
        if not ok:
            if self._mod_atoms_error is not None:
                self._mod_atoms_error.set_text(err)
                self._mod_atoms_error.set_visibility(True)
            return
        if self._mod_atoms_error is not None:
            self._mod_atoms_error.set_visibility(False)
        changed = agent_bridge.apply_ui_tool_overrides(
            self.runtime, {"modifiable_atom_indices": parsed}
        )
        if changed:
            self._pending_keys.update(changed)
            self._refresh_badges_and_impact()
        if self._on_tool_config_changed is not None:
            self._on_tool_config_changed()

    def _handle_add_ban_smarts(self) -> None:
        if self._ban_smarts_input is None:
            return
        raw = str(self._ban_smarts_input.value or "").strip()
        if not raw:
            return
        ok, cleaned_or_err = validate_smarts_pattern(raw)
        if not ok:
            if self._ban_smarts_error is not None:
                self._ban_smarts_error.set_text(cleaned_or_err)
                self._ban_smarts_error.set_visibility(True)
            return
        if self._ban_smarts_error is not None:
            self._ban_smarts_error.set_visibility(False)
        tc = resolve_tool_config(self.runtime.state)
        updated_list = list(tc.custom_ban_smarts)
        if cleaned_or_err not in updated_list:
            updated_list.append(cleaned_or_err)
        self._ban_smarts_input.set_value("")
        self._on_field_changed("custom_ban_smarts", updated_list)
        self._render_ban_smarts_chips(updated_list)

    def _handle_remove_ban_smarts(self, smarts: str) -> None:
        tc = resolve_tool_config(self.runtime.state)
        updated_list = [s for s in tc.custom_ban_smarts if s != smarts]
        self._on_field_changed("custom_ban_smarts", updated_list)
        self._render_ban_smarts_chips(updated_list)

    def _render_ban_smarts_chips(self, items: list[str]) -> None:
        if self._ban_smarts_chips_row is None:
            return
        self._ban_smarts_chips_row.clear()
        with self._ban_smarts_chips_row:
            if not items:
                ui.label("Список пуст").classes("text-[11px] text-slate-500")
            for sm in items:
                ui.chip(
                    sm,
                    removable=True,
                    color="rose",
                    on_value_change=lambda e, s=sm: (not e.value)
                    and self._handle_remove_ban_smarts(s),
                ).props("dense outline size=sm").classes("text-[11px] font-mono")

    def _pull_smarts_from_sketcher(self) -> None:
        if self._get_sketcher_smiles is None:
            return
        smi = str(self._get_sketcher_smiles() or "").strip()
        if not smi:
            ui.notify("Скетчер пуст", type="warning")
            return
        if self._protected_smarts_input is not None:
            self._protected_smarts_input.set_value(smi)
        ui.notify("Структура из скетчера перенесена в protected_smarts", type="info")

    def _push_smarts_to_sketcher(self) -> None:
        if self._set_sketcher_smiles is None or self._protected_smarts_input is None:
            return
        val = str(self._protected_smarts_input.value or "").strip()
        if not val:
            return
        self._set_sketcher_smiles(val)
        ui.notify("protected_smarts отправлен в 2D-скетчер", type="info")

    async def _handle_apply_instant_mpo(self) -> None:
        await agent_bridge.apply_local_mpo_recalculation(self.runtime)
        self._pending_keys.clear()
        self._refresh_badges_and_impact()
        if self._on_local_mpo_updated is not None:
            res = self._on_local_mpo_updated()
            if asyncio.iscoroutine(res):
                await res
        ui.notify("MPO-рейтинг пересчитан локально (< 50 мс)", type="positive")

    async def _handle_restart_stage(self, stage: RestartStage) -> None:
        self._pending_keys.clear()
        self._refresh_badges_and_impact()
        await agent_bridge.restart_branch(self.runtime.session_id, target_stage=stage)
        ui.notify(f"Запущен пересчёт с узла '{stage}'", type="info")

    async def _handle_reset_defaults(self) -> None:
        await agent_bridge.reset_tool_config_to_defaults(self.runtime)
        self._pending_keys.clear()
        self.refresh()
        if self._on_protected_smarts_changed is not None:
            self._on_protected_smarts_changed("")
        if self._on_local_mpo_updated is not None:
            res = self._on_local_mpo_updated()
            if asyncio.iscoroutine(res):
                await res
        if self._on_tool_config_changed is not None:
            res2 = self._on_tool_config_changed()
            if asyncio.iscoroutine(res2):
                await res2
        ui.notify("Параметры ToolConfig сброшены по умолчанию", type="info")

    def _refresh_badges_and_impact(self) -> None:
        sources: dict[str, ParamSourceLiteral] = dict(
            self.runtime.state.get("param_sources", {})
        )
        for field_name, badge in self._badges.items():
            src = sources.get(field_name, "default")
            text, color = PARAM_SOURCE_BADGE_META.get(src, ("default", "slate"))
            badge.set_text(text)
            if src == "default":
                badge.props(f"color={color} outline")
            else:
                badge.props(f"color={color}")
                badge.props(remove="outline")

        if self._impact_label is not None:
            if not self._pending_keys:
                self._impact_label.set_text(
                    "Параметры синхронизированы со state['tool_config']"
                )
            else:
                mode = classify_tool_config_change_impact(self._pending_keys)
                self._impact_label.set_text(
                    f"Изменено параметров: {len(self._pending_keys)} · {_APPLY_MODE_LABELS[mode]}"
                )

    def refresh(self) -> None:
        """Двусторонне синхронизирует все контролы и бейджи с `runtime.state`."""
        tc = resolve_tool_config(self.runtime.state)
        self._syncing = True
        try:
            for updater in self._control_updaters:
                updater(tc)
            for w_key, (sld, lbl) in self._mpo_weight_sliders.items():
                w_label = dict(_MPO_WEIGHT_FIELDS).get(w_key, w_key)
                w_val = float(tc.mpo_weights.get(w_key, 1.0))
                sld.set_value(w_val)
                lbl.set_text(f"{w_label}: {w_val:.2f}")
            if self._protected_smarts_input is not None:
                self._protected_smarts_input.set_value(str(tc.protected_smarts or ""))
            if self._protected_smarts_error is not None:
                self._protected_smarts_error.set_visibility(False)
            if self._mod_atoms_input is not None:
                self._mod_atoms_input.set_value(
                    format_modifiable_atom_indices(tc.modifiable_atom_indices)
                )
            if self._mod_atoms_error is not None:
                self._mod_atoms_error.set_visibility(False)
            self._render_ban_smarts_chips(list(tc.custom_ban_smarts))
            self._refresh_badges_and_impact()
        finally:
            self._syncing = False
