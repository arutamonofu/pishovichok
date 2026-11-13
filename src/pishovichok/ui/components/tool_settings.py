"""Выдвижная панель конфигурации 83 параметров вычислительного ядра (`ToolConfig`) (Clean Wet-Lab Light Mode)."""

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
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import SessionRuntime
from pishovichok.ui.state_vm import (
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
    "evaluation": "Требуется пересчёт оценки ADMET",
    "design": "Требуется перезапуск генерации",
    "sar": "Требуется пересчёт анализа SAR",
    "research": "Требуется полный перезапуск с курации",
}

_PRIMARY_CTA_META: dict[ApplyModeLiteral, tuple[str, str, str]] = {
    "instant_mpo": ("Пересчитать MPO (<50 мс)", "bolt", "positive"),
    "evaluation": ("Пересчитать оценку ADMET", "fact_check", "primary"),
    "design": ("Перезапустить генерацию", "auto_fix_high", "primary"),
    "sar": ("Пересчитать анализ SAR", "hub", "secondary"),
    "research": ("Перезапустить с курации", "restart_alt", "primary"),
}

_DRAWER_BADGE_META: dict[ParamSourceLiteral, tuple[str, str]] = {
    "default": ("", ""),
    "nl_query": ("Из запроса", "info"),
    "ui_manual": ("Изменено", "primary"),
    "critic_loop": ("Авто-фильтр верификации", "warning"),
}


def resolve_primary_cta_spec(pending_keys: set[str] | list[str]) -> tuple[ApplyModeLiteral, str, str, str]:
    """Определяет минимально достаточный уровень пересчёта и метаданные главной CTA-кнопки."""
    mode: ApplyModeLiteral = (
        classify_tool_config_change_impact(pending_keys) if pending_keys else "instant_mpo"
    )
    label, icon, color = _PRIMARY_CTA_META[mode]
    return mode, label, icon, color


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
    """Выдвижная панель настройки всех параметров `ToolConfig` с отклонениями и контекстным Sticky Footer."""

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
        self._primary_cta_btn: ui.button | None = None
        self._badges: dict[str, ui.badge] = {}
        self._cards: dict[str, Any] = {}
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
        self._hitl_select: ui.select | None = None

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
            .classes("bg-white border-l border-[#E2E8F0] p-0 text-[#0F172A]")
        )
        with self._drawer, ui.column().classes("w-full h-full no-wrap gap-0"):
            # Чистая шапка панели
            with ui.column().classes(
                "w-full gap-1 px-4 py-3 bg-[#FAF8F9] border-b border-[#E2D9DD]"
            ):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2 text-[#73394B]"):
                        ui.html(svg_icon("sliders", 15), sanitize=False)
                        ui.label(
                            f"Параметры вычислительного ядра ({total_params})"
                        ).classes("text-sm font-semibold text-[#5C2D3B]")
                    close_btn = ui.button(
                        on_click=self.close,
                    ).props("dense flat round size=sm color=grey-7")
                    with close_btn:
                        ui.html(svg_icon("close", 14), sanitize=False)

                self._impact_label = ui.label(
                    "Параметры синхронизированы с текущей сессией"
                ).classes("text-[11px] text-[#594D52]")

            # Вкладки (5 подсистем)
            with (
                ui.tabs()
                .props("dense align=left no-caps inline-label active-color=primary indicator-color=primary")
                .classes("w-full bg-[#F4EFF1] border-b border-[#E2D9DD] text-xs text-[#261F22] compact-tabs") as tabs
            ):
                t1 = ui.tab("1. Курация")
                t2 = ui.tab("2. Анализ SAR")
                t3 = ui.tab("3. Генерация")
                t4 = ui.tab("4. Фильтры и ADMET")
                t5 = ui.tab("5. MPO, 3D и контроль")

            with (
                ui.scroll_area().classes("w-full flex-1 bg-white"),
                ui.tab_panels(tabs, value=t1).classes("w-full bg-white p-3"),
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

            # Фиксированный нижний футер с одной динамической главной кнопкой и ссылкой сброса
            with ui.row().classes(
                "w-full items-center justify-between gap-2 px-4 py-2.5 bg-[#F8FAFC] border-t border-[#E2E8F0]"
            ):
                self._primary_cta_btn = ui.button(
                    "Пересчитать MPO (<50 мс)",
                    on_click=self._handle_primary_cta,
                ).props("dense unelevated no-caps color=green-7").classes("text-xs font-medium px-3")

                ui.button(
                    "Сбросить по умолчанию",
                    on_click=self._handle_reset_defaults,
                ).props("dense flat no-caps size=sm color=grey-7").classes("text-xs underline")

        self.refresh()

    # =========================================================================
    # Декларативные строители секций, контролов и бейджей отклонений
    # =========================================================================

    def _section_header(self, title: str) -> None:
        ui.label(title).classes(
            "text-[11px] uppercase tracking-wider font-semibold text-[#475569] border-b border-[#E2E8F0] pb-1 mt-2 w-full"
        )

    def _render_param_header(self, field_name: str, label: str) -> None:
        with ui.row().classes("w-full items-center justify-between gap-2"):
            ui.label(label).classes("text-xs font-medium text-[#0F172A]")
            badge = ui.badge("", color="primary").props("dense")
            badge.classes("text-[10px] px-1.5 py-0")
            badge.set_visibility(False)
            self._badges[field_name] = badge

    def _add_select(
        self,
        field_name: str,
        label: str,
        options: list[Any] | dict[Any, str],
    ) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = getattr(tc, field_name)
        card = ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]")
        self._cards[field_name] = card
        with card:
            self._render_param_header(field_name, label)
            sel = (
                ui.select(
                    options=options,
                    value=cur_val,
                    on_change=lambda e, fn=field_name: self._on_field_changed(fn, e.value),
                )
                .props("dense outlined options-dense")
                .classes("w-full text-xs bg-white")
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
        props_str = "dense outlined use-chips options-dense"
        card = ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]")
        self._cards[field_name] = card
        with card:
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
                .classes("w-full text-xs bg-white")
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
        card = ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]")
        self._cards[field_name] = card
        with card:
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
                .props("dense outlined")
                .classes("w-full text-xs font-mono bg-white")
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
        card = ui.column().classes("w-full gap-0.5 bg-white p-2 rounded border border-[#E2E8F0]")
        self._cards[field_name] = card
        with card:
            self._render_param_header(field_name, label)
            val_lbl = ui.label(f"{cur_val:.2f}").classes(
                "text-[11px] font-mono text-[#065F46]"
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
            ).props("dense color=primary")

        def _update(cfg: ToolConfig) -> None:
            val = float(getattr(cfg, field_name))
            sld.set_value(val)
            val_lbl.set_text(f"{val:.2f}")

        self._control_updaters.append(_update)

    def _add_checkbox(self, field_name: str, label: str) -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = bool(getattr(tc, field_name))
        card = ui.row().classes(
            "w-full items-center justify-between bg-white p-2 rounded border border-[#E2E8F0]"
        )
        self._cards[field_name] = card
        with card:
            chk = (
                ui.checkbox(
                    label,
                    value=cur_val,
                    on_change=lambda e, fn=field_name: self._on_field_changed(fn, bool(e.value)),
                )
                .props("dense color=primary")
                .classes("text-xs text-[#0F172A]")
            )
            badge = ui.badge("", color="primary").props("dense")
            badge.classes("text-[10px] px-1.5 py-0")
            badge.set_visibility(False)
            self._badges[field_name] = badge

        def _update(cfg: ToolConfig) -> None:
            chk.set_value(bool(getattr(cfg, field_name)))

        self._control_updaters.append(_update)

    def _add_text(self, field_name: str, label: str, placeholder: str = "") -> None:
        tc = resolve_tool_config(self.runtime.state)
        cur_val = str(getattr(tc, field_name) or "")
        card = ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]")
        self._cards[field_name] = card
        with card:
            self._render_param_header(field_name, label)
            inp = (
                ui.input(
                    value=cur_val,
                    placeholder=placeholder,
                    on_change=lambda e, fn=field_name: self._on_field_changed(
                        fn, str(e.value or "").strip()
                    ),
                )
                .props("dense outlined")
                .classes("w-full text-xs font-mono bg-white")
            )

        def _update(cfg: ToolConfig) -> None:
            inp.set_value(str(getattr(cfg, field_name) or ""))

        self._control_updaters.append(_update)

    # =========================================================================
    # Вкладки параметров с семантической группировкой (83 поля ToolConfig)
    # =========================================================================

    def _build_tab_curation(self) -> None:
        with ui.column().classes("w-full gap-2"):
            self._section_header("Фильтрация источника")
            self._add_multiselect(
                "allowed_bao_formats",
                "Допустимые протоколы BAO",
                _BAO_FORMAT_OPTIONS,
                allow_custom=True,
            )
            self._add_multiselect(
                "standard_types",
                "Типы биоактивности",
                _STANDARD_TYPE_OPTIONS,
                allow_custom=True,
            )
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "relation_mode",
                    "Цензурированные значения",
                    {"include_censored": "Включать (> / <)", "exact_only": "Только точные (=)"},
                )
                self._add_number("pchembl_min_cutoff", "Мин. порог pChEMBL", min_val=0.0, max_val=15.0, step=0.1)
                self._add_number("chembl_api_limit", "Лимит выборки ChEMBL", min_val=10, max_val=5000, step=10, is_int=True)
                self._add_number("min_neighborhood_rows_fallback", "Мин. строк окрестности", min_val=1, step=1, is_int=True)

            self._section_header("Статистическая агрегация")
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "duplicate_agg_method",
                    "Агрегация дубликатов",
                    ["median", "mean", "max"],
                )
                self._add_number("max_duplicate_std", "Макс. разброс дубликатов", min_val=0.0, step=0.05)
                self._add_select(
                    "scaffold_mode",
                    "Режим скаффолда",
                    {"bemis_murcko": "Bemis–Murcko", "generic_framework": "Обобщённый каркас"},
                )
                self._add_number("top_scaffolds_limit", "Лимит топ-скаффолдов", min_val=1, max_val=50, step=1, is_int=True)

            self._section_header("Стандартизация структур RDKit")
            self._add_checkbox("strip_salts", "Удалять соли и противоионы")
            self._add_checkbox("neutralize_charges", "Нейтрализовать заряды")
            self._add_checkbox("canonicalize_tautomers", "Канонизировать таутомеры")
            self._add_checkbox("keep_chirality", "Сохранять хиральность")

    def _build_tab_sar(self) -> None:
        with ui.column().classes("w-full gap-2"):
            self._section_header("Фрагментация rdMMPA и правила SAR")
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select("mmpa_max_cuts", "rdMMPA макс. разрезов", [1, 2, 3])
                self._add_number("mmpa_max_cut_bonds", "Макс. разрезаемых связей", min_val=1, step=1, is_int=True)
                self._add_number("mmpa_min_core_heavy_atoms", "Мин. тяж. атомов кора", min_val=1, step=1, is_int=True)
                self._add_number("min_support_pairs", "Мин. пар поддержки (Support)", min_val=1, step=1, is_int=True)
                self._add_number("min_mean_delta_pic50", "Мин. сдвиг ΔpIC50", step=0.05)
                self._add_number("max_rgroup_heavy_atoms", "Макс. тяж. атомов R-группы", min_val=1, step=1, is_int=True)
            self._add_checkbox("prioritize_positive_rules", "Приоритизировать правила с ΔpIC50 > 0")

            self._section_header("Молекулярные отпечатки и сходство")
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select("fp_type", "Тип отпечатков (fp_type)", ["ecfp", "fcfp", "rdkit", "maccs"])
                self._add_select("fp_radius", "Радиус фингерпринта", [1, 2, 3])
                self._add_select("fp_nbits", "Размерность битов (fp_nbits)", [1024, 2048, 4096])
                self._add_select("similarity_metric", "Метрика сходства", ["tanimoto", "dice", "cosine"])

            self._section_header("Детекция Activity Cliffs (SALI) и обогащение")
            self._add_slider("cliff_similarity_cutoff", "Порог сходства Activity Cliff", min_val=0.0, max_val=1.0, step=0.01)
            self._add_slider("cliff_delta_pic50_cutoff", "Порог ΔpIC50 Activity Cliff", min_val=0.0, max_val=5.0, step=0.1)
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_number("sali_epsilon", "SALI epsilon", min_val=0.001, step=0.01)
                self._add_number("min_rules_before_enrichment", "Мин. правил до обогащения", min_val=0, step=1, is_int=True)
            self._add_text("background_target_id", "Фоновая мишень ChEMBL (background_target_id)", "CHEMBL203")

    def _build_tab_design(self) -> None:
        with ui.column().classes("w-full gap-2"):
            self._section_header("Движок генерации и параметры CReM")
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

            self._section_header("Пул кандидатов и отбор")
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_number("max_parent_seeds", "Макс. родительских сидов", min_val=1, max_val=50, step=1, is_int=True)
                self._add_number("max_candidates_pool", "Макс. пул кандидатов", min_val=1, step=1, is_int=True)
                self._add_number("top_k_output", "Топ-K на выходе (top_k_output)", min_val=1, max_val=50, step=1, is_int=True)
            self._add_slider("neighborhood_tanimoto", "Порог окрестности Танимото", min_val=0.0, max_val=1.0, step=0.01)

            self._section_header("Ограничения каркаса и биоизостеры")
            self._add_checkbox("freeze_murcko_core", "Фиксировать каркас Bemis-Murcko (freeze_murcko_core)")
            self._add_checkbox("use_bioisostere_catalog", "Использовать каталог биоизостеров")

            tc = resolve_tool_config(self.runtime.state)
            ps_card = ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]")
            self._cards["protected_smarts"] = ps_card
            with ps_card:
                self._render_param_header("protected_smarts", "Защищённый SMARTS кора (protected_smarts)")
                self._protected_smarts_input = (
                    ui.input(
                        value=str(tc.protected_smarts or ""),
                        placeholder="Например: c1ncnc2ccccc12",
                        on_change=self._handle_protected_smarts_change,
                    )
                    .props("dense outlined clearable")
                    .classes("w-full text-xs font-mono bg-white")
                )
                self._protected_smarts_error = ui.label("").classes("text-[11px] text-rose-600")
                self._protected_smarts_error.set_visibility(False)
                with ui.row().classes("w-full gap-1.5"):
                    ui.button(
                        "Взять из скетчера",
                        icon="download",
                        on_click=self._pull_smarts_from_sketcher,
                    ).props("dense flat size=sm color=primary").classes("text-[11px]")
                    ui.button(
                        "Загрузить в скетчер",
                        icon="upload",
                        on_click=self._push_smarts_to_sketcher,
                    ).props("dense flat size=sm color=primary").classes("text-[11px]")

            ma_card = ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]")
            self._cards["modifiable_atom_indices"] = ma_card
            with ma_card:
                self._render_param_header("modifiable_atom_indices", "Модифицируемые индексы атомов (через запятую)")
                self._mod_atoms_input = (
                    ui.input(
                        value=format_modifiable_atom_indices(tc.modifiable_atom_indices),
                        placeholder="Например: 0, 1, 5 (пусто = без ограничений)",
                        on_change=self._handle_mod_atoms_change,
                    )
                    .props("dense outlined clearable")
                    .classes("w-full text-xs font-mono bg-white")
                )
                self._mod_atoms_error = ui.label("").classes("text-[11px] text-rose-600")
                self._mod_atoms_error.set_visibility(False)

    def _build_tab_evaluation(self) -> None:
        with ui.column().classes("w-full gap-2"):
            self._section_header("Структурные фильтры (PAINS / Brenk / SMARTS)")
            self._add_multiselect(
                "active_filter_catalogs",
                "Каталоги структурных фильтров",
                _FILTER_CATALOG_OPTIONS,
            )
            self._add_checkbox("brenk_ignore_long_chain", "Brenk: игнорировать Aliphatic_long_chain")

            ban_card = ui.column().classes("w-full gap-1.5 bg-white p-2 rounded border border-[#E2E8F0]")
            self._cards["custom_ban_smarts"] = ban_card
            with ban_card:
                self._render_param_header("custom_ban_smarts", "Пользовательские запрещённые SMARTS")
                with ui.row().classes("w-full items-center gap-1.5"):
                    self._ban_smarts_input = (
                        ui.input(placeholder="Введите SMARTS, напр. [N+](=O)[O-]")
                        .props("dense outlined")
                        .classes("flex-1 text-xs font-mono bg-white")
                    )
                    ui.button(
                        "Добавить",
                        on_click=self._handle_add_ban_smarts,
                    ).props("dense outline no-caps size=sm color=green-7").classes("text-xs")
                self._ban_smarts_error = ui.label("").classes("text-[11px] text-rose-600")
                self._ban_smarts_error.set_visibility(False)
                self._ban_smarts_chips_row = ui.row().classes("w-full gap-1 flex-wrap")

            self._section_header("Дескрипторы RDKit и фрагментация BRICS")
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

            self._section_header("Пороги Липински и ADMET-классификации")
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
            self._section_header("Профиль желательности Харрингтона (MPO)")
            mpo_card = ui.column().classes("w-full gap-1.5 bg-white p-2 rounded border border-[#E2E8F0]")
            self._cards["mpo_weights"] = mpo_card
            with mpo_card:
                self._render_param_header("mpo_weights", "Веса Харрингтона MPO (синхр. с графиком)")
                with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-1"):
                    for w_key, w_label in _MPO_WEIGHT_FIELDS:
                        w_val = float(tc.mpo_weights.get(w_key, 1.0))
                        with ui.column().classes("w-full gap-0"):
                            lbl = ui.label(f"{w_label}: {w_val:.2f}").classes(
                                "text-[11px] text-[#334155]"
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
                            ).props("dense color=primary")
                            self._mpo_weight_sliders[w_key] = (sld, lbl)

            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select(
                    "cns_profile_mode",
                    "Профиль CNS / TPSA",
                    ["peripheral_low_bbb", "cns_high_bbb", "neutral"],
                )
                self._add_select(
                    "alert_penalty_mode",
                    "Штраф за структурные фильтры",
                    ["veto", "soft_penalty", "ignore"],
                )
            self._add_slider(
                "alert_soft_multiplier",
                "Множитель мягкого штрафа",
                min_val=0.0,
                max_val=1.0,
                step=0.05,
            )

            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_number("harrington_activity_low_delta", "Харрингтон pIC50 нижн. Δ", step=0.05)
                self._add_number("harrington_activity_high_delta", "Харрингтон pIC50 верхн. Δ", step=0.05)
                self._add_number("harrington_qed_min", "Харрингтон QED мин.", min_val=0.0, max_val=1.0, step=0.05)
                self._add_number("harrington_qed_target", "Харрингтон QED цель", min_val=0.0, max_val=1.0, step=0.05)
                self._add_number("harrington_sa_min", "Харрингтон SAscore мин.", min_val=1.0, max_val=10.0, step=0.1)

            self._section_header("Генерация 3D-конформеров и PCA")
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_select("etkdg_variant", "3D генератор ETKDG", ["ETKDGv3", "ETKDGv2"])
                self._add_select("force_field", "Силовое поле 3D", ["MMFF94s", "MMFF94", "UFF", "none"])
                self._add_number("num_conformers", "Число 3D-конформеров", min_val=1, max_val=50, step=1, is_int=True)
                self._add_number("conformer_opt_max_iters", "Макс. итераций 3D-опт.", min_val=0, max_val=5000, step=25, is_int=True)
                self._add_number("conformer_random_seed", "Seed 3D-конформеров", step=1, is_int=True)
                self._add_number("pca_random_seed", "Seed PCA-проекции", step=1, is_int=True)

            self._section_header("Оркестрация и контроль выполнения")
            with ui.column().classes("w-full gap-1 bg-white p-2 rounded border border-[#E2E8F0]"):
                ui.label("Режим паузы с участием исследователя").classes(
                    "text-xs font-medium text-[#0F172A]"
                )
                self._hitl_select = (
                    ui.select(
                        options={
                            "auto": "Автоматический проход",
                            "pause_after_scaffolds": "Пауза после анализа SAR",
                            "pause_before_critic": "Пауза перед верификацией",
                        },
                        value=self.runtime.hitl_mode,
                        on_change=self._handle_hitl_change,
                    )
                    .props("dense outlined options-dense")
                    .classes("w-full text-xs bg-white")
                )

            self._section_header("Петли самокоррекции и верификация")
            with ui.grid(columns=2).classes("w-full gap-2"):
                self._add_number("max_iterations", "Макс. итераций генерации", min_val=0, max_val=10, step=1, is_int=True)
                self._add_number("max_refilter_iterations", "Макс. итераций перефильтрации BAO", min_val=0, max_val=5, step=1, is_int=True)
                self._add_number("critic_fallback_min_tpsa", "Резервный порог TPSA", min_val=0.0, step=1.0)
                self._add_number("critic_logp_step_reduction", "Шаг снижения cLogP", min_val=0.0, step=0.1)

            self._add_checkbox("strict_assay_homogeneity", "Строгий аудит однородности BAO (Петля 1)")
            self._add_checkbox("strict_property_thresholds", "Строгая проверка порогов свойств (Петля 2)")

    # =========================================================================
    # Обработка изменений, валидация, синхронизация и контекстный Sticky Footer
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
                ui.label("Список пуст").classes("text-[11px] text-[#64748B]")
            for sm in items:
                ui.chip(
                    sm,
                    removable=True,
                    color="red-7",
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

    async def _handle_primary_cta(self) -> None:
        mode, _, _, _ = resolve_primary_cta_spec(self._pending_keys)
        if mode == "instant_mpo":
            await self._handle_apply_instant_mpo()
        else:
            await self._handle_restart_stage(mode)  # type: ignore[arg-type]

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
        ui.notify("Параметры сброшены по умолчанию", type="info")

    def _handle_hitl_change(self, e: Any) -> None:
        if self._syncing:
            return
        val = str(e.value or "auto")
        if val in {"auto", "hitl_all", "hitl_custom"}:
            self.runtime.hitl_mode = val  # type: ignore[assignment]

    def _refresh_badges_and_impact(self) -> None:
        sources: dict[str, ParamSourceLiteral] = dict(
            self.runtime.state.get("param_sources", {})
        )
        critic_reason = str(
            self.runtime.state.get("critic_feedback")
            or "Параметр автоматически скорректирован модулем верификации"
        )

        for field_name, badge in self._badges.items():
            src = sources.get(field_name, "default")
            card = self._cards.get(field_name)
            if src == "default":
                badge.set_visibility(False)
                if card is not None:
                    card.classes(
                        remove="border-[#F59E0B] bg-[#FFFBEB] border-[#8B495C] bg-[#FBF5F7] border-[#6D597A] bg-[#F6F3F8]",
                        add="border-[#E2E8F0] bg-white",
                    )
            else:
                text, color = _DRAWER_BADGE_META.get(src, ("Изменено", "primary"))
                badge.set_text(text)
                badge.props(f"color={color}")
                badge.set_visibility(True)
                if src == "critic_loop":
                    badge.tooltip(critic_reason)
                    if card is not None:
                        card.classes(
                            remove="border-[#E2E8F0] bg-white border-[#8B495C] bg-[#FBF5F7] border-[#6D597A] bg-[#F6F3F8]",
                            add="border-[#F59E0B] bg-[#FFFBEB]",
                        )
                elif src == "ui_manual":
                    badge.tooltip("Изменено пользователем вручную")
                    if card is not None:
                        card.classes(
                            remove="border-[#E2E8F0] bg-white border-[#F59E0B] bg-[#FFFBEB] border-[#6D597A] bg-[#F6F3F8]",
                            add="border-[#8B495C] bg-[#FBF5F7]",
                        )
                else:
                    badge.tooltip("Извлечено из исследовательского запроса")
                    if card is not None:
                        card.classes(
                            remove="border-[#E2E8F0] bg-white border-[#F59E0B] bg-[#FFFBEB] border-[#8B495C] bg-[#FBF5F7]",
                            add="border-[#6D597A] bg-[#F6F3F8]",
                        )

        mode, cta_label, _cta_icon, cta_color = resolve_primary_cta_spec(self._pending_keys)
        if self._primary_cta_btn is not None:
            self._primary_cta_btn.set_text(cta_label)
            self._primary_cta_btn.props(f"color={cta_color}")

        if self._impact_label is not None:
            if not self._pending_keys:
                self._impact_label.set_text(
                    "Параметры синхронизированы с текущей сессией"
                )
            else:
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
            if self._hitl_select is not None:
                self._hitl_select.set_value(self.runtime.hitl_mode)
            self._refresh_badges_and_impact()
        finally:
            self._syncing = False
