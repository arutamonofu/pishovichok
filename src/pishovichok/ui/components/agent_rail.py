"""Верхняя статусная панель и левый рельс оркестрации агентов (Clean Wet-Lab Light Mode)."""

from __future__ import annotations

import asyncio
import html
import json
import re
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Any

from nicegui import run, ui

from pishovichok.agents.workflow import (
    export_candidates_to_csv,
    export_candidates_to_sdf,
)
from pishovichok.config import settings
from pishovichok.state.graph_state import resolve_tool_config
from pishovichok.state.schemas import (
    ParamSourceLiteral,
    TraceEvent,
)
from pishovichok.ui.bridge import agent_bridge
from pishovichok.ui.icons import svg_icon
from pishovichok.ui.session import SessionRuntime, session_registry
from pishovichok.ui.state_vm import PARAM_SOURCE_BADGE_META

DEMO_SCENARIOS: dict[str, dict[str, str]] = {
    "scenario1_btk": {
        "title": "Сценарий 1: BTK / Ибрутиниб (ChEMBL5251)",
        "scenario_type": "target_based_sar",
        "query": "Найди наиболее активные скаффолды и SAR-правила для мишени CHEMBL5251 (BTK)",
        "lead_smiles": "",
    },
    "scenario2_gefitinib": {
        "title": "Сценарий 2: Гефитиниб / EGFR (ChEMBL203)",
        "scenario_type": "lead_optimization",
        "query": "Оптимизируй Гефитиниб для мишени CHEMBL203 (EGFR) с logP <= 4.0 и без PAINS",
        "lead_smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    },
    "scenario3_wetlab": {
        "title": "Сценарий 3: Локальная серия (.csv)",
        "scenario_type": "custom_csv_analysis",
        "query": "Проанализируй лабораторную серию из CSV и предложи безопасные биоизостеры",
        "lead_smiles": "COc1ccc(Nc2ncnc3ccccc23)cc1",
    },
}

_AGENT_TO_STAGE: dict[str, str] = {
    "ResearchAgent": "research",
    "SARAgent": "sar",
    "DesignAgent": "design",
    "EvaluationAgent": "evaluation",
    "CriticAgent": "critic",
}


def resolve_stage_actor_meta(stage_key: str) -> dict[str, str]:
    """Возвращает метаданные актора (ИИ-агент или вычислительный инструмент) по ключу этапа."""
    mapping: dict[str, dict[str, str]] = {
        "research": {
            "name": "Исследователь-агент",
            "kind": "agent",
            "kind_label": "ИИ-агент",
            "role": "Постановка задачи и курация",
            "icon": "bot",
        },
        "sar": {
            "name": "SAR-анализатор (rdMMPA / SALI)",
            "kind": "tool",
            "kind_label": "Инструмент",
            "role": "Анализ пар MMP и перепадов",
            "icon": "wrench",
        },
        "design": {
            "name": "Генератор (CReM / GraphGA)",
            "kind": "tool",
            "kind_label": "Инструмент",
            "role": "Комбинаторный дизайн структур",
            "icon": "wrench",
        },
        "evaluation": {
            "name": "Оценщик-агент",
            "kind": "agent",
            "kind_label": "ИИ-агент",
            "role": "ADMET и синтез гипотез",
            "icon": "bot",
        },
        "docking": {
            "name": "3D-докинг (AutoDock Vina)",
            "kind": "tool",
            "kind_label": "Инструмент",
            "role": "Физический расчёт связывания",
            "icon": "cube",
        },
        "critic": {
            "name": "Критик-агент",
            "kind": "agent",
            "kind_label": "ИИ-агент",
            "role": "Аудит и верификация",
            "icon": "bot",
        },
    }
    return mapping.get(
        stage_key,
        {
            "name": stage_key,
            "kind": "tool",
            "kind_label": "Инструмент",
            "role": "Вычислительный узел",
            "icon": "wrench",
        },
    )


def resolve_trace_actor_meta(ev: TraceEvent) -> dict[str, str]:
    """Определяет отображаемое имя и тип (ИИ-агент vs Инструмент) для события `TraceEvent`."""
    if ev.phase and "docking" in ev.phase:
        return resolve_stage_actor_meta("docking")
    stage_key = _AGENT_TO_STAGE.get(ev.agent_name, ev.agent_name.lower())
    return resolve_stage_actor_meta(stage_key)

_COMMON_FRAGMENT_NAMES: dict[str, str] = {
    "C": "Me",
    "CC": "Et",
    "OC": "OMe",
    "OCC": "OEt",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "C#N": "CN",
    "C(F)(F)F": "CF₃",
    "C(=O)NC": "CONHMe",
    "C(=O)N": "CONH₂",
    "OCCO": "OCH₂CH₂OH",
    "S(=O)(=O)C": "SO₂Me",
    "[H]": "H",
}

_TRANSFORMATION_TYPE_RU: dict[str, str] = {
    "any": "любые",
    "polar": "полярные",
    "less_lipophilic": "менее липофильные",
    "exclude_halogens": "без галогенов",
    "bioisostere": "биоизостеры",
    "hbd_reduction": "снижение доноров H-связей",
    "mw_reduction": "снижение мол. массы",
    "aromatic_reduction": "снижение ароматичности",
    "rigidify": "повышение жёсткости",
    "solubility": "повышение растворимости",
}

_ACTION_TRANSLATIONS: dict[str, str] = {
    "Parsed user constraints and performed initial standardization and SMILES curation.": (
        "Сформулированы ограничения поиска, выполнена стандартизация и курация SMILES."
    ),
    "Executed dual-engine candidate generation (rdMMPA/SMIRKS transformations and CReM bioisosteric replacements).": (
        "Выполнена двухдвижковая генерация кандидатов (трансформации rdMMPA/SMIRKS и биоизостерные замены CReM)."
    ),
    "Calculated RDKit 2D metrics, PAINS/Brenk alerts, BRICS precursors, ADMET probabilities, and Harrington MPO desirability index.": (
        "Рассчитаны 2D-дескрипторы RDKit, алерты PAINS/Brenk, прекурсоры BRICS, вероятности ADMET и интегральный индекс Харрингтона."
    ),
    "Resolved receptor pocket (cache/RCSB PDB) and executed 3D docking + interaction profiling.": (
        "Подготовлен карман рецептора (кэш/RCSB PDB), выполнен 3D-докинг и профилирование взаимодействий."
    ),
    "Executed standalone physical 3D docking and updated Harrington MPO scores.": (
        "Выполнен автономный 3D-докинг и обновлены оценки Харрингтона."
    ),
    "Skipped automatic 3D docking according to docking_mode.": (
        "Автоматический 3D-докинг пропущен согласно настройкам режима."
    ),
    "Confirmed assay protocol homogeneity. Routed workflow to DesignAgent.": (
        "Однородность экспериментальных протоколов подтверждена. Переход к этапу генерации."
    ),
    "Approved candidates satisfying all user constraints. Synthesized final report.": (
        "Кандидаты удовлетворяют всем ограничениям. Сформирован итоговый аналитический отчёт."
    ),
}


def _translate_preferred_types(pref_str: str) -> str:
    """Преобразует список типов трансформаций из Python-представления в читаемый русский текст."""
    cleaned = pref_str.strip("[]").replace("'", "").replace('"', "")
    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
    if not tokens:
        return "любые"
    ru_tokens = [_TRANSFORMATION_TYPE_RU.get(t, t) for t in tokens]
    return ", ".join(ru_tokens)


def _format_readable_fragment(frag: str) -> str:
    """Преобразует фрагмент SMILES с точкой крепления `[*:1]` в читаемое обозначение заместителя."""
    cleaned = frag.replace("[*:1]", "").replace("*", "").strip()
    if not cleaned:
        return "H"
    return _COMMON_FRAGMENT_NAMES.get(cleaned, cleaned)


def _compact_telemetry_val(val: Any) -> str:
    """Компактно форматирует значение параметра или счётчика для чипов телеметрии."""
    if isinstance(val, list):
        if len(val) > 3:
            return f"[{', '.join(str(x) for x in val[:2])}, +{len(val) - 2}]"
        return f"[{', '.join(str(x) for x in val)}]"
    if isinstance(val, dict):
        return f"{{{len(val)}}}"
    if isinstance(val, float):
        return f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
    return str(val)


def format_node_telemetry_chips(
    telemetry: dict[str, Any] | None,
    param_sources: dict[str, ParamSourceLiteral] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    """Преобразует `node_telemetry[node]` в структурированные чипы параметров и счётчиков."""
    if not isinstance(telemetry, dict):
        return [], []
    sources = param_sources or {}
    raw_params = telemetry.get("active_params") or {}
    raw_stats = telemetry.get("stats") or {}

    param_chips: list[dict[str, str]] = []
    if isinstance(raw_params, dict):
        for k, v in raw_params.items():
            if v is None or v == [] or v == "":
                continue
            src: ParamSourceLiteral = sources.get(str(k), "default")
            _, color = PARAM_SOURCE_BADGE_META.get(src, ("", "slate"))
            param_chips.append(
                {
                    "key": str(k),
                    "label": f"{k}: {_compact_telemetry_val(v)}",
                    "source": src,
                    "color": color,
                }
            )

    stat_chips: list[tuple[str, str]] = []
    if isinstance(raw_stats, dict):
        for k, v in raw_stats.items():
            if k == "duration_ms" or v is None or v == []:
                continue
            stat_chips.append((str(k), _compact_telemetry_val(v)))

    return param_chips, stat_chips


def _localize_trace_summary(raw_text: str) -> str:
    """Переводит шаблонные англоязычные сводки узлов графа на русский язык для UI."""
    text = (raw_text or "").strip()
    if not text:
        return ""

    # Курация
    m_cur = re.match(
        r"^Curated\s+(\d+)\s+molecules\s+\(dropped\s+(\d+)\s+invalid,\s+(\d+)\s+censored,\s+(\d+)\s+duplicates\)\.\s*Dominant BAO:\s*(.+?)\s*\(homogeneous=(True|False)\)\.?$",
        text,
    )
    if m_cur:
        n_mol, n_inv, n_cens, n_dup, bao, homog = m_cur.groups()
        homog_ru = "да" if homog == "True" else "нет"
        return (
            f"Отобрано {n_mol} молекул (отсеяно: {n_inv} невал., {n_cens} цензур., {n_dup} дубл.). "
            f"Протокол BAO: {bao} (однородный: {homog_ru})."
        )
    m_cur_short = re.match(r"^Curated\s+(\d+)\s+molecules\.?$", text)
    if m_cur_short:
        return f"Отобрано {m_cur_short.group(1)} молекул."

    # Анализ SAR
    m_sar = re.match(
        r"^Extracted\s+(\d+)\s+SAR rules and\s+(\d+)\s+Activity Cliffs\.\s*Homogeneous rules:\s*(\d+/\d+)\s*\(background_enriched=(True|False)\)\.?$",
        text,
    )
    if m_sar:
        n_rules, n_cliffs, homog_ratio, enriched = m_sar.groups()
        enr_ru = "да" if enriched == "True" else "нет"
        return (
            f"Извлечено правил SAR: {n_rules}, перепадов активности: {n_cliffs}. "
            f"Однородных правил: {homog_ratio} (обогащение фоном: {enr_ru})."
        )

    # Генерация
    m_gen = re.match(
        r"^Generated\s+(\d+)\s+candidate structures from\s+(\d+)\s+parent seeds\s*\((.*?)\)\.?$",
        text,
    )
    if m_gen:
        n_cand, n_seeds, meta_str = m_gen.groups()
        it_m = re.search(r"iteration=(\d+)", meta_str)
        it_num = it_m.group(1) if it_m else "1"

        methods_parts: list[str] = []
        for key in ("SMIRKS", "CReM", "GraphGA"):
            km = re.search(rf"\b{key}=(\d+)", meta_str)
            if km:
                methods_parts.append(f"{key}: {km.group(1)}")
        methods_str = f", {', '.join(methods_parts)}" if methods_parts else ""

        pref_m = re.search(r"preferred_types=(.+)$", meta_str)
        pref_ru = _translate_preferred_types(pref_m.group(1)) if pref_m else "любые"

        return (
            f"Сгенерировано {n_cand} структур-кандидатов из {n_seeds} прототипов "
            f"(итерация {it_num}{methods_str}, типы: {pref_ru})."
        )

    # Оценка ADMET
    m_eval = re.match(
        r"^Evaluated\s+(\d+)\s+candidates;\s*top-(\d+)\s+selected\s*\(max Harrington D=([\d.]+)\)\.\s*Constraints fully satisfied:\s*(\d+/\d+)\.?$",
        text,
    )
    if m_eval:
        n_ev, n_top, max_d, sat = m_eval.groups()
        return (
            f"Оценено кандидатов: {n_ev}; выбрано топ-{n_top} (макс. D Харрингтона = {max_d}). "
            f"Ограничения соблюдены: {sat}."
        )

    # 3D-докинг
    m_dock = re.match(
        r"^status=([A-Za-z0-9_]+)\s*\|\s*pdb=([^|]+?)(?:\s*\(([^)]*)\))?\s*\|\s*docked=(\d+/\d+)\s*\|\s*best\s*ΔG=(.+?)\.?$",
        text,
    )
    if m_dock:
        st, pdb, src, docked, dg = m_dock.groups()
        st_map = {
            "SUCCESS": "Успешно",
            "SKIPPED": "Пропущено",
            "FAILED": "Ошибка",
            "NO_RECEPTOR": "Рецептор не найден",
        }
        st_ru = st_map.get(st, st)

        src_clean = (src or "").strip()
        src_map = {
            "DEMO_CACHE": "демо-кэш",
            "RCSB_LIVE": "RCSB PDB",
            "RCSB_CACHE": "RCSB кэш",
            "LOCAL_PDB": "локальный PDB",
            "None": "",
            "N/A": "",
        }
        src_ru = src_map.get(src_clean, src_clean)

        pdb_clean = pdb.strip()
        if pdb_clean in ("None", "N/A", "-"):
            pdb_str = "нет"
        elif src_ru:
            pdb_str = f"{pdb_clean} ({src_ru})"
        else:
            pdb_str = pdb_clean

        dg_clean = dg.strip().replace("kcal/mol", "ккал/моль")
        if dg_clean in ("None", "N/A", "-"):
            dg_str = "Н/Д"
        else:
            dg_str = dg_clean

        return f"Статус: {st_ru} | PDB: {pdb_str} | состыковано: {docked} | лучшая ΔG = {dg_str}."

    if text in _ACTION_TRANSLATIONS:
        return _ACTION_TRANSLATIONS[text]

    return text.replace("Critic Agent:", "Верификация:").replace("Critic Agent", "Верификатор")


class TopStatusBar:
    """Верхняя панель: бренд, переключатель глубины, новая сессия и экспорт."""

    def __init__(
        self,
        runtime: SessionRuntime,
        *,
        on_session_switched: Callable[[str], Any] | None = None,
        on_depth_mode_changed: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self._on_session_switched = on_session_switched
        self._on_depth_mode_changed = on_depth_mode_changed
        self._export_row: ui.row | None = None
        self._clear_btn: ui.button | None = None
        self._depth_toggle: ui.toggle | None = None
        self._build()

    def _build(self) -> None:
        with ui.header(elevated=False).classes(
            "bg-white border-t-[3px] border-t-[#8B495C] border-b border-[#E2DCE0] text-[#0F172A] px-3 py-1 items-center justify-between gap-2 h-[52px]"
        ):
            with ui.row().classes("items-center gap-2 no-wrap min-w-0"):
                ui.html(
                    '<div class="w-8 h-8 rounded-full bg-[#FBF5F7] border border-[#E5CFD6] overflow-hidden flex items-center justify-center shrink-0">'
                    '<img src="/static/mascot_head.png" alt="Пишовичок" class="w-7 h-7 object-contain" />'
                    "</div>",
                    sanitize=False,
                )
                ui.label("ПИШОВИЧОК").classes(
                    "text-sm font-semibold tracking-tight text-[#5C2D3B] whitespace-nowrap"
                )

            with ui.row().classes("items-center gap-1.5 no-wrap"):
                self._depth_toggle = (
                    ui.toggle(
                        {"lab": "Простой", "expert": "Продвинутый"},
                        value=self.runtime.depth_mode,
                        on_change=self._handle_depth_mode_change,
                    )
                    .props("dense no-caps toggle-color=primary color=grey-2 text-color=grey-9 size=sm")
                    .classes("border border-[#D5C8CD] rounded compact-toggle")
                )

                self._clear_btn = (
                    ui.button(on_click=self._handle_clear_session)
                    .props("dense outline no-caps color=grey-8 size=sm")
                    .classes("text-[11px] bg-white border-[#CBD5E1] px-2 gap-1")
                    .tooltip("Очистить рабочую область и начать новую сессию")
                )
                self._clear_btn._props["title"] = "Новая сессия"
                with self._clear_btn:
                    ui.html(svg_icon("clear", 13), sanitize=False)
                    ui.label("Новая сессия").classes("text-[11px] text-[#1E293B]")

                self._export_row = ui.row().classes("items-center gap-1.5 no-wrap")
                with self._export_row:
                    with (
                        ui.button()
                        .props("dense outline no-caps size=sm color=grey-8")
                        .classes("text-[11px] bg-white border-[#CBD5E1] px-2.5 gap-1")
                        .tooltip("Выгрузить результаты и протокол расчёта")
                    ):
                        ui.html(svg_icon("download", 13), sanitize=False)
                        ui.label("Экспорт").classes("text-[11px] text-[#1E293B]")
                        ui.html(svg_icon("chevron_down", 12), sanitize=False)
                        with ui.menu().props("anchor='bottom right' self='top right'").classes(
                            "bg-white border border-[#D5C8CD] shadow-lg rounded-lg py-1 min-w-[210px]"
                        ):
                            ui.menu_item(
                                "Таблица кандидатов (.csv)",
                                on_click=self._export_csv,
                            ).classes("text-xs text-[#1F181B]")
                            ui.menu_item(
                                "Структуры кандидатов (.sdf)",
                                on_click=self._export_sdf,
                            ).classes("text-xs text-[#1F181B]")
                            ui.separator()
                            ui.menu_item(
                                "Протокол верификации (.json)",
                                on_click=self._export_trace,
                            ).classes("text-xs text-[#1F181B]")
                            ui.menu_item(
                                "Научный отчёт (.md)",
                                on_click=self._export_report,
                            ).classes("text-xs text-[#1F181B]")

        self.refresh()

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

    async def _export_trace(self) -> None:
        traces = [ev.model_dump(mode="json") for ev in self.runtime.state.get("trace_log", [])]
        payload = json.dumps(traces, ensure_ascii=False, indent=2).encode("utf-8")
        ui.download(payload, filename=f"{self.runtime.session_id}_protocol.json")

    async def _export_report(self) -> None:
        report_md = str(
            self.runtime.state.get("final_report_markdown") or "# Отчёт ещё не сформирован"
        )
        ui.download(
            report_md.encode("utf-8"),
            filename=f"{self.runtime.session_id}_report.md",
        )

    def _handle_depth_mode_change(self, e: Any) -> None:
        val = str(e.value)
        if val in ("lab", "expert"):
            self.runtime.depth_mode = val  # type: ignore[assignment]
            if self._on_depth_mode_changed is not None:
                self._on_depth_mode_changed(val)

    async def _handle_clear_session(self) -> None:
        await agent_bridge.cancel_current_run(self.runtime.session_id)
        new_sid = f"sess_{uuid.uuid4().hex[:8]}"
        session_registry.get_or_create(session_id=new_sid)
        if self._on_session_switched is not None:
            res = self._on_session_switched(new_sid)
            if asyncio.iscoroutine(res):
                await res

    def refresh(self) -> None:
        """Обновляет видимость элементов шапки."""
        is_clean = self.runtime.is_clean_welcome
        if self._export_row is not None:
            self._export_row.set_visibility(not is_clean)
        if self._clear_btn is not None:
            self._clear_btn.set_visibility(not is_clean)
        if self._depth_toggle is not None and self._depth_toggle.value != self.runtime.depth_mode:
            self._depth_toggle.set_value(self.runtime.depth_mode)


def format_pishovichok_journal_summary(runtime: SessionRuntime) -> tuple[str, str]:
    """Формирует краткую 1-строчную сводку верификации и тон индикатора для островка Пишовичка.

    Возвращает `(summary_text, tone)`, где `tone` in `{"approved", "warning", "error", "neutral"}`.
    """
    traces: list[TraceEvent] = list(runtime.state.get("trace_log", []))
    if not traces:
        return ("", "neutral")

    total_ms = sum(float(getattr(ev, "duration_ms", 0.0) or 0.0) for ev in traces)
    total_sec = total_ms / 1000.0
    n_steps = len(traces)
    loops_count = sum(
        1
        for ev in traces
        if ev.verdict_or_routing in ("refilter_data", "redesign_molecules")
    )

    last_critic = next(
        (ev for ev in reversed(traces) if ev.agent_name == "CriticAgent"),
        None,
    )
    tone = "neutral"
    verdict_part = ""
    if runtime.status == "error":
        tone = "error"
        verdict_part = "Ошибка расчёта"
    elif last_critic is not None and last_critic.verdict_or_routing == "approve":
        tone = "warning" if loops_count > 0 else "approved"
        m_val = re.search(r"\(валидных:\s*(\d+/\d+)\)", last_critic.output_summary or "")
        verdict_part = f"Утверждено ({m_val.group(1)})" if m_val else "Утверждено"
    elif last_critic is not None and last_critic.verdict_or_routing in (
        "refilter_data",
        "redesign_molecules",
    ):
        tone = "warning"
        verdict_part = (
            "Возврат: перефильтрация BAO"
            if last_critic.verdict_or_routing == "refilter_data"
            else "Возврат: повторная генерация"
        )
    elif runtime.status == "cancelled":
        tone = "warning"
        verdict_part = "Остановлено пользователем"
    else:
        last_ev = traces[-1]
        verdict_part = _localize_trace_summary(
            last_ev.output_summary or last_ev.action_taken
        )

    parts = [verdict_part, f"шагов: {n_steps}"]
    if loops_count > 0:
        parts.append(f"возвратов: {loops_count}")
    if total_sec >= 0.1:
        parts.append(f"{total_sec:.1f} с")
    return (" · ".join(p for p in parts if p), tone)


def render_sar_rules_panel(
    runtime: SessionRuntime,
    *,
    scaffolds_section: ui.element | None,
    scaffolds_box: ui.element | None,
    sar_header_label: ui.label | None,
    sar_box: ui.element | None,
    on_scaffold_selected: Callable[[str], Any] | None = None,
) -> None:
    """Рендерит чипы базовых скаффолдов Bemis–Murcko и карточки правил SAR (MMP)."""
    summary = runtime.state.get("curation_summary")
    scaffolds = getattr(summary, "top_scaffolds", []) if summary else []
    if scaffolds_section is not None:
        scaffolds_section.set_visibility(bool(scaffolds))
    if scaffolds_box is not None:
        scaffolds_box.clear()
        if scaffolds:
            with scaffolds_box:
                for idx, scaf in enumerate(scaffolds[:6]):
                    smi = str(scaf.get("scaffold_smiles", ""))
                    cnt = scaf.get("count", 0)
                    label = f"S{idx + 1} (n={cnt})"
                    ui.button(
                        label,
                        on_click=lambda s=smi: (
                            on_scaffold_selected(s)
                            if on_scaffold_selected is not None
                            else None
                        ),
                    ).props("dense outline size=xs color=primary").classes(
                        "text-[11px] font-mono bg-[#FBF5F7] px-1.5 py-0.5"
                    ).tooltip(f"{smi} — загрузить в редактор")

    if sar_box is None:
        return
    sar_box.clear()
    raw_rules = list(runtime.state.get("extracted_sar_rules", []))
    rules = sorted(
        raw_rules,
        key=lambda r: float(getattr(r, "mean_delta_pic50", 0.0)),
        reverse=True,
    )

    bao_counts: Counter[str] = Counter()
    for r in rules:
        for b in getattr(r, "participating_bao_formats", []) or []:
            if b and b != "USER_CONSTRAINT":
                bao_counts[str(b)] += 1
    common_bao = bao_counts.most_common(1)[0][0] if bao_counts else ""
    if sar_header_label is not None:
        count_txt = f" ({len(rules)})" if rules else ""
        sar_header_label.set_text(
            f"Правила SAR (MMP){count_txt} · {common_bao}"
            if common_bao
            else f"Правила SAR (MMP){count_txt}"
        )

    sid = runtime.session_id
    is_expert = runtime.depth_mode == "expert"
    with sar_box:
        if not rules:
            ui.label("Правила SAR ещё не извлечены").classes(
                "text-xs text-[#475569] col-span-full"
            )
            return
        for rule in rules[:12]:
            rule_id = str(getattr(rule, "rule_id", "R?"))
            delta = float(getattr(rule, "mean_delta_pic50", 0.0))
            pairs = int(getattr(rule, "support_pairs_count", 0))
            homog = bool(getattr(rule, "assay_homogeneous", False))
            from_readable = _format_readable_fragment(str(getattr(rule, "from_fragment", "")))
            to_readable = _format_readable_fragment(str(getattr(rule, "to_fragment", "")))
            smirks_raw = str(getattr(rule, "smirks", ""))

            delta_cls = (
                "bg-[#ECFDF5] text-[#065F46] border-[#A7F3D0]"
                if delta > 0
                else "bg-[#FFF1F2] text-[#9F1239] border-[#FECDD3]"
            )

            with ui.card().classes(
                "w-full bg-white border border-[#E2E8F0] border-l-[3px] border-l-[#2563EB] p-2 gap-1 shadow-none hover:bg-[#F8FAFC] transition-colors"
            ) as card:
                with ui.row().classes("w-full items-center justify-between gap-1 no-wrap"):
                    ui.label(f"{rule_id}: {from_readable} → {to_readable}").classes(
                        "text-xs font-semibold text-[#0F172A] truncate"
                    )
                    with ui.row().classes("items-center gap-1 shrink-0"):
                        if not homog:
                            ui.label("Неоднородно").classes(
                                "text-[10px] font-medium px-1 py-0 rounded border bg-[#FFFBEB] text-[#78350F] border-[#FDE68A]"
                            )
                        ui.label(f"ΔpIC50: {delta:+.2f}").classes(
                            f"text-[10px] font-semibold px-1.5 py-0 rounded border {delta_cls}"
                        )
                        ui.label(f"N={pairs}").classes(
                            "text-[10px] font-medium px-1.5 py-0 rounded border bg-[#EFF6FF] text-[#1E3A8A] border-[#BFDBFE]"
                        )
                if is_expert and smirks_raw:
                    ui.label(smirks_raw).classes(
                        "text-[10px] mono-code px-1.5 py-0.5 truncate w-full"
                    )

                with card, ui.tooltip().classes("bg-white border border-[#CBD5E1] shadow-md p-1.5"):
                    ui.html(
                        f'<img src="/api/sessions/{html.escape(sid)}/rule/{html.escape(rule_id)}/svg" '
                        f'style="width:300px;height:105px;display:block;background:#FFFFFF;" />',
                        sanitize=False,
                    )


def render_verification_journal(
    runtime: SessionRuntime,
    *,
    timeline_box: ui.element | None,
    state_spoiler_box: ui.element | None,
) -> None:
    """Рендерит хронологическую ленту верификации и спойлер диагностики состояния."""
    if timeline_box is not None:
        timeline_box.clear()
        traces: list[TraceEvent] = list(runtime.state.get("trace_log", []))
        node_telemetry = runtime.state.get("node_telemetry", {})
        param_sources = runtime.state.get("param_sources", {})
        is_expert = runtime.depth_mode == "expert"
        is_running = runtime.status == "running" or runtime.is_docking_running

        with timeline_box:
            if not traces and not is_running:
                ui.label("Ожидание запуска расчёта...").classes(
                    "text-xs text-[#475569]"
                )
            else:
                iter_groups: list[tuple[str, list[TraceEvent]]] = []
                cur_iter_idx = 0
                cur_label = "Итерация 1 · Основной проход"
                cur_list: list[TraceEvent] = []
                for ev in traces:
                    if ev.verdict_or_routing in ("refilter_data", "redesign_molecules"):
                        cur_list.append(ev)
                        iter_groups.append((cur_label, cur_list))
                        cur_iter_idx += 1
                        reason_lbl = (
                            "Перефильтрация BAO"
                            if ev.verdict_or_routing == "refilter_data"
                            else "Повторная генерация"
                        )
                        cur_label = f"Итерация {cur_iter_idx + 1} · {reason_lbl}"
                        cur_list = []
                    else:
                        cur_list.append(ev)
                if cur_list:
                    iter_groups.append((cur_label, cur_list))

                for group_title, group_events in iter_groups:
                    ui.label(group_title).classes(
                        "text-[11px] font-semibold text-[#475569] bg-[#F1F5F9] px-2 py-1 rounded border border-[#E2E8F0]"
                    )
                    with ui.column().classes(
                        "w-full gap-1.5 pl-2.5 ml-1 border-l-2 border-[#CBD5E1]"
                    ):
                        for ev in group_events:
                            actor = resolve_trace_actor_meta(ev)
                            is_agent = actor["kind"] == "agent"
                            is_critic = ev.agent_name == "CriticAgent"
                            is_loop_back = is_critic and ev.verdict_or_routing in (
                                "refilter_data",
                                "redesign_molecules",
                            )
                            is_approve = is_critic and ev.verdict_or_routing == "approve"
                            verdict_chip: tuple[str, str] | None = None

                            if is_loop_back:
                                card_cls = "w-full bg-[#FFFBEB] border border-[#F59E0B] border-l-[3px] border-l-[#D97706] p-2 gap-1 shadow-none"
                                text_cls = "text-xs text-[#78350F]"
                                badge_cls = "bg-[#D97706] text-white"
                                loop_ru = (
                                    "Перефильтрация BAO"
                                    if ev.verdict_or_routing == "refilter_data"
                                    else "Повторная генерация"
                                )
                                verdict_chip = (
                                    f"Возврат: {loop_ru}",
                                    "bg-[#FEF3C7] text-[#92400E] border border-[#F59E0B]",
                                )
                            elif is_approve:
                                card_cls = "w-full bg-[#ECFDF5] border border-[#059669] border-l-[3px] border-l-[#059669] p-2 gap-1 shadow-none"
                                text_cls = "text-xs text-[#065F46]"
                                badge_cls = "bg-[#059669] text-white"
                                m_val = re.search(
                                    r"\(валидных:\s*(\d+/\d+)\)",
                                    ev.output_summary or "",
                                )
                                verdict_txt = (
                                    f"Утверждено ({m_val.group(1)})"
                                    if m_val
                                    else "Утверждено"
                                )
                                verdict_chip = (
                                    verdict_txt,
                                    "bg-[#D1FAE5] text-[#065F46] border border-[#34D399]",
                                )
                            elif is_agent:
                                card_cls = "w-full bg-white border border-[#E5CFD6] border-l-[3px] border-l-[#8B495C] p-2 gap-1 shadow-none"
                                text_cls = "text-xs text-[#1F181B]"
                                badge_cls = (
                                    "bg-[#D97706] text-white"
                                    if ev.fallback_triggered
                                    else "bg-[#8B495C] text-white"
                                )
                                if is_critic and ev.verdict_or_routing == "proceed_to_design":
                                    verdict_chip = (
                                        "Переход к дизайну",
                                        "bg-[#FBF5F7] text-[#73394B] border border-[#E5CFD6]",
                                    )
                            else:
                                card_cls = "w-full bg-[#F8FAFC] border border-[#CBD5E1] border-l-[3px] border-l-[#0284C7] p-2 gap-1 shadow-none"
                                text_cls = "text-xs text-[#1E293B]"
                                badge_cls = "bg-[#EFF6FF] text-[#1E3A8A] border border-[#BFDBFE]"

                            with ui.card().classes(card_cls):
                                with ui.row().classes(
                                    "w-full items-center justify-between gap-1 flex-wrap"
                                ):
                                    with ui.row().classes(
                                        f"items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded {badge_cls}"
                                    ):
                                        ui.html(svg_icon(actor["icon"], 11), sanitize=False)
                                        ui.label(actor["name"]).classes("leading-none")
                                    ui.label(f"{ev.duration_ms:.0f} мс").classes(
                                        "text-[10px] font-mono text-[#475569] shrink-0"
                                    )
                                if verdict_chip is not None:
                                    v_txt, v_cls = verdict_chip
                                    ui.label(v_txt).classes(
                                        f"text-[10px] font-medium px-1.5 py-0.5 rounded self-start {v_cls}"
                                    )
                                ui.label(
                                    _localize_trace_summary(
                                        ev.output_summary or ev.action_taken
                                    )
                                ).classes(text_cls)

                                if is_expert:
                                    stage_key = _AGENT_TO_STAGE.get(
                                        ev.agent_name, ev.agent_name.lower()
                                    )
                                    telemetry = (
                                        node_telemetry.get(stage_key)
                                        if isinstance(node_telemetry, dict)
                                        else None
                                    )
                                    param_chips, stat_chips = format_node_telemetry_chips(
                                        telemetry, param_sources
                                    )
                                    if stat_chips:
                                        with ui.row().classes("w-full gap-1 flex-wrap"):
                                            for sk, sv in stat_chips[:6]:
                                                ui.label(f"{sk}={sv}").classes(
                                                    "text-[10px] mono-code px-1 py-0"
                                                )
                                    if param_chips:
                                        with ui.row().classes("w-full gap-1 flex-wrap"):
                                            for chip in param_chips[:10]:
                                                src_meta = PARAM_SOURCE_BADGE_META.get(
                                                    chip["source"],
                                                    (chip["source"], "slate"),
                                                )
                                                src_title = src_meta[0] or "По умолчанию"
                                                ui.label(chip["label"]).classes(
                                                    "text-[10px] mono-code px-1 py-0"
                                                ).tooltip(f"Источник: {src_title}")

                if is_running:
                    active_key = (
                        "docking"
                        if runtime.is_docking_running
                        else (runtime.current_node or "research")
                    )
                    active_actor = resolve_stage_actor_meta(active_key)
                    is_active_agent = active_actor["kind"] == "agent"
                    live_card_cls = (
                        "w-full bg-[#FBF5F7] border border-[#D9BCC5] border-l-[3px] border-l-[#8B495C] p-2 gap-1 shadow-none"
                        if is_active_agent
                        else "w-full bg-[#EFF6FF] border border-[#BFDBFE] border-l-[3px] border-l-[#0284C7] p-2 gap-1 shadow-none"
                    )
                    live_badge_cls = (
                        "bg-[#8B495C] text-white"
                        if is_active_agent
                        else "bg-white text-[#1E3A8A] border border-[#BFDBFE]"
                    )
                    with ui.card().classes(live_card_cls):
                        with ui.row().classes("w-full items-center justify-between gap-1"):
                            with ui.row().classes(
                                f"items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded {live_badge_cls}"
                            ):
                                ui.html(svg_icon(active_actor["icon"], 11), sanitize=False)
                                ui.label(active_actor["name"]).classes("leading-none")
                            with ui.row().classes("items-center gap-1"):
                                ui.spinner("dots", size="xs", color="primary")
                                ui.label("В процессе").classes(
                                    "text-[10px] font-medium text-[#73394B]"
                                )
                        ui.label(runtime.get_pishovichok_status_text()).classes(
                            "text-[11px] text-[#261F22]"
                        )

    if state_spoiler_box is not None:
        state_spoiler_box.clear()
        if runtime.depth_mode != "expert":
            return
        tc = resolve_tool_config(runtime.state)
        state_flags = {
            "strict_assay_homogeneity": tc.strict_assay_homogeneity,
            "strict_property_thresholds": tc.strict_property_thresholds,
            "assay_filter_strict": runtime.state.get("assay_filter_strict"),
            "active_bao_filter": runtime.state.get("active_bao_filter"),
            "data_refilter_iteration": runtime.state.get("data_refilter_iteration"),
            "design_iteration": runtime.state.get("design_iteration"),
            "critic_verdict": runtime.state.get("critic_verdict"),
            "non_default_param_sources": {
                k: v
                for k, v in runtime.state.get("param_sources", {}).items()
                if v != "default"
            },
        }
        with state_spoiler_box, (
            ui.expansion("Диагностика состояния", value=False)
            .props("dense header-class='text-[11px] text-[#475569] bg-[#F1F5F9] rounded'")
            .classes("w-full bg-white border border-[#E2E8F0] rounded-lg")
        ):
            ui.html(
                f"<pre class='mono-code p-2 text-[10px] overflow-x-auto m-1'>{html.escape(json.dumps(state_flags, ensure_ascii=False, indent=2))}</pre>",
                sanitize=False,
            )
