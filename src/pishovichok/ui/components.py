"""UI-компоненты веб-интерфейса Streamlit (Слой 4).

Включает:
1. Панель управления и ввода с 3 демонстрационными сценариями (Раздел 7 SPEC.md) и загрузкой CSV.
2. Журнал трассировки агентов (Explainable Trace) с наглядным выделением петель самокоррекции.
3. Интерактивная SAR-матрица, граф Activity Cliffs и диаграмма рассеяния Plotly.
4. Сводная таблица кандидатов со строгим цветовым разделением 4 категорий природы данных.
5. Кнопки экспорта артефактов сессии (CSV, SDF, JSON, MD).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from pishovichok.state.schemas import (
    ActivityCliff,
    CandidateMolecule,
    SARRule,
    TraceEvent,
)
from pishovichok.ui.mol_rendering import render_mmp_pair_svg, render_molecule_svg

# Описания 3 демонстрационных сценариев из Раздела 7 SPEC.md
DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    "scenario1": {
        "title": "Сценарий 1. Target-Based SAR (BTK / CHEMBL5251)",
        "subtitle": "Поисковый дизайн ингибиторов тирозинкиназы Брутона с фильтрацией неоднородных тестов (Петля №1)",
        "query": (
            "Проведи анализ ингибиторов тирозинкиназы Брутона (BTK, CHEMBL5251), "
            "выдели ключевые правила 'структура–активность' на однородных биохимических тестах "
            "и сгенерируй 5 новых кандидатов с расчетным LogP от 2.0 до 4.0, SAscore < 3.2 "
            "и отсутствием структурных алертов PAINS."
        ),
        "csv_path": None,
        "session_id": "scenario1_btk",
    },
    "scenario2": {
        "title": "Сценарий 2. Lead Optimization (EGFR / Gefitinib, CHEMBL203)",
        "subtitle": "Направленная оптимизация гефитиниба с фиксацией хиназолинового ядра (Петли №1 и №2)",
        "query": (
            "Оптимизируй структуру ингибитора EGFR со строкой SMILES "
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1 (гефитиниб): "
            "сохрани хиназолиновый скаффолд, снизь липофильность (LogP < 3.5), "
            "минимизируй риск блокады каналов hERG и предложи синтетически доступные аналоги (SAscore < 3.0)."
        ),
        "csv_path": None,
        "session_id": "scenario2_gefitinib",
    },
    "scenario3": {
        "title": "Сценарий 3. Custom Wet-Lab CSV Analysis (25 соединений)",
        "subtitle": "Курация лабораторного файла, детекция Activity Cliffs и ретросинтез BRICS",
        "query": (
            "Проанализируй результаты испытаний нашей лабораторной серии из 25 соединений, "
            "найди критические перепады активности (Activity Cliffs) и предложи 5 структур "
            "для следующей очереди синтеза с разбивкой на коммерчески доступные прекурсоры."
        ),
        "csv_path": "data/samples/custom_wetlab_series.csv",
        "session_id": "scenario3_wetlab",
    },
}


def render_control_panel() -> dict[str, Any]:
    """Рендерит панель управления: выбор пресетов, ввод запроса, загрузка CSV."""
    st.markdown("### 🎮 Панель управления и запуска скрининга")

    # Быстрый запуск демонстрационных сценариев
    st.markdown("**Быстрый запуск демонстрационных сценариев (Раздел 7 SPEC.md):**")
    cols = st.columns(3)

    selected_scenario_key = None
    if cols[0].button(
        "🎯 Сценарий 1: BTK (Target SAR)",
        use_container_width=True,
        help="Мишень CHEMBL5251, Петля №1 по BAO",
    ):
        selected_scenario_key = "scenario1"
    if cols[1].button(
        "💊 Сценарий 2: Гефитиниб (Lead Opt)",
        use_container_width=True,
        help="EGFR CHEMBL203, Петли №1 и №2",
    ):
        selected_scenario_key = "scenario2"
    if cols[2].button(
        "🧪 Сценарий 3: Wet-Lab CSV (25 соед.)",
        use_container_width=True,
        help="Пользовательский CSV + Activity Cliffs",
    ):
        selected_scenario_key = "scenario3"

    # Инициализация дефолтного значения запроса
    if "user_query_input" not in st.session_state:
        st.session_state.user_query_input = DEMO_SCENARIOS["scenario1"]["query"]
    if "session_id_input" not in st.session_state:
        st.session_state.session_id_input = DEMO_SCENARIOS["scenario1"]["session_id"]
    if "uploaded_csv_path" not in st.session_state:
        st.session_state.uploaded_csv_path = None

    if selected_scenario_key:
        scen = DEMO_SCENARIOS[selected_scenario_key]
        st.session_state.user_query_input = scen["query"]
        st.session_state.session_id_input = scen["session_id"]
        st.session_state.uploaded_csv_path = scen["csv_path"]
        st.info(f"Выбран {scen['title']}: {scen['subtitle']}")

    # Поле ввода текста запроса
    user_query = st.text_area(
        "Запрос на естественном языке (цель, мишень ChEMBL, ограничения cLogP, SAscore, hERG):",
        value=st.session_state.user_query_input,
        height=105,
        key="query_text_area",
    )

    col_csv, col_session = st.columns([2, 1])

    # Загрузка пользовательского файла CSV
    with col_csv:
        uploaded_file = st.file_uploader(
            "Загрузить пользовательскую лабораторную таблицу CSV (опционально):",
            type=["csv"],
            help="Колонки: mol_id, SMILES, IC50_nM, units (или стандартный экспорт ChEMBL)",
        )
        if uploaded_file is not None:
            # Сохранение во временный сессионный файл
            tmp_csv_path = Path("workspace") / f"uploaded_{uploaded_file.name}"
            tmp_csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_csv_path.write_bytes(uploaded_file.getvalue())
            st.session_state.uploaded_csv_path = str(tmp_csv_path)
            st.success(
                f"Загружен файл: {uploaded_file.name} ({len(uploaded_file.getvalue())} байт)"
            )

    with col_session:
        session_id = st.text_input(
            "Идентификатор сессии (workspace):",
            value=st.session_state.session_id_input,
            key="session_id_input_widget",
        )

    # Кнопки запуска конвейера
    btn_col1, btn_col2 = st.columns([1, 4])
    start_clicked = btn_col1.button(
        "🚀 Запустить скрининг", type="primary", use_container_width=True
    )
    load_saved_clicked = btn_col2.button(
        "📂 Загрузить сохраненные результаты сценария", use_container_width=False
    )

    return {
        "start_screening": start_clicked,
        "load_saved": load_saved_clicked,
        "user_query": user_query,
        "session_id": session_id,
        "uploaded_csv_path": st.session_state.uploaded_csv_path,
    }


def render_explainable_trace(
    trace_log: list[TraceEvent] | list[dict[str, Any]],
) -> None:
    """Визуализирует журнал Explainable Trace с выделением петель самокоррекции."""
    st.markdown("### 🧭 Журнал трассировки агентов (Explainable Trace)")
    if not trace_log:
        st.info(
            "Трассировка пока пуста. Запустите скрининг или выберите демонстрационный сценарий."
        )
        return

    # Преобразование в типизированные объекты при необходимости
    events: list[TraceEvent] = []
    for item in trace_log:
        if isinstance(item, TraceEvent):
            events.append(item)
        elif isinstance(item, dict):
            events.append(TraceEvent.model_validate(item))

    # Подсчет срабатываний петель
    loop1_events = [e for e in events if e.verdict_or_routing == "refilter_data"]
    loop2_events = [e for e in events if e.verdict_or_routing == "redesign_molecules"]
    approved_events = [e for e in events if e.verdict_or_routing == "approve"]

    # Метрические плашки статуса
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Всего шагов в графе", len(events))
    m2.metric(
        "Петля №1 (Refilter BAO)",
        f"{len(loop1_events)} раз(а)",
        delta="Активирована" if loop1_events else "Не требовалась",
    )
    m3.metric(
        "Петля №2 (Redesign)",
        f"{len(loop2_events)} раз(а)",
        delta="Активирована" if loop2_events else "Не требовалась",
    )
    m4.metric(
        "Статус Critic Agent", "Утверждено ✅" if approved_events else "В процессе"
    )

    # Выделенные уведомления о сработавших петлях
    if loop1_events:
        for ev in loop1_events:
            st.warning(
                f"🔄 **Сработала Петля самокоррекции №1 (Critic -> Research):**\n\n"
                f"**Причина:** {ev.output_summary}\n\n"
                f"*Действие:* Перефильтрация выборки со строгим соблюдением однородности протоколов испытаний (BAO)."
            )

    if loop2_events:
        for ev in loop2_events:
            st.warning(
                f"🔁 **Сработала Петля самокоррекции №2 (Critic -> Design):**\n\n"
                f"**Причина:** {ev.output_summary}\n\n"
                f"*Действие:* Изменение структурных ограничений и повторная генерация более полярных/доступных кандидатов."
            )

    # Детальный таймлайн каждого шага
    st.markdown("#### Пошаговая детализация маршрута:")
    for ev in events:
        icon = "🤖"
        if ev.agent_name == "CriticAgent":
            icon = "⚖️"
        elif ev.agent_name == "ResearchAgent":
            icon = "🔍"
        elif ev.agent_name == "SARAgent":
            icon = "📊"
        elif ev.agent_name == "DesignAgent":
            icon = "🧬"
        elif ev.agent_name == "EvaluationAgent":
            icon = "🔬"

        routing_badge = (
            f"`Маршрут: {ev.verdict_or_routing}`" if ev.verdict_or_routing else ""
        )
        header = f"{icon} Шаг {ev.step_index + 1}: **{ev.agent_name}** | {ev.phase} ({ev.duration_ms:.1f} мс) {routing_badge}"

        with st.expander(
            header,
            expanded=bool(
                ev.verdict_or_routing in ("refilter_data", "redesign_molecules")
            ),
        ):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Входные параметры:**\n`{ev.input_summary}`")
                st.markdown(f"**Выполненное действие:**\n{ev.action_taken}")
            with c2:
                st.markdown(f"**Результат шага:**\n{ev.output_summary}")
                if ev.fallback_triggered:
                    st.caption(
                        f"⚡ Задействован детерминированный фолбэк: `{ev.fallback_triggered}`"
                    )


def render_sar_and_cliffs_view(
    sar_rules: list[SARRule] | list[dict[str, Any]],
    activity_cliffs: list[ActivityCliff] | list[dict[str, Any]],
    curated_df: pd.DataFrame | None = None,
) -> None:
    """Отображает интерактивную SAR-матрицу, пары Activity Cliffs и график рассеяния Plotly."""
    st.markdown("### 🔬 Анализ SAR и детекция Activity Cliffs")

    tab_rules, tab_cliffs, tab_scatter = st.tabs(
        [
            "📋 Правила трансформации (SARRule)",
            "⚡ Карточки Activity Cliffs (2D)",
            "📈 Ландшафт свойств и активности (Plotly)",
        ]
    )

    # 1. Вкладка правил трансформации
    with tab_rules:
        if not sar_rules:
            st.info("Нет извлеченных правил SAR.")
        else:
            rules_data = []
            for r in sar_rules:
                rule_dict = r if isinstance(r, dict) else r.model_dump()
                rules_data.append(
                    {
                        "ID": rule_dict.get("rule_id"),
                        "SMIRKS": rule_dict.get("smirks"),
                        "ΔpIC50 (средний)": f"{rule_dict.get('mean_delta_pic50', 0.0):+.2f}",
                        "Число пар (Support)": rule_dict.get("support_pairs_count"),
                        "ΔcLogP": f"{rule_dict.get('mean_delta_clogp', 0.0):+.2f}",
                        "ΔTPSA": f"{rule_dict.get('mean_delta_tpsa', 0.0):+.1f}",
                        "Гомогенность BAO": "Да ✅"
                        if rule_dict.get("assay_homogeneous")
                        else "Нет ❌",
                    }
                )
            st.dataframe(
                pd.DataFrame(rules_data), hide_index=True, use_container_width=True
            )

    # 2. Вкладка пар Activity Cliffs с 2D-рендерингом
    with tab_cliffs:
        if not activity_cliffs:
            st.info(
                "В выборке не выявлено резких скачков активности (Activity Cliffs с Tanimoto >= 0.85 и |ΔpIC50| >= 1.5)."
            )
        else:
            st.markdown(
                f"Найдено критических точек перепада активности: **{len(activity_cliffs)}**"
            )
            for cliff in activity_cliffs[:6]:  # Отображаем топ-6
                c_dict = cliff if isinstance(cliff, dict) else cliff.model_dump()
                st.markdown(
                    f"**Activity Cliff {c_dict.get('cliff_id')}:** замена `{c_dict.get('smirks_difference')}`"
                )
                st.markdown(
                    f"Активная: `{c_dict.get('mol_id_active')}` (pIC50 = {c_dict.get('pic50_active', 0.0):.2f}) ➔ "
                    f"Неактивная: `{c_dict.get('mol_id_inactive')}` (pIC50 = {c_dict.get('pic50_inactive', 0.0):.2f}) | "
                    f"Tanimoto: `{c_dict.get('tanimoto_similarity', 0.0):.2f}`"
                )
                pair_html = render_mmp_pair_svg(
                    smiles_lhs=c_dict.get("smiles_active", ""),
                    smiles_rhs=c_dict.get("smiles_inactive", ""),
                    title_lhs=f"Активная ({c_dict.get('mol_id_active')})",
                    title_rhs=f"Неактивная ({c_dict.get('mol_id_inactive')})",
                    delta_pic50=c_dict.get("abs_delta_pic50", 1.5),
                    sali_index=c_dict.get("sali_index"),
                )
                st.components.v1.html(pair_html, height=270, scrolling=False)

    # 3. Вкладка интерактивного графика Plotly
    with tab_scatter:
        if curated_df is not None and not curated_df.empty:
            # Расчет приблизительного cLogP и MW при необходимости
            df_plot = curated_df.copy()
            if "pchembl_value" in df_plot.columns:
                from rdkit import Chem
                from rdkit.Chem import Descriptors

                if (
                    "clogp" not in df_plot.columns
                    and "canonical_smiles" in df_plot.columns
                ):
                    clogps = []
                    mws = []
                    for sm in df_plot["canonical_smiles"]:
                        m = Chem.MolFromSmiles(str(sm)) if sm else None
                        clogps.append(Descriptors.MolLogP(m) if m else 0.0)
                        mws.append(Descriptors.ExactMolWt(m) if m else 0.0)
                    df_plot["cLogP"] = clogps
                    df_plot["MW"] = mws

                fig = px.scatter(
                    df_plot,
                    x="cLogP",
                    y="pchembl_value",
                    size="MW" if "MW" in df_plot.columns else None,
                    color="bao_format" if "bao_format" in df_plot.columns else None,
                    hover_data=["mol_id", "canonical_smiles"],
                    title="Ландшафт биоактивности: pIC50 vs cLogP (размер маркера — молекулярная масса)",
                    labels={
                        "pchembl_value": "pIC50 (-log10 M)",
                        "cLogP": "cLogP (липофильность)",
                    },
                    template="plotly_white",
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет данных pchembl_value для построения графика.")
        else:
            st.info("Таблица датасета недоступна для построения распределения.")


def render_candidates_four_zones_table(
    candidates: list[CandidateMolecule],
    lead_reference: CandidateMolecule | None = None,
) -> None:
    """Отображает сводную таблицу кандидатов с явным разделением на 4 категории природы данных."""
    st.markdown("### 🧬 Сводная таблица кандидатов (4 категории природы данных)")
    if not candidates:
        st.info("Кандидаты еще не сгенерированы.")
        return

    # Карточка исходного лидера (для Сценария 2)
    if lead_reference is not None:
        st.markdown("#### 📌 Исходная молекула-лидер (Lead Reference):")
        lead_c1, lead_c2 = st.columns([1, 2])
        with lead_c1:
            lead_svg = render_molecule_svg(
                lead_reference.canonical_smiles,
                width=280,
                height=200,
                legend="Исходный лидер",
            )
            st.components.v1.html(lead_svg, height=210)
        with lead_c2:
            st.markdown(
                f"- **Идентификатор:** `{lead_reference.candidate_id}`\n"
                f"- **Экспериментальный pIC50:** `{lead_reference.parent_exp_pic50:.2f}` (IC50 = {lead_reference.parent_exp_ic50_nm:.1f} нМ)\n"
                f"- **cLogP (базовый):** `{lead_reference.calc_rdkit.clogp:.2f}` | **SAscore:** `{lead_reference.calc_rdkit.sa_score:.2f}`\n"
                f"- **Риск блокады hERG:** `{lead_reference.pred_admet.herg_liability_prob:.2f}` ({lead_reference.pred_admet.herg_risk_level})"
            )

    # Пояснительная легенда 4 цветовых зон
    st.markdown(
        """
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:15px;">
          <span style="background-color:#f3e8ff; color:#6b21a8; border:1px solid #d8b4fe; padding:4px 10px; border-radius:6px; font-weight:bold; font-size:13px;">
            🟪 Категория 1: Экспериментальный факт (ChEMBL / CSV)
          </span>
          <span style="background-color:#dcfce7; color:#166534; border:1px solid #86efac; padding:4px 10px; border-radius:6px; font-weight:bold; font-size:13px;">
            🟩 Категория 2: Точный расчет RDKit + BRICS
          </span>
          <span style="background-color:#fef9c3; color:#854d0e; border:1px solid #fde047; padding:4px 10px; border-radius:6px; font-weight:bold; font-size:13px;">
            🟨 Категория 3: Модельный прогноз ADMET / QSAR
          </span>
          <span style="background-color:#dbeafe; color:#1e40af; border:1px solid #93c5fd; padding:4px 10px; border-radius:6px; font-weight:bold; font-size:13px;">
            🟦 Категория 4: Научная гипотеза SAR
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Формирование структурированной таблицы для отображения
    rows = []
    for cand in candidates:
        rows.append(
            {
                "Кандидат": cand.candidate_id,
                # Зона 1: Эксперимент
                "Parent ChEMBL": cand.parent_chembl_id,
                "Parent Exp pIC50": f"{cand.parent_exp_pic50:.2f}",
                "BAO Format": cand.parent_bao_format,
                # Зона 2: Расчет RDKit
                "cLogP": f"{cand.calc_rdkit.clogp:.2f}",
                "ΔcLogP": f"{cand.calc_rdkit.delta_clogp_vs_parent:+.2f}",
                "SAscore": f"{cand.calc_rdkit.sa_score:.2f}",
                "PAINS/Brenk": f"{cand.calc_rdkit.pains_alerts_count}/{cand.calc_rdkit.brenk_alerts_count}",
                "Прекурсоры BRICS": len(cand.calc_rdkit.brics_fragments),
                # Зона 3: ML-прогноз
                "Pred pIC50": f"{cand.pred_admet.predicted_pic50:.2f}",
                "hERG Risk": f"{cand.pred_admet.herg_liability_prob:.2f} ({cand.pred_admet.herg_risk_level})",
                "Модель": cand.pred_admet.model_source,
                # Зона 4: Гипотеза SAR
                "Правило SAR": cand.hypothesis_rationale.rule_id,
                "ΔpIC50 (MMP)": f"{cand.hypothesis_rationale.historical_mean_delta_pic50:+.2f}",
                "Support (пар)": cand.hypothesis_rationale.support_pairs_count,
                # Интегральный показатель
                "Индекс Харрингтона D": f"{cand.mpo_harrington_score:.3f}",
                "Все ограничения": "Соответствует ✅"
                if cand.passes_all_constraints
                else "Компромисс ⚠️",
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Детальные карточки топ-кандидатов с 2D-структурами и блоками BRICS
    st.markdown("#### 🔬 Карточки структуры и ретросинтетической декомпозиции BRICS:")
    for i, cand in enumerate(candidates, 1):
        with st.expander(
            f"Кандидат {i}: **{cand.candidate_id}** | SMILES: `{cand.canonical_smiles}` (Харрингтон D = {cand.mpo_harrington_score:.3f})"
        ):
            card_col1, card_col2 = st.columns([1, 2])
            with card_col1:
                # Отрисовка структуры с подсветкой модифицированных атомов
                svg_cand = render_molecule_svg(
                    smiles=cand.canonical_smiles,
                    highlight_atoms=cand.highlighted_atom_indices,
                    width=320,
                    height=230,
                    legend=f"{cand.candidate_id} (D={cand.mpo_harrington_score:.2f})",
                )
                st.components.v1.html(svg_cand, height=240)
            with card_col2:
                # 4-зонная детализация карточки
                st.markdown(
                    f"**🟪 1. Экспериментальный факт:** Родитель `{cand.parent_chembl_id}` | "
                    f"Exp pIC50: `{cand.parent_exp_pic50:.2f}` (IC50 = {cand.parent_exp_ic50_nm:.1f} нМ) | "
                    f"Формат BAO: `{cand.parent_bao_format}`\n\n"
                    f"**🟩 2. Расчет RDKit:** MW: `{cand.calc_rdkit.mw:.1f}` | cLogP: `{cand.calc_rdkit.clogp:.2f}` "
                    f"(Δ: `{cand.calc_rdkit.delta_clogp_vs_parent:+.2f}`) | TPSA: `{cand.calc_rdkit.tpsa:.1f}` | "
                    f"QED: `{cand.calc_rdkit.qed:.2f}` | SAscore: `{cand.calc_rdkit.sa_score:.2f}`\n\n"
                    f"**🟨 3. Модельный прогноз:** Ожидаемый pIC50: `{cand.pred_admet.predicted_pic50:.2f}` | "
                    f"LogS: `{cand.pred_admet.log_s_aq:.2f}` | Вероятность hERG: `{cand.pred_admet.herg_liability_prob:.2f}` ({cand.pred_admet.herg_risk_level}) | "
                    f"Источник: `{cand.pred_admet.model_source}`\n\n"
                    f"**🟦 4. Научная гипотеза SAR:** Правило `{cand.hypothesis_rationale.rule_id}`: "
                    f"`{cand.hypothesis_rationale.applied_smirks}`. {cand.hypothesis_rationale.text_summary}"
                )

                # Прекурсоры BRICS
                if cand.calc_rdkit.brics_fragments:
                    st.markdown("**🧱 Синтетические прекурсоры BRICS:**")
                    chips = " ".join(
                        [f"`{frag}`" for frag in cand.calc_rdkit.brics_fragments]
                    )
                    st.markdown(chips)


def render_export_buttons(
    session_dir: str | Path | None,
    candidates: list[CandidateMolecule],
    trace_log: list[TraceEvent] | list[dict[str, Any]],
    final_report_md: str | None = None,
) -> None:
    """Рендерит кнопки скачивания артефактов сессии."""
    st.markdown("### 📥 Экспорт результатов и воспроизводимость")

    c1, c2, c3, c4 = st.columns(4)

    # 1. Скачивание CSV таблицы кандидатов
    if candidates:
        from pishovichok.agents.workflow import (
            export_candidates_to_csv,
            export_candidates_to_sdf,
        )

        tmp_dir = Path("workspace") / "exports"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        csv_file = tmp_dir / "candidates_export.csv"
        export_candidates_to_csv(candidates, csv_file)
        c1.download_button(
            label="📄 Скачать CSV кандидатов",
            data=csv_file.read_bytes(),
            file_name="candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # 2. Скачивание SDF с 2D координатами и тегами 4 категорий
        sdf_file = tmp_dir / "candidates_export.sdf"
        export_candidates_to_sdf(candidates, sdf_file)
        c2.download_button(
            label="🧪 Скачать SDF (2D структуры)",
            data=sdf_file.read_bytes(),
            file_name="candidates.sdf",
            mime="chemical/x-mdl-sdfile",
            use_container_width=True,
        )

    # 3. Скачивание полного протокола трассировки JSON
    if trace_log:
        trace_json = json.dumps(
            [
                e.model_dump(mode="json") if isinstance(e, TraceEvent) else e
                for e in trace_log
            ],
            indent=2,
            ensure_ascii=False,
        )
        c3.download_button(
            label="📜 Скачать trace_log.json",
            data=trace_json.encode("utf-8"),
            file_name="trace_log.json",
            mime="application/json",
            use_container_width=True,
        )

    # 4. Скачивание финального отчета Markdown
    if final_report_md:
        c4.download_button(
            label="📑 Скачать отчет report.md",
            data=final_report_md.encode("utf-8"),
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    elif session_dir and (Path(session_dir) / "report.md").is_file():
        report_bytes = (Path(session_dir) / "report.md").read_bytes()
        c4.download_button(
            label="📑 Скачать отчет report.md",
            data=report_bytes,
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True,
        )
