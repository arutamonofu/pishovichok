"""Главный модуль веб-интерфейса Streamlit (Слой 4).

Точка входа веб-приложения:
    uv run streamlit run src/pishovichok/ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

from pishovichok import __version__
from pishovichok.agents.workflow import run_sar_pipeline
from pishovichok.ui.components import (
    render_candidates_four_zones_table,
    render_control_panel,
    render_explainable_trace,
    render_export_buttons,
    render_sar_and_cliffs_view,
)


def _load_saved_session(session_id: str) -> bool:
    """Загружает сохраненное состояние из `workspace/<session_id>/` в `st.session_state`."""
    sess_dir = Path("workspace") / session_id
    if not sess_dir.is_dir():
        st.error(f"Директория сессии не найдена: {sess_dir}")
        return False

    trace_file = sess_dir / "trace_log.json"
    candidates_file = sess_dir / "evaluated_candidates.parquet"
    report_file = sess_dir / "report.md"
    curated_file = sess_dir / "curated_dataset.parquet"

    trace_log = []
    if trace_file.is_file():
        try:
            trace_log = json.loads(trace_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            st.warning(f"Ошибка чтения trace_log.json: {e}")

    candidates = []
    if candidates_file.is_file():
        try:
            # Чтение Parquet кандидатов через DuckDB / PyArrow
            import duckdb

            from pishovichok.state.schemas import CandidateMolecule

            df_cands = (
                duckdb.connect()
                .execute(f"SELECT * FROM '{candidates_file.as_posix()}'")
                .df()
            )
            for _, row in df_cands.iterrows():
                candidates.append(
                    CandidateMolecule.from_flat_parquet_record(row.to_dict())
                )
        except (RuntimeError, ValueError, OSError) as e:
            st.warning(f"Ошибка чтения кандидатов: {e}")

    report_md = (
        report_file.read_text(encoding="utf-8") if report_file.is_file() else None
    )

    curated_df = None
    if curated_file.is_file():
        try:
            import duckdb

            curated_df = (
                duckdb.connect()
                .execute(f"SELECT * FROM '{curated_file.as_posix()}'")
                .df()
            )
        except (RuntimeError, ValueError, OSError):
            curated_df = None

    st.session_state.current_state = {
        "session_id": session_id,
        "workspace_dir": str(sess_dir),
        "trace_log": trace_log,
        "top_candidates": candidates,
        "extracted_sar_rules": [],
        "detected_activity_cliffs": [],
        "curated_df": curated_df,
        "final_report_markdown": report_md,
    }
    st.session_state.session_dir = str(sess_dir)
    return True


def init_app_state() -> None:
    """Инициализирует сессионное состояние Streamlit."""
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    if "session_dir" not in st.session_state:
        st.session_state.session_dir = None


def main() -> None:
    """Основная функция рендеринга страницы Streamlit."""
    st.set_page_config(
        page_title="Pishovichok | SAR & Drug Design MAS",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_app_state()

    # Заголовок системы
    st.title("🧪 Pishovichok: Мультиагентный SAR-анализ и дизайн малых молекул")
    st.caption(
        f"Версия {__version__} | Слой 4: Интерфейс и трассируемость (Streamlit + rdMolDraw2D + LangGraph) | "
        "Академическая CADD-платформа с гарантией отсутствия галлюцинаций"
    )

    # Боковая панель с информацией об архитектуре
    with st.sidebar:
        st.header("ℹ️ Архитектура системы")
        st.markdown(
            """
            - **Слой 1:** Детерминированное ядро (RDKit, DuckDB, CReM, CatBoost)
            - **Слой 2:** Blackboard + Fat Pointers (Parquet)
            - **Слой 3:** Мультиагентный граф LangGraph:
              - *ResearchAgent* 🔍
              - *SARAgent* 📊
              - *DesignAgent* 🧬
              - *EvaluationAgent* 🔬
              - *CriticAgent* ⚖️ (Петли №1 и №2)
            - **Слой 4:** Explainable Trace & 2D SVG
            - **Слой 5:** Предкэшированный контур `/data`
            """
        )
        st.divider()
        st.markdown("**Демонстрационные сценарии (SPEC.md):**")
        st.markdown("1. `BTK (CHEMBL5251)` — Target SAR")
        st.markdown("2. `EGFR (Gefitinib)` — Lead Opt")
        st.markdown("3. `Wet-Lab CSV` — Серия из 25 соед.")

    # Рендеринг панели управления
    control_actions = render_control_panel()

    # Обработка действий пользователя
    if control_actions["load_saved"]:
        with st.spinner(
            f"Загрузка артефактов сессии '{control_actions['session_id']}'..."
        ):
            if _load_saved_session(control_actions["session_id"]):
                st.success(
                    f"Артефакты сессии '{control_actions['session_id']}' успешно загружены!"
                )

    if control_actions["start_screening"]:
        with st.spinner(
            "Запуск мультиагентного оркестратора LangGraph (Research -> SAR -> Critic -> Design -> Evaluation)..."
        ):
            final_state, _trace_path, _csv_path, _sdf_path = run_sar_pipeline(
                user_query=control_actions["user_query"],
                session_id=control_actions["session_id"],
                uploaded_csv_path=control_actions["uploaded_csv_path"],
                export_artifacts=True,
            )
            # Извлечение DataFrame для графика
            curated_df = None
            if final_state.get("dataset_uri"):
                try:
                    import duckdb

                    curated_df = (
                        duckdb.connect()
                        .execute(f"SELECT * FROM '{final_state['dataset_uri'].uri}'")
                        .df()
                    )
                except (RuntimeError, ValueError, OSError):
                    curated_df = None

            st.session_state.current_state = {
                "session_id": final_state.get("session_id"),
                "workspace_dir": final_state.get("workspace_dir"),
                "trace_log": final_state.get("trace_log", []),
                "top_candidates": final_state.get("top_candidates", []),
                "extracted_sar_rules": final_state.get("extracted_sar_rules", []),
                "detected_activity_cliffs": final_state.get(
                    "detected_activity_cliffs", []
                ),
                "lead_reference_metrics": final_state.get("lead_reference_metrics"),
                "curated_df": curated_df,
                "final_report_markdown": final_state.get("final_report_markdown"),
            }
            st.session_state.session_dir = final_state.get("workspace_dir")
            st.success("Вычислительная сессия оркестратора успешно завершена!")

    st.divider()

    # Основные рабочие вкладки системы
    if st.session_state.current_state is not None:
        state = st.session_state.current_state
        tab_trace, tab_sar, tab_candidates, tab_exports = st.tabs(
            [
                "🧭 Журнал трассировки (Explainable Trace)",
                "🔬 SAR-матрица и Activity Cliffs",
                "🧬 Сводная таблица кандидатов (4 зоны)",
                "📥 Экспорт результатов и отчет",
            ]
        )

        with tab_trace:
            render_explainable_trace(state.get("trace_log", []))

        with tab_sar:
            render_sar_and_cliffs_view(
                sar_rules=state.get("extracted_sar_rules", []),
                activity_cliffs=state.get("detected_activity_cliffs", []),
                curated_df=state.get("curated_df"),
            )

        with tab_candidates:
            render_candidates_four_zones_table(
                candidates=state.get("top_candidates", []),
                lead_reference=state.get("lead_reference_metrics"),
            )

        with tab_exports:
            render_export_buttons(
                session_dir=st.session_state.session_dir,
                candidates=state.get("top_candidates", []),
                trace_log=state.get("trace_log", []),
                final_report_md=state.get("final_report_markdown"),
            )
    else:
        st.info(
            "👆 Выберите один из демонстрационных сценариев выше и нажмите **Запустить скрининг** или **Загрузить сохраненные результаты**."
        )


def run_app() -> None:
    """Точка входа CLI для запуска streamlit run."""
    import subprocess

    app_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
