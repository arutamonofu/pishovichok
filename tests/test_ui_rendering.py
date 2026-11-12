"""Автоматические тесты Слой 4: Веб-интерфейс, 2D-рендеринг молекул и экспорт артефактов.

Критерий готовности:
1. Отсутствие исключений в функциях генерации 2D-графики (SVG, пары, подсветка атомов, фолбэки).
2. Программное тестирование рендеринга интерфейса через Streamlit AppTest.
3. Корректность формирования файлов экспорта (CSV, SDF с 2D координатами, JSON) на всех 3 сценариях.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pishovichok.agents.workflow import (
    export_candidates_to_csv,
    export_candidates_to_sdf,
)
from pishovichok.state.schemas import CandidateMolecule, TraceEvent
from pishovichok.state.storage import export_trace_log
from pishovichok.ui.mol_rendering import (
    COLOR_ACTIVITY_GAIN,
    molecule_svg_to_data_uri,
    render_mmp_pair_svg,
    render_molecule_svg,
)


class TestMol2DRendering:
    """Тестирование C++ модуля 2D-рендеринга молекул на базе rdMolDraw2D."""

    def test_render_valid_molecule_svg(self) -> None:
        """Проверка рендеринга стандартных молекул в SVG."""
        svg = render_molecule_svg("c1ccccc1O", width=300, height=200, legend="Phenol")
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "300" in svg and "200" in svg

    def test_render_with_highlighted_atoms(self) -> None:
        """Проверка подсветки атомов с настраиваемой палитрой."""
        # Подсветка кислорода в феноле
        svg = render_molecule_svg(
            "c1ccccc1O",
            highlight_atoms=[6],
            highlight_colors={6: COLOR_ACTIVITY_GAIN},
            width=320,
            height=220,
        )
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_gefitinib_complex_structure(self) -> None:
        """Проверка рендеринга сложной молекулы-лидера (гефитиниб)."""
        gefitinib = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
        svg = render_molecule_svg(gefitinib, width=400, height=300, legend="Gefitinib")
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_fallback_on_invalid_smiles(self) -> None:
        """Проверка детерминированного фолбэка на невалидный или пустой SMILES."""
        svg_empty = render_molecule_svg("")
        assert "<svg" in svg_empty
        assert "Пустой SMILES" in svg_empty

        svg_invalid = render_molecule_svg("NOT_A_VALID_SMILES_12345")
        assert "<svg" in svg_invalid
        assert "Невалидная структура" in svg_invalid

    def test_render_mmp_pair_svg(self) -> None:
        """Проверка рендеринга пары молекулярных трансформаций с дельтой активности."""
        pair_html = render_mmp_pair_svg(
            smiles_lhs="c1ccccc1O",
            smiles_rhs="c1ccccc1Cl",
            title_lhs="Фенол",
            title_rhs="Хлорбензол",
            delta_pic50=1.65,
            sali_index=22.4,
            width=650,
            height=250,
        )
        assert "<svg" in pair_html
        assert "ΔpIC50 = +1.65" in pair_html
        assert "SALI = 22.4" in pair_html
        assert "Фенол" in pair_html
        assert "Хлорбензол" in pair_html

    def test_molecule_svg_to_data_uri(self) -> None:
        """Проверка кодирования SVG в Data URI."""
        svg = "<svg><circle r='10'/></svg>"
        uri = molecule_svg_to_data_uri(svg)
        assert uri.startswith("data:image/svg+xml;base64,")


class TestScenarioArtifactExports:
    """Тестирование корректности формирования файлов экспорта на 3 демонстрационных сценариях."""

    @pytest.mark.parametrize(
        "scenario_key",
        ["scenario1_btk", "scenario2_gefitinib", "scenario3_wetlab"],
    )
    def test_scenario_export_formats_and_content(
        self, scenario_key: str, tmp_path: Path
    ) -> None:
        """Проверяет экспорт CSV, SDF (2D) и JSON трассировки для каждого сценария."""
        ws_dir = Path("workspace") / scenario_key
        assert ws_dir.is_dir(), f"Workspace directory missing: {ws_dir}"

        # 1. Проверка чтения кандидатов и 4 категорий
        cands_parquet = ws_dir / "evaluated_candidates.parquet"
        assert cands_parquet.is_file(), (
            f"Parquet candidates file missing: {cands_parquet}"
        )

        import duckdb

        rows = (
            duckdb.connect().execute(f"SELECT * FROM '{cands_parquet.as_posix()}'").df()
        )
        assert len(rows) >= 5, (
            f"Expected at least 5 candidates in {scenario_key}, got {len(rows)}"
        )

        candidates = [
            CandidateMolecule.from_flat_parquet_record(r.to_dict())
            for _, r in rows.iterrows()
        ]

        for cand in candidates:
            # Категория 1: Экспериментальный факт
            assert cand.parent_chembl_id, "Missing parent_chembl_id"
            assert cand.parent_exp_pic50 > 0.0, "Invalid parent_exp_pic50"
            assert cand.parent_bao_format, "Missing parent_bao_format"

            # Категория 2: Расчет RDKit + BRICS
            assert cand.calc_rdkit.mw > 0.0
            assert cand.calc_rdkit.sa_score >= 1.0
            assert isinstance(cand.calc_rdkit.brics_fragments, list)
            if scenario_key == "scenario3_wetlab":
                assert len(cand.calc_rdkit.brics_fragments) >= 2, (
                    "Scenario 3 must have BRICS precursors"
                )

            # Категория 3: ML-прогноз
            assert cand.pred_admet.model_source in (
                "admet_ai_dmpnn",
                "catboost_local_qsar",
                "rdkit_analytical_qsar",
            )
            assert 0.0 <= cand.pred_admet.herg_liability_prob <= 1.0

            # Категория 4: Гипотеза SAR
            assert cand.hypothesis_rationale.rule_id
            assert cand.hypothesis_rationale.applied_smirks

        # 2. Тестирование генерации CSV
        csv_out = tmp_path / f"{scenario_key}_test.csv"
        export_candidates_to_csv(candidates, csv_out)
        assert csv_out.is_file()
        csv_text = csv_out.read_text(encoding="utf-8")
        assert "candidate_id" in csv_text
        assert "parent_chembl_id" in csv_text
        assert "brics_fragments" in csv_text
        assert "model_source" in csv_text
        assert "applied_smirks" in csv_text

        # 3. Тестирование генерации SDF с 2D координатами
        sdf_out = tmp_path / f"{scenario_key}_test.sdf"
        export_candidates_to_sdf(candidates, sdf_out)
        assert sdf_out.is_file()
        sdf_text = sdf_out.read_text(encoding="utf-8", errors="ignore")
        assert "$$$$" in sdf_text
        assert "Cat1_Parent_ChEMBL_ID" in sdf_text
        assert "Cat2_SAscore" in sdf_text
        assert "Cat3_Model_Source" in sdf_text
        assert "Cat4_Applied_SMIRKS" in sdf_text

        # 4. Тестирование генерации trace_log.json
        trace_file = ws_dir / "trace_log.json"
        assert trace_file.is_file()
        trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
        assert len(trace_data) >= 3

        events = [TraceEvent.model_validate(e) for e in trace_data]
        json_out = export_trace_log(events, tmp_path / f"{scenario_key}_trace")
        assert json_out.is_file()


class TestStreamlitAppRendering:
    """Интеграционное тестирование рендеринга веб-интерфейса Streamlit через AppTest."""

    @classmethod
    def setup_class(cls) -> None:
        cls.app_file = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "pishovichok"
            / "ui"
            / "app.py"
        ).as_posix()

    def test_streamlit_app_loads_without_exceptions(self) -> None:
        """Проверяет первоначальную загрузку Streamlit-приложения без ошибок."""
        at = AppTest.from_file(self.app_file)
        at.run(timeout=15)
        assert not at.exception, f"App loaded with exception: {at.exception}"

    def test_streamlit_app_loads_scenario1_data(self) -> None:
        """Проверяет загрузку и рендеринг данных Сценария 1 (BTK) в UI."""
        at = AppTest.from_file(self.app_file)
        at.run(timeout=15)

        at.text_input(key="session_id_input_widget").input("scenario1_btk").run()
        load_btn = next((b for b in at.button if "Загрузить" in b.label), None)
        assert load_btn is not None, (
            "Button 'Загрузить сохраненные результаты' not found"
        )
        load_btn.click().run(timeout=15)

        assert not at.exception, (
            f"App exception after loading Scenario 1: {at.exception}"
        )
        curr_state = at.session_state.get("current_state")
        assert curr_state is not None, (
            "current_state must not be None after loading Scenario 1"
        )
        assert len(curr_state.get("top_candidates", [])) >= 5

    def test_streamlit_app_loads_scenario2_gefitinib(self) -> None:
        """Проверяет загрузку и рендеринг данных Сценария 2 (Gefitinib) с обеими петлями."""
        at = AppTest.from_file(self.app_file)
        at.run(timeout=15)

        at.text_input(key="session_id_input_widget").input("scenario2_gefitinib").run()
        load_btn = next((b for b in at.button if "Загрузить" in b.label), None)
        assert load_btn is not None
        load_btn.click().run(timeout=15)

        assert not at.exception, (
            f"App exception after loading Scenario 2: {at.exception}"
        )
        curr_state = at.session_state.get("current_state")
        assert curr_state is not None
        trace_log = curr_state.get("trace_log", [])
        loop1_found = any(
            e.get("verdict_or_routing") == "refilter_data" for e in trace_log
        )
        loop2_found = any(
            e.get("verdict_or_routing") == "redesign_molecules" for e in trace_log
        )
        assert loop1_found, "Scenario 2 trace must contain Loop 1 (refilter_data)"
        assert loop2_found, "Scenario 2 trace must contain Loop 2 (redesign_molecules)"

    def test_streamlit_app_loads_scenario3_wetlab(self) -> None:
        """Проверяет загрузку и рендеринг данных Сценария 3 (Wet-Lab CSV)."""
        at = AppTest.from_file(self.app_file)
        at.run(timeout=15)

        at.text_input(key="session_id_input_widget").input("scenario3_wetlab").run()
        load_btn = next((b for b in at.button if "Загрузить" in b.label), None)
        assert load_btn is not None
        load_btn.click().run(timeout=15)

        assert not at.exception, (
            f"App exception after loading Scenario 3: {at.exception}"
        )
        curr_state = at.session_state.get("current_state")
        assert curr_state is not None
        assert len(curr_state.get("top_candidates", [])) >= 5
