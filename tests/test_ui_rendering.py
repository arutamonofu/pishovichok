"""Тесты NiceGUI 4.0 Chemical IDE, 2D/3D рендеринга, изоляции слоёв и контрактов UI."""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pishovichok.config import settings
from pishovichok.core.rendering import (
    compute_chemical_space_projection,
    generate_3d_conformer_molblock,
    molecule_svg_to_data_uri,
    render_aligned_diff_svg,
    render_mmp_pair_svg,
    render_mmp_pair_svgs,
    render_molecule_svg,
)
from pishovichok.state.schemas import (
    CalcRDKitMetrics,
    CandidateMolecule,
    PredADMETMetrics,
    SARHypothesisRationale,
)
from pishovichok.ui.bridge import AgentUIBridge
from pishovichok.ui.routes import router as api_router
from pishovichok.ui.session import HITLDecision, SessionRegistry
from pishovichok.ui.state_vm import StateViewModel

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "pishovichok"


def _make_dummy_candidate(
    candidate_id: str = "cand_test_1",
    smiles: str = "COc1ccc(Nc2ncnc3ccccc23)cc1",
    parent_smiles: str = "c1ccc(Nc2ncnc3ccccc23)cc1",
    scaffold_smiles: str = "c1ccc(Nc2ncnc3ccccc23)cc1",
    pains_count: int = 0,
    brenk_count: int = 0,
    pic50: float = 8.2,
    clogp: float = 3.1,
) -> CandidateMolecule:
    mpo = 0.0 if (pains_count > 0 or brenk_count > 0) else 0.78
    return CandidateMolecule(
        candidate_id=candidate_id,
        canonical_smiles=smiles,
        scaffold_smiles=scaffold_smiles,
        generation_iteration=1,
        parent_chembl_id="CHEMBL1873475",
        parent_smiles=parent_smiles,
        parent_exp_pic50=7.5,
        parent_exp_ic50_nm=31.6,
        parent_bao_format="BAO_0000357",
        parent_assay_type="B",
        parent_source_dataset="chembl_test",
        calc_rdkit=CalcRDKitMetrics(
            mw=251.3,
            clogp=clogp,
            tpsa=47.0,
            hbd=1,
            hba=4,
            rotatable_bonds=3,
            fsp3=0.1,
            qed=0.81,
            sa_score=2.1,
            lipinski_violations=0,
            pains_alerts_count=pains_count,
            brenk_alerts_count=brenk_count,
            delta_clogp_vs_parent=0.1,
            delta_tpsa_vs_parent=9.2,
            brics_fragments=["c1ccccc1"],
        ),
        pred_admet=PredADMETMetrics(
            model_source="catboost_local_qsar",
            log_s_aq=-3.2,
            solubility_class="moderate",
            bbb_permeability_prob=0.55,
            herg_liability_prob=0.18,
            herg_risk_level="low",
            cyp3a4_inhibition_prob=0.22,
            predicted_pic50=pic50,
            prediction_confidence="high",
        ),
        hypothesis_rationale=SARHypothesisRationale(
            generation_origin="crem_fragment_replacement",
            rule_id="rule_1",
            applied_smirks="[cH:1]>>[c:1]OC",
            replaced_group_smiles="*[H]",
            introduced_group_smiles="*OC",
            historical_mean_delta_pic50=0.7,
            support_pairs_count=5,
            assay_homogeneous=True,
            text_summary="Test methoxy substitution",
        ),
        highlighted_atom_indices=[0, 1],
        mpo_harrington_score=mpo,
        passes_all_constraints=(pains_count == 0 and brenk_count == 0),
        constraint_violations=["PAINS"] if pains_count > 0 else [],
    )


class TestCoreRendering:
    """Проверка чистых RDKit-функций рендеринга и совместимости с `spawn` (`pickle`)."""

    def test_render_molecule_svg_and_alignment(self) -> None:
        svg = render_molecule_svg(
            "COc1ccc(Nc2ncnc3ccccc23)cc1",
            highlight_atoms=[0, 1],
            align_to_scaffold_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
        )
        assert "<svg" in svg
        assert "</svg>" in svg

        bad_svg = render_molecule_svg("INVALID_SMILES_###")
        assert "<svg" in bad_svg

    def test_render_aligned_diff_svg_and_mmp_pairs(self) -> None:
        svg = render_aligned_diff_svg(
            smiles="COc1ccc(Nc2ncnc3ccccc23)cc1",
            parent_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
            scaffold_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
            is_positive_delta=True,
        )
        assert "<svg" in svg

        p_svg, c_svg = render_mmp_pair_svgs(
            "c1ccc(Nc2ncnc3ccccc23)cc1",
            "COc1ccc(Nc2ncnc3ccccc23)cc1",
            delta_pic50=0.7,
        )
        assert "<svg" in p_svg and "<svg" in c_svg

        combined = render_mmp_pair_svg(
            "c1ccc(Nc2ncnc3ccccc23)cc1",
            "COc1ccc(Nc2ncnc3ccccc23)cc1",
            delta_pic50=-0.4,
        )
        assert "<svg" in combined
        assert molecule_svg_to_data_uri(p_svg).startswith("data:image/svg+xml;base64,")

    def test_generate_3d_conformer_and_pca(self) -> None:
        molblock, hl = generate_3d_conformer_molblock("COc1ccc(Nc2ncnc3ccccc23)cc1")
        assert "V2000" in molblock or "V3000" in molblock
        assert "M  END" in molblock
        assert isinstance(hl, list)

        coords = compute_chemical_space_projection(
            [
                "c1ccc(Nc2ncnc3ccccc23)cc1",
                "COc1ccc(Nc2ncnc3ccccc23)cc1",
                "Fc1ccc(Nc2ncnc3ccccc23)cc1",
            ]
        )
        assert len(coords) == 3
        assert all(len(pt) == 2 for pt in coords)

    def test_spawn_pickle_compatibility(self) -> None:
        """Все функции для `run.cpu_bound` должны быть сериализуемы через `pickle`."""
        for fn in (
            render_molecule_svg,
            render_aligned_diff_svg,
            generate_3d_conformer_molblock,
            compute_chemical_space_projection,
        ):
            unpickled = pickle.loads(pickle.dumps(fn))
            assert unpickled is fn


class TestLayerIsolationAndAssets:
    """Проверка изоляции слоёв, удаления Streamlit и локальных JS-ассетов."""

    def test_no_web_imports_in_domain_layers(self) -> None:
        forbidden = ("nicegui", "fastapi", "starlette", "streamlit")
        for subpkg in ("core", "agents", "state"):
            for py_file in (_SRC_ROOT / subpkg).rglob("*.py"):
                text = py_file.read_text(encoding="utf-8")
                for mod in forbidden:
                    assert f"import {mod}" not in text, f"{mod} found in {py_file}"
                    assert f"from {mod}" not in text, f"{mod} found in {py_file}"

    def test_no_streamlit_and_no_external_cdn_in_ui(self) -> None:
        ui_dir = _SRC_ROOT / "ui"
        cdn_markers = ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com", "3dmol.org")
        for file_path in list(ui_dir.rglob("*.py")) + list(ui_dir.rglob("*.js")):
            if "jsme/" in file_path.as_posix():
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            assert "streamlit" not in text.lower(), f"streamlit found in {file_path}"
            for cdn in cdn_markers:
                assert cdn not in text, f"External CDN {cdn} found in {file_path}"

    def test_local_static_js_assets_exist(self) -> None:
        static_dir = _SRC_ROOT / "ui" / "static"
        assert (static_dir / "3Dmol-min.js").is_file()
        assert (static_dir / "viewer3d.js").is_file()
        assert (static_dir / "jsme_sketcher.js").is_file()
        assert (static_dir / "ketcher_sketcher.js").is_file()
        assert (static_dir / "jsme" / "jsme.nocache.js").is_file()


class TestStateViewModel:
    """Проверка векторного пересчёта MPO и лёгких строк `ui.aggrid`."""

    def test_aggrid_rows_contain_no_svg_and_mpo_recalculates(self) -> None:
        vm = StateViewModel("sess_vm_1")
        c_ok = _make_dummy_candidate("c1", pic50=8.6, clogp=2.8)
        c_veto = _make_dummy_candidate("c2", pains_count=1, pic50=9.0, clogp=2.5)
        vm.upsert_candidates([c_ok, c_veto])

        rows = vm.pop_new_grid_rows()
        assert len(rows) == 2
        assert vm.pop_new_grid_rows() == []

        for row in rows:
            assert "<svg" not in str(row)
            assert "/api/sessions/sess_vm_1/mol/" in row["preview_html"]
            assert 'loading="lazy"' in row["preview_html"]

        vm.weights.max_logp = 2.6
        vm.recalculate_mpo()
        assert vm.rows_by_id["c1"].passes_all is False
        assert vm.rows_by_id["c2"].mpo_score == 0.0

        vm.set_pinned("c1", True)
        assert vm.rows_by_id["c1"].pinned is True
        vm.set_excluded("c1", True)
        assert vm.rows_by_id["c1"].excluded is True
        assert vm.rows_by_id["c1"].pinned is False


class TestSessionRegistryAndRoutes:
    """Проверка `SessionRegistry`, отписок клиентов и FastAPI-маршрута SVG."""

    def test_svg_route_and_session_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = SessionRegistry()
        monkeypatch.setattr("pishovichok.ui.routes.session_registry", registry)

        rt = registry.get_or_create("sess_route_1")
        cand = _make_dummy_candidate("cand_svg_1")
        rt.view_model.upsert_candidates([cand])
        registry.subscribe("sess_route_1", "client_1", lambda _ev: None)
        assert "client_1" in rt.subscribers
        registry.unsubscribe("sess_route_1", "client_1")
        assert "client_1" not in rt.subscribers

        from nicegui import run as nicegui_run

        nicegui_run.setup()
        try:
            app = FastAPI()
            app.include_router(api_router)
            client = TestClient(app)

            resp = client.get("/api/sessions/sess_route_1/mol/cand_svg_1/svg?variant=diff")
            assert resp.status_code == 200
            assert "image/svg+xml" in resp.headers["content-type"]
            assert "max-age=3600" in resp.headers.get("cache-control", "")
            assert "<svg" in resp.text

            # Повторный запрос должен попадать в кэш сессии
            resp_cached = client.get("/api/sessions/sess_route_1/mol/cand_svg_1/svg?variant=diff")
            assert resp_cached.status_code == 200
            assert resp_cached.text == resp.text

            resp_404 = client.get("/api/sessions/sess_route_1/mol/missing_id/svg")
            assert resp_404.status_code == 404
        finally:
            nicegui_run.tear_down()


class TestAgentUIBridge:
    """Проверка HITL-шлюза, отмены и перезапуска ветки в `AgentUIBridge`."""

    def test_hitl_gate_and_cancel(self) -> None:
        async def _run_test() -> None:
            reg = SessionRegistry()
            bridge = AgentUIBridge(registry=reg)
            rt = reg.get_or_create("sess_hitl")

            # Имитация паузы HITL
            rt.status = "paused_hitl"
            rt.hitl_gate.clear()

            async def _approve_soon() -> None:
                await asyncio.sleep(0.05)
                await bridge.submit_hitl_decision(
                    "sess_hitl", HITLDecision(action="approve")
                )

            task = asyncio.create_task(_approve_soon())
            await asyncio.wait_for(rt.hitl_gate.wait(), timeout=2.0)
            await task
            assert rt.pending_decision is not None
            assert rt.pending_decision.action == "approve"

            # Проверка отмены активной задачи
            rt.active_task = asyncio.create_task(asyncio.sleep(10.0))
            rt.status = "running"
            await bridge.cancel_current_run("sess_hitl")
            assert rt.status == "cancelled"

        asyncio.run(_run_test())


class TestScenarioArtifactExports:
    """Проверка восстановления сохранённых сценариев из `workspace/`."""

    @pytest.mark.parametrize(
        "scenario_id",
        ["scenario1_btk", "scenario2_gefitinib", "scenario3_wetlab"],
    )
    def test_restore_saved_scenarios(self, scenario_id: str) -> None:
        if not (settings.resolved_workspace_dir / scenario_id).is_dir():
            pytest.skip(f"Scenario {scenario_id} not present in workspace")

        reg = SessionRegistry()
        rt = asyncio.run(reg.restore_from_workspace(scenario_id))
        assert rt is not None
        assert len(rt.view_model.rows_by_id) > 0
        assert len(rt.state.get("trace_log", [])) > 0


class TestUIComponentAndCheckpointFixes:
    """Регрессионные тесты для 3D-вьюера и строгой msgpack-сериализации чекпоинтов."""

    def test_viewer3d_clear_methods_match_js_component(self) -> None:
        from pishovichok.ui.components.chem_widgets import MoleculeViewer3D

        js_text = (_SRC_ROOT / "ui" / "static" / "viewer3d.js").read_text(
            encoding="utf-8"
        )
        assert "clear()" in js_text
        assert "clearViewer()" in js_text

        called: list[tuple[str, tuple[object, ...]]] = []

        class _FakeViewer(MoleculeViewer3D):
            def __init__(self) -> None:
                pass

            def run_method(self, name: str, *args: object, **_kwargs: object) -> None:  # type: ignore[override]
                called.append((name, args))

        viewer = _FakeViewer()
        viewer.clear_viewer()
        viewer.clear()
        assert called == [("clearViewer", ()), ("clearViewer", ())]

    def test_checkpoint_serializer_allows_all_schema_models_without_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        from pishovichok.state.schemas import (
            ActivityCliff,
            DatasetCurationSummary,
            ParquetPointer,
            QueryConstraints,
            SARRule,
            TraceEvent,
        )
        from pishovichok.ui.bridge import _ALLOWED_CHECKPOINT_MODULES

        serde = JsonPlusSerializer(
            allowed_msgpack_modules=_ALLOWED_CHECKPOINT_MODULES
        )
        saver = MemorySaver(serde=serde)
        cand = _make_dummy_candidate("cand_ckpt_1")
        payload = {
            "query_constraints": QueryConstraints(scenario_type="target_based_sar"),
            "dataset_uri": ParquetPointer(
                uri="/tmp/curated.parquet",
                table_type="curated_dataset",
                row_count=1,
                columns=[
                    "mol_id",
                    "canonical_smiles",
                    "inchi_key",
                    "scaffold_smiles",
                    "pchembl_value",
                    "pchembl_std",
                    "replicate_count",
                    "is_censored",
                    "assay_type",
                    "bao_format",
                    "standard_type",
                    "tanimoto_to_lead",
                ],
                sha256_checksum="0" * 64,
            ),
            "curation_summary": DatasetCurationSummary(
                source_type="chembl_parquet",
                initial_rows_count=1,
                invalid_smiles_dropped=0,
                salts_stripped_count=0,
                duplicates_merged_count=0,
                censored_values_count=0,
                curated_molecules_count=1,
                dominant_bao_format="BAO_0000357",
                dominant_standard_type="IC50",
                bao_distribution={"BAO_0000357": 1},
                is_dataset_assay_homogeneous=True,
            ),
            "extracted_sar_rules": [
                SARRule(
                    rule_id="R1",
                    smirks="[*:1]C>>[*:1]OC",
                    from_fragment="*C",
                    to_fragment="*OC",
                    mean_delta_pic50=0.5,
                    support_pairs_count=2,
                    mean_delta_clogp=-0.1,
                    mean_delta_tpsa=9.2,
                    assay_homogeneous=True,
                    participating_bao_formats=["BAO_0000357"],
                )
            ],
            "detected_activity_cliffs": [
                ActivityCliff(
                    cliff_id="CL1",
                    mol_id_active="M1",
                    mol_id_inactive="M2",
                    smiles_active="COc1ccccc1",
                    smiles_inactive="c1ccccc1",
                    pic50_active=8.0,
                    pic50_inactive=6.0,
                    abs_delta_pic50=2.0,
                    tanimoto_similarity=0.9,
                    sali_index=20.0,
                    smirks_difference="[cH:1]>>[c:1]OC",
                    assay_homogeneous=True,
                )
            ],
            "top_candidates": [cand],
            "trace_log": [
                TraceEvent(
                    step_index=0,
                    timestamp_iso="2026-01-01T00:00:00Z",
                    agent_name="ResearchAgent",
                    phase="init",
                    input_summary="in",
                    action_taken="act",
                    output_summary="out",
                    duration_ms=1.0,
                )
            ],
        }
        with caplog.at_level("WARNING"):
            blob = saver.serde.dumps_typed(payload)
            restored = saver.serde.loads_typed(blob)

        assert isinstance(restored["query_constraints"], QueryConstraints)
        assert isinstance(restored["top_candidates"][0], CandidateMolecule)
        assert "Deserializing unregistered type" not in caplog.text
        assert "Blocked deserialization" not in caplog.text

    def test_candidate_grid_uses_valid_nicegui_aggrid_theme(self) -> None:
        grid_src = (_SRC_ROOT / "ui" / "components" / "candidate_grid.py").read_text(
            encoding="utf-8"
        )
        assert 'theme="quartz"' in grid_src
        assert "quartz-dark" not in grid_src

    def test_mpo_canvas_plots_lead_in_pareto_and_pca_without_lead_reference(self) -> None:
        from pishovichok.ui.components.mpo_canvas import MPOCanvas
        from pishovichok.ui.state_vm import ReferencePointVM

        reg = SessionRegistry()
        rt = reg.get_or_create("sess_lead_plot")
        cand = _make_dummy_candidate("cand_lead_1")
        rt.view_model.reference_points = [
            ReferencePointVM(
                mol_id=cand.parent_chembl_id,
                canonical_smiles=cand.parent_smiles,
                pchembl_value=cand.parent_exp_pic50,
                clogp=3.4,
                tpsa=38.0,
                source_group="chembl",
                pca_x=-0.5,
                pca_y=0.25,
            )
        ]
        rt.view_model.upsert_candidates([cand])

        canvas = object.__new__(MPOCanvas)
        canvas.runtime = rt
        canvas._mode = "pareto"
        opts_pareto = canvas._build_chart_options()
        lead_series_pareto = next(
            s for s in opts_pareto["series"] if s["name"] == "Lead"
        )
        assert len(lead_series_pareto["data"]) == 1
        assert lead_series_pareto["data"][0]["name"] == f"Lead ({cand.parent_chembl_id})"

        canvas._mode = "pca"
        opts_pca = canvas._build_chart_options()
        lead_series_pca = next(s for s in opts_pca["series"] if s["name"] == "Lead")
        assert len(lead_series_pca["data"]) == 1
        assert lead_series_pca["data"][0]["value"] == [-0.5, 0.25]

    def test_clean_startup_and_clear_session_button(self) -> None:
        app_src = (_SRC_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
        assert 'restore_from_workspace("scenario1_btk")' not in app_src

        from pishovichok.ui.components.agent_rail import TopStatusBar

        reg = SessionRegistry()
        rt = reg.get_or_create("sess_to_clear")
        switched: list[str] = []

        bar = object.__new__(TopStatusBar)
        bar.runtime = rt
        bar._on_session_switched = lambda sid: switched.append(sid)

        async def _run() -> None:
            await bar._handle_clear_session()

        asyncio.run(_run())
        assert len(switched) == 1
        assert switched[0] != "sess_to_clear"
