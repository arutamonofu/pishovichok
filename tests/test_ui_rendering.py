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
    ToolConfig,
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
            ],
            tool_config=ToolConfig(),
        )
        assert len(coords) == 3
        assert all(len(pt) == 2 for pt in coords)

        tool_cfg = ToolConfig()
        molblock_kw, hl_kw = generate_3d_conformer_molblock(
            "COc1ccc(Nc2ncnc3ccccc23)cc1",
            num_conformers=tool_cfg.num_conformers,
            random_seed=tool_cfg.conformer_random_seed,
            tool_config=tool_cfg,
        )
        assert "V2000" in molblock_kw or "V3000" in molblock_kw
        assert hl_kw == []
        assert pickle.loads(pickle.dumps(tool_cfg)) == tool_cfg

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

    def test_restore_workspace_with_null_docking_parquet(
        self, tmp_path: Path
    ) -> None:
        from pishovichok.state.storage import write_candidates_parquet

        sess_dir = tmp_path / "sess_null_docking"
        sess_dir.mkdir(parents=True, exist_ok=True)
        cand = _make_dummy_candidate("CAND_NULL_DOCK_01")
        assert cand.docking is None
        write_candidates_parquet([cand], sess_dir, "evaluated_candidates.parquet")

        reg = SessionRegistry(base_workspace_dir=tmp_path)
        rt = asyncio.run(reg.restore_from_workspace("sess_null_docking"))
        assert rt is not None
        assert cand.candidate_id in rt.view_model.rows_by_id
        assert rt.view_model.candidates_by_id[cand.candidate_id].docking is None



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
            s for s in opts_pareto["series"] if s["name"] == "Прототип"
        )
        assert len(lead_series_pareto["data"]) == 1
        assert lead_series_pareto["data"][0]["name"] == f"Прототип ({cand.parent_chembl_id})"

        canvas._mode = "pca"
        opts_pca = canvas._build_chart_options()
        lead_series_pca = next(s for s in opts_pca["series"] if s["name"] == "Прототип")
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


class TestToolSettingsAndTelemetry:
    """Тесты синхронизации ToolConfig <-> UI, 4 режимов перезапуска и телеметрии."""

    def test_tool_config_ui_sync_and_mpo_parity_with_core_scoring(self) -> None:
        import time

        from pishovichok.core.scoring import calculate_harrington_mpo_score
        from pishovichok.state.graph_state import resolve_tool_config
        from pishovichok.state.schemas import QueryConstraints, ToolConfig

        reg = SessionRegistry()
        bridge = AgentUIBridge(registry=reg)
        rt = reg.get_or_create("sess_sync_mpo")

        cand_clean = _make_dummy_candidate("c_clean", pic50=8.4, clogp=2.7)
        cand_alert = _make_dummy_candidate("c_alert", pains_count=1, pic50=8.8, clogp=2.5)
        rt.view_model.upsert_candidates([cand_clean, cand_alert])

        # Синхронизация весов и порогов Харрингтона в ToolConfig и обратно в ViewModel
        overrides = {
            "mpo_weights": {"pic50": 2.5, "herg": 1.8},
            "cns_profile_mode": "cns_high_bbb",
            "alert_penalty_mode": "soft_penalty",
            "alert_soft_multiplier": 0.4,
            "harrington_qed_min": 0.45,
            "harrington_activity_high_delta": 0.85,
            "protected_smarts": "c1ncnc2ccccc12",
            "modifiable_atom_indices": [0, 3, 7],
        }
        changed = bridge.apply_ui_tool_overrides(rt, overrides)
        assert set(overrides).issubset(set(changed))
        for k in overrides:
            assert rt.state["param_sources"][k] == "ui_manual"

        tc = resolve_tool_config(rt.state)
        assert tc.mpo_weights["pic50"] == 2.5
        assert rt.view_model.weights.w_activity == 2.5
        assert rt.view_model.weights.w_herg == 1.8
        assert rt.view_model.weights.cns_profile_mode == "cns_high_bbb"
        assert rt.view_model.weights.alert_penalty_mode == "soft_penalty"
        assert rt.view_model.weights.alert_soft_multiplier == 0.4

        # Локальный пересчёт MPO через core/scoring.py (< 50 мс) и точный паритет с calculate_harrington_mpo_score
        t0 = time.perf_counter()
        rt.view_model.recalculate_mpo()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0

        constraints = QueryConstraints(scenario_type="target_based_sar")
        expected_clean = calculate_harrington_mpo_score(
            cand_clean.calc_rdkit,
            cand_clean.pred_admet,
            cand_clean.parent_exp_pic50,
            constraints,
            tool_config=tc,
        )
        expected_alert = calculate_harrington_mpo_score(
            cand_alert.calc_rdkit,
            cand_alert.pred_admet,
            cand_alert.parent_exp_pic50,
            constraints,
            tool_config=tc,
        )
        assert rt.view_model.rows_by_id["c_clean"].mpo_score == pytest.approx(expected_clean)
        assert rt.view_model.rows_by_id["c_alert"].mpo_score == pytest.approx(expected_alert)
        assert rt.view_model.rows_by_id["c_alert"].mpo_score > 0.0

        # Явная очистка опциональных полей (protected_smarts=None, modifiable_atom_indices=None)
        cleared = bridge.apply_ui_tool_overrides(
            rt, {"protected_smarts": None, "modifiable_atom_indices": None}
        )
        assert set(cleared) == {"protected_smarts", "modifiable_atom_indices"}
        tc_after = resolve_tool_config(rt.state)
        assert tc_after.protected_smarts is None
        assert tc_after.modifiable_atom_indices is None

        # Сброс к дефолтам
        asyncio.run(bridge.reset_tool_config_to_defaults(rt))
        assert resolve_tool_config(rt.state) == ToolConfig()
        assert all(v == "default" for v in rt.state["param_sources"].values())
        assert rt.view_model.rows_by_id["c_alert"].mpo_score == 0.0

    def test_apply_mode_classification_and_all_tool_config_fields_covered(self) -> None:
        from pishovichok.state.schemas import ToolConfig
        from pishovichok.ui.state_vm import classify_tool_config_change_impact

        assert classify_tool_config_change_impact(["mpo_weights", "harrington_qed_min"]) == "instant_mpo"
        assert classify_tool_config_change_impact(["cns_profile_mode", "alert_penalty_mode"]) == "instant_mpo"
        assert classify_tool_config_change_impact(["etkdg_variant", "force_field"]) == "instant_mpo"
        assert classify_tool_config_change_impact(["active_filter_catalogs", "admet_backend"]) == "evaluation"
        assert classify_tool_config_change_impact(["custom_ban_smarts", "extra_rdkit_descriptors"]) == "evaluation"
        assert classify_tool_config_change_impact(["brics_min_fragment_size", "lipinski_mw_cutoff"]) == "evaluation"
        assert classify_tool_config_change_impact(["crem_mode", "protected_smarts"]) == "design"
        assert classify_tool_config_change_impact(["fp_type", "cliff_similarity_cutoff"]) == "sar"
        assert classify_tool_config_change_impact(["allowed_bao_formats", "strip_salts"]) == "research"
        # При смешанном изменении выбирается наиболее ранний узел графа
        assert classify_tool_config_change_impact(["mpo_weights", "fp_type", "strip_salts"]) == "research"

        # Проверка, что все 83 поля ToolConfig привязаны в tool_settings.py
        ts_src = (_SRC_ROOT / "ui" / "components" / "tool_settings.py").read_text(encoding="utf-8")
        missing = [
            field_name
            for field_name in ToolConfig.model_fields
            if f'"{field_name}"' not in ts_src
        ]
        assert missing == []

    def test_restart_branch_supports_four_stages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import pishovichok.ui.bridge as bridge_mod
        from pishovichok.state.schemas import ParquetPointer

        reg = SessionRegistry(base_workspace_dir=tmp_path)
        bridge = AgentUIBridge(registry=reg)
        rt = reg.get_or_create("sess_stages")

        # Подготовка артефактов на диске для частичных перезапусков
        curated_path = rt.workspace_dir / "curated_dataset.parquet"
        curated_path.write_bytes(b"parquet-placeholder")
        (rt.workspace_dir / "raw_candidates_it1.json").write_text("[]", encoding="utf-8")
        rt.state["dataset_uri"] = ParquetPointer(
            uri=str(curated_path),
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
        )

        recorded: list[tuple[str, str | None]] = []

        async def _fake_loop(runtime_arg: object, resume_from_stage: str | None = None) -> None:
            recorded.append((getattr(runtime_arg, "session_id", ""), resume_from_stage))

        def _fake_bg_create(coro: object, *, name: str = "") -> asyncio.Task[None]:
            return asyncio.create_task(coro, name=name)  # type: ignore[arg-type]

        monkeypatch.setattr(bridge, "_run_graph_loop", _fake_loop)
        monkeypatch.setattr(bridge_mod.background_tasks, "create", _fake_bg_create)

        async def _run_all() -> None:
            for stage in ("evaluation", "design", "sar", "research"):
                await bridge.restart_branch(
                    "sess_stages",
                    target_stage=stage,  # type: ignore[arg-type]
                    tool_config_patch={"admet_backend": "empirical_only"},
                )
                if rt.active_task is not None:
                    await rt.active_task

        asyncio.run(_run_all())
        assert [st for _, st in recorded] == ["evaluation", "design", "sar", "research"]
        assert rt.state["param_sources"]["admet_backend"] == "ui_manual"

    def test_smarts_and_atom_indices_validation_and_descriptors_list(self) -> None:
        from pishovichok.ui.components.tool_settings import (
            ALL_RDKIT_DESCRIPTOR_NAMES,
            format_modifiable_atom_indices,
            parse_modifiable_atom_indices,
            validate_smarts_pattern,
        )

        assert "MolWt" in ALL_RDKIT_DESCRIPTOR_NAMES
        assert len(ALL_RDKIT_DESCRIPTOR_NAMES) > 150

        ok_valid, cleaned = validate_smarts_pattern("  c1ncnc2ccccc12  ")
        assert ok_valid is True and cleaned == "c1ncnc2ccccc12"
        ok_empty, cleaned_empty = validate_smarts_pattern("   ")
        assert ok_empty is True and cleaned_empty == ""
        ok_bad, err_msg = validate_smarts_pattern("[invalid_smarts(((")
        assert ok_bad is False and "SMARTS" in err_msg

        ok_idx, parsed_idx, _ = parse_modifiable_atom_indices("0, 2, 5, 2")
        assert ok_idx is True and parsed_idx == [0, 2, 5]
        assert format_modifiable_atom_indices(parsed_idx) == "0, 2, 5"
        ok_idx_empty, parsed_empty, _ = parse_modifiable_atom_indices("  ")
        assert ok_idx_empty is True and parsed_empty is None
        ok_idx_bad, _, err_idx = parse_modifiable_atom_indices("1, abc")
        assert ok_idx_bad is False and "abc" in err_idx

    def test_node_telemetry_and_panel_banners_rendering(self) -> None:
        from pishovichok.state.schemas import ToolConfig
        from pishovichok.ui.components.agent_rail import format_node_telemetry_chips
        from pishovichok.ui.components.candidate_grid import build_grid_telemetry_badges
        from pishovichok.ui.components.inspector import (
            build_inspector_telemetry_summary,
        )

        telemetry = {
            "active_params": {
                "fp_type": "fcfp",
                "similarity_metric": "dice",
                "protected_smarts": "c1ccccc1",
                "mmpa_max_cuts": 1,
                "max_candidates_pool": 40,
            },
            "stats": {
                "rules_count": 6,
                "activity_cliffs_count": 2,
                "duration_ms": 12.3,
            },
        }
        param_sources = {
            "fp_type": "nl_query",
            "similarity_metric": "nl_query",
            "protected_smarts": "ui_manual",
            "max_candidates_pool": "critic_loop",
            "mmpa_max_cuts": "default",
        }
        param_chips, stat_chips = format_node_telemetry_chips(telemetry, param_sources)  # type: ignore[arg-type]
        chip_map = {c["key"]: c for c in param_chips}
        assert chip_map["fp_type"]["color"] == "sky"
        assert chip_map["protected_smarts"]["color"] == "blue-7"
        assert chip_map["max_candidates_pool"]["color"] == "amber"
        assert chip_map["mmpa_max_cuts"]["color"] == "slate"
        assert ("rules_count", "6") in stat_chips
        assert ("activity_cliffs_count", "2") in stat_chips

        tc = ToolConfig(
            fp_type="fcfp",
            fp_radius=3,
            fp_nbits=4096,
            similarity_metric="dice",
            admet_backend="empirical_only",
            active_filter_catalogs=["PAINS_A", "Brenk", "Glaxo"],
            alert_penalty_mode="soft_penalty",
            brics_min_fragment_size=4,
            etkdg_variant="ETKDGv2",
            force_field="UFF",
        )
        grid_badges = build_grid_telemetry_badges(tc)
        joined_grid = " | ".join(lbl for lbl, _ in grid_badges)
        assert "fcfp/r3/4096" in joined_grid
        assert "dice" in joined_grid
        assert "empirical_only" in joined_grid
        assert "Glaxo" in joined_grid
        assert "soft_penalty" in joined_grid
        assert "BRICS≥4" in joined_grid

        insp_telem = build_inspector_telemetry_summary(tc)
        assert "ETKDGv2 + UFF" in insp_telem["conformer_3d"]
        assert "empirical_only" in insp_telem["admet"]
        assert "fcfp/r3/4096" in insp_telem["sar_fp"]
        assert "BRICS≥4" in insp_telem["rdkit_filters"]


class TestWetLabLightModeRefactoring:
    """Тесты Clean Wet-Lab Light Mode, 4-категорийных колонок, маршрутов SVG и контекстной CTA."""

    def test_light_mode_rendering_and_pastel_highlights(self) -> None:
        from pishovichok.core.rendering import (
            COLOR_ACTIVITY_GAIN,
            COLOR_ACTIVITY_LOSS,
            render_molecule_svg,
            render_parent_leaving_svg,
            render_rule_transformation_svg,
        )

        assert COLOR_ACTIVITY_GAIN == (0.73, 0.97, 0.82)
        assert COLOR_ACTIVITY_LOSS == (0.99, 0.80, 0.83)

        svg = render_molecule_svg("c1ccccc1O", width=260, height=200)
        assert "<svg" in svg
        assert "#FFFFFF" in svg or "#ffffff" in svg.lower()

        parent_svg = render_parent_leaving_svg("c1ccccc1Cl", "c1ccccc1F")
        assert "<svg" in parent_svg

        rule_svg = render_rule_transformation_svg("*Cl", "*F", delta_pic50=0.45)
        assert "<svg" in rule_svg

    def test_brics_and_rule_svg_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from nicegui import run as nicegui_run

        from pishovichok.state.schemas import SARRule
        from pishovichok.ui.routes import router as api_router
        from pishovichok.ui.session import SessionRegistry

        sid = "sess_wetlab_routes"
        registry = SessionRegistry()
        monkeypatch.setattr("pishovichok.ui.routes.session_registry", registry)
        rt = registry.get_or_create(sid)
        cand = _make_dummy_candidate("C_BRICS")
        rule = SARRule(
            rule_id="R1",
            smirks="[*:1]O>>[*:1]OC",
            from_fragment="*O",
            to_fragment="*OC",
            mean_delta_pic50=0.55,
            support_pairs_count=7,
            mean_delta_clogp=0.2,
            mean_delta_tpsa=-9.2,
            assay_homogeneous=True,
            participating_bao_formats=["BAO_0000357"],
        )
        rt.view_model.upsert_candidates([cand])
        rt.state["extracted_sar_rules"] = [rule]

        nicegui_run.setup()
        try:
            app = FastAPI()
            app.include_router(api_router)
            client = TestClient(app)

            r_parent = client.get(f"/api/sessions/{sid}/mol/C_BRICS/svg?variant=parent")
            assert r_parent.status_code == 200
            assert "<svg" in r_parent.text

            r_brics = client.get(f"/api/sessions/{sid}/mol/C_BRICS/brics/0/svg")
            assert r_brics.status_code == 200
            assert "<svg" in r_brics.text

            r_rule = client.get(f"/api/sessions/{sid}/rule/R1/svg")
            assert r_rule.status_code == 200
            assert "<svg" in r_rule.text
        finally:
            nicegui_run.tear_down()

    def test_theme_tokens_and_4_group_aggrid_columns(self) -> None:
        from pishovichok.ui.components.candidate_grid import build_candidate_column_defs
        from pishovichok.ui.state_vm import StateViewModel
        from pishovichok.ui.theme import THEME_CSS, build_theme_head_css

        css = build_theme_head_css()
        assert THEME_CSS == css
        assert "#F8FAFC" in css
        assert "#7C3AED" in css
        assert "#059669" in css
        assert "#D97706" in css
        assert "#2563EB" in css

        cols = build_candidate_column_defs()
        assert len(cols) == 4  # 4 grouped categories (12 leaf columns)
        group_headers = [c["headerName"] for c in cols]
        assert group_headers == [
            "1. Эксперимент",
            "2. Расчёт RDKit",
            "3. Прогноз ADMET",
            "4. Гипотеза SAR",
        ]
        total_leaf_cols = sum(len(grp["children"]) for grp in cols)
        assert total_leaf_cols == 12
        struct_col = cols[0]["children"][0]
        assert struct_col["field"] == "preview_html"
        assert ":cellRenderer" in struct_col
        assert struct_col.get("pinned") == "left"
        for grp in cols:
            assert "children" in grp and len(grp["children"]) >= 2
            for child in grp["children"]:
                assert "minWidth" in child

        cand = _make_dummy_candidate("C1", pic50=8.23, clogp=3.89)
        vm = StateViewModel("s1")
        vm.upsert_candidates([cand])
        row = vm.rows_by_id["C1"].to_aggrid_row("s1")
        assert row["clogp_display"] == "3.89 (+0.10)"
        assert row["pred_pic50_display"] == "8.23 (+0.73)"
        assert row["alerts_display"] == "Чисто"
        assert row["method_rule_display"] == "CReM · rule_1"
        assert "BAO_0000357" in row["parent_bao_tooltip"]

    def test_mpo_canvas_light_thresholds_and_sali_tab(self) -> None:
        from pishovichok.state.schemas import ActivityCliff, QueryConstraints
        from pishovichok.ui.components.mpo_canvas import MPOCanvas
        from pishovichok.ui.state_vm import StateViewModel

        vm = StateViewModel("sess_mpo_light")
        vm.upsert_candidates([_make_dummy_candidate("C1", pic50=8.0, clogp=3.1)])
        vm.query_constraints = QueryConstraints(
            scenario_type="target_based_sar",
            target_chembl_id="CHEMBL5251",
            max_logp=3.5,
        )
        vm.weights.max_logp = 3.5
        vm.activity_cliffs = [
            ActivityCliff(
                cliff_id="AC1",
                mol_id_active="CHEMBL2",
                mol_id_inactive="CHEMBL1",
                smiles_active="c1ccccc1Cl",
                smiles_inactive="c1ccccc1",
                pic50_active=8.4,
                pic50_inactive=6.1,
                abs_delta_pic50=2.3,
                tanimoto_similarity=0.88,
                sali_index=19.2,
                smirks_difference="[cH:1]>>[c:1]Cl",
                assay_homogeneous=True,
            )
        ]

        from types import SimpleNamespace

        canvas = object.__new__(MPOCanvas)
        canvas._mode = "pareto"
        canvas.runtime = SimpleNamespace(
            view_model=vm,
            state={
                "query_constraints": vm.query_constraints,
                "detected_activity_cliffs": vm.activity_cliffs,
            },
        )

        pareto_opts = canvas._build_chart_options()
        assert pareto_opts["backgroundColor"] == "#FFFFFF"
        ok_series = next(s for s in pareto_opts["series"] if s.get("name") == "Кандидаты (допустимые)")
        assert "markLine" in ok_series
        assert "markArea" in ok_series

        canvas._mode = "sali"
        sali_opts = canvas._build_chart_options()
        assert sali_opts["xAxis"]["name"] == "Сходство (Tanimoto)"
        assert sali_opts["yAxis"]["name"] == "|ΔpIC50|"
        assert len(sali_opts["series"][1]["data"]) == 1

    def test_tool_settings_contextual_primary_cta(self) -> None:
        from pishovichok.ui.components.tool_settings import resolve_primary_cta_spec

        mode_empty, _, _, _ = resolve_primary_cta_spec([])
        assert mode_empty == "instant_mpo"

        mode_design, _, _, _ = resolve_primary_cta_spec(["crem_mode"])
        assert mode_design == "design"

        mode_eval, _, _, _ = resolve_primary_cta_spec(["admet_backend"])
        assert mode_eval == "evaluation"

        mode_sar, _, _, _ = resolve_primary_cta_spec(["mmpa_max_cuts"])
        assert mode_sar == "sar"

    def test_zero_emojis_and_banned_terms_in_ui(self) -> None:
        ui_files = list((_SRC_ROOT / "ui").rglob("*.py"))
        assert ui_files
        banned_emojis = [
            "🧪", "🔄", "✅", "⚠️", "✂️", "🌐", "📌", "🚫",
            "📁", "🧹", "🎛️", "⚡", "🧬", "👁", "⛔", "🔬",
        ]
        banned_terms = [
            "AG GRID",
            "PROVENANCE",
            "Dropzone",
            "Wet-Lab CSV Dropzone",
            "Chemical IDE",
        ]
        for path in ui_files:
            content = path.read_text(encoding="utf-8")
            for emoji in banned_emojis:
                assert emoji not in content, f"Found banned emoji {emoji!r} in {path}"
            for term in banned_terms:
                assert term not in content, f"Found banned term {term!r} in {path}"

