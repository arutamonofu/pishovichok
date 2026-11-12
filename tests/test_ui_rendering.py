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
