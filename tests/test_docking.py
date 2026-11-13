"""Тесты модуля универсального 3D-докинга и анализа белок-лигандных взаимодействий."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pishovichok.config import DEMO_RECEPTOR_PRESETS
from pishovichok.core.docking import run_docking
from pishovichok.core.evaluation import evaluate_docking_batch
from pishovichok.core.receptor_resolver import (
    generate_minimal_demo_receptor,
    resolve_receptor,
)
from pishovichok.core.scoring import (
    calculate_harrington_mpo_score,
    compute_harrington_mpo_breakdown,
)
from pishovichok.state.schemas import (
    CalcRDKitMetrics,
    CandidateMolecule,
    DockingResult,
    PredADMETMetrics,
    QueryConstraints,
    SARHypothesisRationale,
    ToolConfig,
)
from pishovichok.state.storage import (
    read_candidates_parquet,
    write_candidates_parquet,
)


def _sample_candidate(cid: str = "CAND_DOCK_01") -> CandidateMolecule:
    return CandidateMolecule(
        candidate_id=cid,
        canonical_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        scaffold_smiles="c1ccc2ncncc2c1",
        generation_iteration=1,
        parent_chembl_id="CHEMBL939",
        parent_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        parent_exp_pic50=7.5,
        parent_exp_ic50_nm=31.6,
        parent_bao_format="BAO_0000357",
        parent_assay_type="B",
        parent_source_dataset="ChEMBL_Parquet",
        calc_rdkit=CalcRDKitMetrics(
            mw=446.9,
            clogp=3.8,
            tpsa=68.7,
            hbd=1,
            hba=7,
            rotatable_bonds=8,
            fsp3=0.36,
            qed=0.52,
            sa_score=2.6,
            lipinski_violations=0,
            pains_alerts_count=0,
            brenk_alerts_count=0,
            brics_fragments=["c1cc(F)c(Cl)cc1", "c1ncnc2ccccc12"],
            delta_clogp_vs_parent=0.0,
            delta_tpsa_vs_parent=0.0,
        ),
        pred_admet=PredADMETMetrics(
            predicted_pic50=7.9,
            herg_liability_prob=0.25,
            herg_risk_level="low",
            log_s_aq=-3.2,
            solubility_class="moderate",
            bbb_permeability_prob=0.20,
            cyp3a4_inhibition_prob=0.18,
            model_source="catboost_local_qsar",
            prediction_confidence="high",
        ),
        hypothesis_rationale=SARHypothesisRationale(
            rule_id="RULE_DOCK",
            applied_smirks="[*:1]Cl>>[*:1]F",
            replaced_group_smiles="*Cl",
            introduced_group_smiles="*F",
            support_pairs_count=4,
            historical_mean_delta_pic50=0.4,
            assay_homogeneous=True,
            generation_origin="mmp_smirks_rule",
            text_summary="Тестовая гипотеза.",
        ),
        mpo_harrington_score=0.72,
        passes_all_constraints=True,
    )


def test_demo_cache_presets_resolve_without_network(tmp_path: Path) -> None:
    receptors_dir = tmp_path / "receptors"
    for target_id, preset in DEMO_RECEPTOR_PRESETS.items():
        meta = generate_minimal_demo_receptor(target_id, receptors_dir=receptors_dir)
        assert meta["status"] == "READY"
        assert meta["pdb_id"] == preset["pdb_id"]
        assert Path(meta["pdb_path"]).is_file()
        assert Path(meta["pdbqt_path"]).is_file()

        # Повторный вызов должен попадать в локальный кэш мгновенно
        cached = resolve_receptor(target_id, receptors_dir=receptors_dir)
        assert cached["status"] == "READY"
        assert cached["source"] == "DEMO_CACHE"
        assert cached["pdb_id"] == preset["pdb_id"]
        assert cached["box_center"] == preset["box_center"]


def test_resolve_arbitrary_target_from_mocked_chembl_and_rcsb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Минимальный PDB с водой (HOH), буфером (SO4), ко-лигандом (>=7 тяжелых атомов) и белком
    synthetic_pdb = "\n".join(
        [
            "ATOM      1  N   MET A 100      10.000  20.000  30.000  1.00 20.00           N",
            "ATOM      2  CA  MET A 100      11.000  20.000  30.000  1.00 20.00           C",
            "ATOM      3  C   MET A 100      12.000  20.000  30.000  1.00 20.00           C",
            "ATOM      4  O   MET A 100      12.500  21.000  30.000  1.00 20.00           O",
            "ATOM      5  SD  MET A 100      11.200  18.500  31.000  1.00 20.00           S",
            "HETATM    6  S   SO4 A 201      50.000  50.000  50.000  1.00 30.00           S",
            "HETATM    7  O1  SO4 A 201      51.000  50.000  50.000  1.00 30.00           O",
            "HETATM    8  O   HOH A 301      60.000  60.000  60.000  1.00 30.00           O",
            "HETATM    9  C1  LIG A 401      10.000  18.000  31.000  1.00 20.00           C",
            "HETATM   10  C2  LIG A 401      11.000  18.000  31.000  1.00 20.00           C",
            "HETATM   11  C3  LIG A 401      12.000  18.000  31.000  1.00 20.00           C",
            "HETATM   12  C4  LIG A 401      10.000  19.000  31.000  1.00 20.00           C",
            "HETATM   13  C5  LIG A 401      11.000  19.000  31.000  1.00 20.00           C",
            "HETATM   14  N1  LIG A 401      12.000  19.000  31.000  1.00 20.00           N",
            "HETATM   15  O1  LIG A 401      11.000  18.500  32.000  1.00 20.00           O",
            "END",
        ]
    )

    class _MockResponse:
        def __init__(self, status_code: int, text: str = "", json_data: Any = None) -> None:
            self.status_code = status_code
            self.text = text
            self._json = json_data

        def json(self) -> Any:
            return self._json

    class _MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _MockClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str) -> _MockResponse:
            if "chembl/api/data/target/CHEMBL999999.json" in url:
                return _MockResponse(
                    200,
                    json_data={
                        "target_components": [
                            {
                                "accession": "P12345",
                                "target_component_xrefs": [
                                    {"xref_src_db": "PDB", "xref_id": "9XYZ"}
                                ],
                            }
                        ]
                    },
                )
            if "files.rcsb.org/download/9XYZ.pdb" in url:
                return _MockResponse(200, text=synthetic_pdb)
            return _MockResponse(404)

        def post(self, url: str, json: Any = None) -> _MockResponse:
            return _MockResponse(200, json_data={"result_set": [{"identifier": "9XYZ"}]})

    monkeypatch.setattr(httpx, "Client", _MockClient)

    res = resolve_receptor("CHEMBL999999", receptors_dir=tmp_path / "receptors")
    assert res["status"] == "READY"
    assert res["pdb_id"] == "9XYZ"
    assert res["source"] == "RCSB_LIVE"
    assert res["ligand_resname"] == "LIG"
    assert res["is_apo"] is False
    # Центроид ко-лиганда: x ~ 11.0, y ~ 18.5, z ~ 31.14
    assert 10.5 <= res["box_center"][0] <= 11.5
    assert 18.0 <= res["box_center"][1] <= 19.0
    assert "MET100" in res["key_residues"]
    assert Path(res["pdbqt_path"]).is_file()
    pdbqt_txt = Path(res["pdbqt_path"]).read_text(encoding="utf-8")
    assert "HETATM" not in pdbqt_txt
    assert "OA" in pdbqt_txt


def test_resolve_receptor_no_structure_returns_safe_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _FailClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str) -> Any:
            raise httpx.ConnectError("offline")

        def post(self, url: str, json: Any = None) -> Any:
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "Client", _FailClient)
    res = resolve_receptor("CHEMBL_UNKNOWN_GPCR", receptors_dir=tmp_path / "receptors")
    assert res["status"] == "NO_STRUCTURE"

    dock_res = run_docking(
        "c1ccccc1O",
        "CHEMBL_UNKNOWN_GPCR",
        receptors_dir=tmp_path / "receptors",
    )
    assert dock_res["status"] == "SKIPPED"


def test_run_docking_volume_barrier_and_salt_stripping_and_fallback(
    tmp_path: Path,
) -> None:
    receptors_dir = tmp_path / "receptors"
    generate_minimal_demo_receptor("CHEMBL203", receptors_dir=receptors_dir)

    # 1. Защитный барьер объёма бокса > 27000 Å³
    bad_box = run_docking(
        "c1ccccc1O",
        "CHEMBL203",
        box_size=[35.0, 35.0, 35.0],
        receptors_dir=receptors_dir,
    )
    assert bad_box["status"] == "FAILED"
    assert "27000" in str(bad_box.get("message", ""))

    # 2. Докинг молекулы с солью (.Cl) и сохранение позы в SDF
    sdf_out = tmp_path / "poses" / "test_pose.sdf"
    res = run_docking(
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1.Cl",
        "CHEMBL203",
        exhaustiveness=2,
        output_sdf_path=sdf_out.as_posix(),
        receptors_dir=receptors_dir,
    )
    assert res["status"] == "SUCCESS"
    assert res["pdb_id"] == "1M17"
    assert isinstance(res["docking_score_kcal"], float)
    assert res["docking_score_kcal"] < 0.0
    assert isinstance(res["interactions"], dict)
    assert sdf_out.is_file()
    assert "V2000" in sdf_out.read_text(encoding="utf-8")


def test_scoring_mpo_preserves_none_and_uses_docking_when_present() -> None:
    cand = _sample_candidate()
    constraints = QueryConstraints(scenario_type="target_based_sar", max_logp=4.5)
    tc = ToolConfig()

    base_score = calculate_harrington_mpo_score(
        cand.calc_rdkit,
        cand.pred_admet,
        cand.parent_exp_pic50,
        constraints,
        tool_config=tc,
    )
    none_score = calculate_harrington_mpo_score(
        cand.calc_rdkit,
        cand.pred_admet,
        cand.parent_exp_pic50,
        constraints,
        docking_score_kcal=None,
        tool_config=tc,
    )
    assert base_score == none_score

    strong_binder_score = calculate_harrington_mpo_score(
        cand.calc_rdkit,
        cand.pred_admet,
        cand.parent_exp_pic50,
        constraints,
        docking_score_kcal=-9.2,
        tool_config=tc,
    )
    weak_binder_score = calculate_harrington_mpo_score(
        cand.calc_rdkit,
        cand.pred_admet,
        cand.parent_exp_pic50,
        constraints,
        docking_score_kcal=-4.8,
        tool_config=tc,
    )
    assert strong_binder_score > weak_binder_score

    _, breakdown = compute_harrington_mpo_breakdown(
        predicted_pic50=cand.pred_admet.predicted_pic50,
        parent_exp_pic50=cand.parent_exp_pic50,
        clogp=cand.calc_rdkit.clogp,
        sa_score=cand.calc_rdkit.sa_score,
        qed=cand.calc_rdkit.qed,
        tpsa=cand.calc_rdkit.tpsa,
        herg_liability_prob=cand.pred_admet.herg_liability_prob,
        log_s_aq=cand.pred_admet.log_s_aq,
        has_alerts=False,
        min_logp=None,
        max_logp=4.5,
        max_sa_score=3.5,
        max_herg_prob=0.5,
        docking_score_kcal=-8.8,
        tool_config=tc,
    )
    assert "docking" in breakdown


def test_parquet_roundtrip_and_standalone_docking_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receptors_dir = tmp_path / "receptors"
    generate_minimal_demo_receptor("CHEMBL203", receptors_dir=receptors_dir)

    cand = _sample_candidate("C_DOCK_RT")
    constraints = QueryConstraints(
        scenario_type="target_based_sar",
        target_chembl_id="CHEMBL203",
        max_logp=4.5,
    )
    docked_list, summary = evaluate_docking_batch(
        [cand],
        target_id="CHEMBL203",
        top_k=1,
        exhaustiveness=2,
        workspace_dir=tmp_path,
        constraints=constraints,
        receptors_dir=receptors_dir,
    )
    assert summary["status"] == "SUCCESS"
    docked_cand = docked_list[0]
    assert isinstance(docked_cand.docking, DockingResult)
    assert docked_cand.docking.status == "SUCCESS"
    assert set(docked_cand.to_four_category_dict().keys()) == {
        "category_1_experiment",
        "category_2_calculation",
        "category_3_prediction",
        "category_4_hypothesis",
    }

    ptr = write_candidates_parquet([docked_cand], tmp_path, "evaluated_candidates.parquet")
    restored = read_candidates_parquet(ptr)
    assert len(restored) == 1
    assert restored[0].docking is not None
    assert restored[0].docking.pdb_id == "1M17"
    assert restored[0].docking.docking_score_kcal == docked_cand.docking.docking_score_kcal

    # Проверка ручного маршрута POST /api/sessions/{sid}/docking через Bridge
    from pishovichok.ui.bridge import AgentUIBridge
    from pishovichok.ui.routes import router as api_router
    from pishovichok.ui.session import SessionRegistry

    registry = SessionRegistry(base_workspace_dir=tmp_path)
    bridge = AgentUIBridge(registry=registry)
    monkeypatch.setattr("pishovichok.ui.routes.session_registry", registry)
    monkeypatch.setattr("pishovichok.ui.routes.agent_bridge", bridge)

    sid = "sess_manual_dock"
    rt = registry.get_or_create(sid, query_constraints=constraints)
    rt.workspace_dir = tmp_path / sid
    rt.workspace_dir.mkdir(parents=True, exist_ok=True)
    plain_cand = _sample_candidate("C_MANUAL_01")
    rt.state["query_constraints"] = constraints
    rt.state["candidates_table_uri"] = write_candidates_parquet(
        [plain_cand], rt.workspace_dir, "evaluated_candidates.parquet"
    )
    rt.state["top_candidates"] = [plain_cand]

    # Для быстрого теста подменяем resolve_receptors на уже подготовленный tmp receptors_dir
    import pishovichok.core.receptor_resolver as rr_mod

    monkeypatch.setattr(rr_mod, "RECEPTORS_DIR", receptors_dir)

    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)

    resp = client.post(
        f"/api/sessions/{sid}/docking",
        json={"candidate_ids": ["C_MANUAL_01"], "exhaustiveness": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "SUCCESS"
    assert rt.view_model.rows_by_id["C_MANUAL_01"].docking_status == "SUCCESS"
    assert docked_cand.docking.receptor_name == "EGFR (Epidermal Growth Factor Receptor)"
    assert docked_cand.docking.parent_docking_score_kcal is not None


def test_critic_node_records_poor_docking_affinity_in_feedback_and_trace() -> None:
    """Если докинг включён и ΔG > -6.0 ккал/моль, critic_node фиксирует это в feedback и TraceEvent."""
    from pishovichok.agents.critic_node import critic_node
    from pishovichok.state import create_initial_graph_state

    weak_dock = DockingResult(
        status="SUCCESS",
        pdb_id="1M17",
        receptor_name="EGFR (Epidermal Growth Factor Receptor)",
        source="DEMO_CACHE",
        docking_score_kcal=-5.2,
        parent_docking_score_kcal=-7.8,
        backend="RDKit-MMFF94-3D",
    )
    cand = _sample_candidate("C_WEAK_DOCK").model_copy(update={"docking": weak_dock})

    # 1. Итерация < max_iterations -> redesign_molecules с фиксацией докинга
    state_redesign = create_initial_graph_state(
        user_query="Оптимизируй ингибитор EGFR CHEMBL203",
        session_id="critic_dock_redesign",
        workspace_dir="/tmp",
    )
    state_redesign["last_executed_stage"] = "evaluation"
    state_redesign["design_iteration"] = 1
    state_redesign["top_candidates"] = [cand]
    state_redesign["tool_config"] = ToolConfig(docking_mode="auto_top_k", max_iterations=2)

    out_redesign = critic_node(state_redesign)
    assert out_redesign["critic_verdict"] == "redesign_molecules"
    assert "-6.0" in out_redesign["critic_feedback"]
    assert "3D-докинг" in out_redesign["critic_feedback"]
    assert out_redesign["node_telemetry"]["critic"]["stats"]["poor_docking_affinity"] is True
    ev_redesign = out_redesign["trace_log"][0]
    assert "Weak docking affinity" in ev_redesign.action_taken
    assert "-6.0" in ev_redesign.output_summary

    # 2. Исчерпан лимит итераций -> approve, но с предупреждением о докинге в feedback и трассе
    state_approve = create_initial_graph_state(
        user_query="Оптимизируй ингибитор EGFR CHEMBL203",
        session_id="critic_dock_approve",
        workspace_dir="/tmp",
    )
    state_approve["last_executed_stage"] = "evaluation"
    state_approve["design_iteration"] = 2
    state_approve["top_candidates"] = [cand]
    state_approve["tool_config"] = ToolConfig(docking_mode="auto_top_k", max_iterations=2)

    out_approve = critic_node(state_approve)
    assert out_approve["critic_verdict"] == "approve"
    assert "-6.0" in out_approve["critic_feedback"]
    assert "3D-докинг" in out_approve["critic_feedback"]
    ev_approve = out_approve["trace_log"][0]
    assert "Weak 3D docking binding energy" in ev_approve.action_taken or "weak 3D docking binding energy" in ev_approve.action_taken
    assert "-6.0" in ev_approve.output_summary
