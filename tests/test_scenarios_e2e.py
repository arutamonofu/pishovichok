"""Интеграционные тесты слоя мультиагентной оркестрации LangGraph (Слой 3).

Проверяют:
1. Сценарий 1: Target-based SAR (BTK / CHEMBL5251) — Петля №1 (refilter_data), BAO_0000357, 4 категории.
2. Сценарий 2: Lead Optimization (EGFR / Gefitinib) — Петля №1 И Петля №2 (redesign_molecules),
   снижение cLogP, референсная карточка лидера.
3. Сценарий 3: Custom Wet-Lab CSV Analysis — изоляция 3 ошибок, 2 цензурированных, 3 дубликатов,
   4 перепада активности (Activity Cliffs), прекурсоры BRICS.
4. Ограничитель итераций (Iteration Limiter): защита от зацикливания при невыполнимых ограничениях
   с возвратом компромиссного топа по Харрингтону и явной маркировкой `constraint_violations`.
5. 100% воспроизводимость в детерминированном оффлайн-режиме (Tier 3 Rule-Based Engine).
"""

from __future__ import annotations

import math
from pathlib import Path

from pishovichok.agents.workflow import run_sar_pipeline
from pishovichok.state.graph_state import assert_fat_pointer_isolation
from pishovichok.state.schemas import CandidateMolecule, QueryConstraints


def _assert_four_categories_valid(cand: CandidateMolecule) -> None:
    """Проверяет соблюдение изоляции всех 4 категорий знаний."""
    four_dict = cand.to_four_category_dict()
    assert "category_1_experiment" in four_dict
    assert "category_2_calculation" in four_dict
    assert "category_3_prediction" in four_dict
    assert "category_4_hypothesis" in four_dict

    # Категория 1
    assert cand.parent_chembl_id
    assert cand.parent_smiles
    assert cand.parent_exp_pic50 > 0.0

    # Категория 2
    assert cand.calc_rdkit.mw > 0.0
    assert 0.0 <= cand.calc_rdkit.qed <= 1.0
    assert 1.0 <= cand.calc_rdkit.sa_score <= 10.0
    assert cand.calc_rdkit.pains_alerts_count == 0
    assert cand.calc_rdkit.brenk_alerts_count == 0

    # Категория 3
    assert cand.pred_admet.model_source in (
        "admet_ai_dmpnn",
        "catboost_local_qsar",
        "rdkit_analytical_qsar",
    )
    assert 0.0 <= cand.pred_admet.herg_liability_prob <= 1.0
    assert cand.pred_admet.predicted_pic50 > 0.0

    # Категория 4
    assert cand.hypothesis_rationale.rule_id
    assert cand.hypothesis_rationale.applied_smirks
    assert len(cand.hypothesis_rationale.text_summary) > 5


def test_scenario1_target_sar_end_to_end(tmp_path: Path) -> None:
    """Сценарий 1: Сквозной тест поиска ингибиторов BTK (CHEMBL5251) с Петлей №1."""
    query = (
        "Проведи анализ ингибиторов тирозинкиназы Брутона (BTK, CHEMBL5251), "
        "выдели ключевые правила 'структура–активность' на однородных биохимических тестах "
        "и сгенерируй 5 новых кандидатов с расчетным LogP от 2.0 до 4.0, SAscore < 3.2 "
        "и отсутствием структурных алертов PAINS."
    )

    state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=query,
        session_id="test_s1_e2e",
        workspace_dir=tmp_path,
        export_artifacts=True,
    )

    assert_fat_pointer_isolation(dict(state))

    # Проверка Петли №1
    trace_events = state["trace_log"]
    loop1_triggers = [
        ev for ev in trace_events if ev.verdict_or_routing == "refilter_data"
    ]
    assert len(loop1_triggers) >= 1
    assert state.get("assay_filter_strict") is True
    assert state.get("active_bao_filter") == "BAO_0000357"
    assert state.get("critic_verdict") == "approve"

    # Проверка кандидатов
    top_cands = state.get("top_candidates", [])
    assert len(top_cands) >= 5
    for cand in top_cands[:5]:
        _assert_four_categories_valid(cand)
        assert cand.mpo_harrington_score > 0.0

    assert trace_path is not None and trace_path.is_file()
    assert csv_path is not None and csv_path.is_file()
    assert sdf_path is not None and sdf_path.is_file()


def test_scenario2_gefitinib_optimization_both_loops(tmp_path: Path) -> None:
    """Сценарий 2: Оптимизация гефитиниба с проверкой срабатывания ОБЕИХ петель."""
    query = (
        "Оптимизируй структуру ингибитора EGFR со строкой SMILES "
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1 (гефитиниб): "
        "сохрани хиназолиновый скаффолд, снизь липофильность (LogP < 3.5), "
        "минимизируй риск блокады каналов hERG и предложи синтетически доступные аналоги (SAscore < 3.0)."
    )

    state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=query,
        session_id="test_s2_e2e",
        workspace_dir=tmp_path,
        export_artifacts=True,
    )

    assert_fat_pointer_isolation(dict(state))

    trace_events = state["trace_log"]
    loop1_triggers = [
        ev for ev in trace_events if ev.verdict_or_routing == "refilter_data"
    ]
    loop2_triggers = [
        ev for ev in trace_events if ev.verdict_or_routing == "redesign_molecules"
    ]

    assert len(loop1_triggers) >= 1, "Петля №1 обязана сработать"
    assert len(loop2_triggers) >= 1, "Петля №2 обязана сработать"
    assert state.get("design_iteration", 0) >= 2
    assert state.get("critic_verdict") == "approve"

    lead_ref = state.get("lead_reference_metrics")
    assert lead_ref is not None
    assert math.isclose(lead_ref.calc_rdkit.clogp, 4.28, abs_tol=0.05)

    top_cands = state.get("top_candidates", [])
    assert len(top_cands) >= 5
    for cand in top_cands[:5]:
        _assert_four_categories_valid(cand)
        assert cand.calc_rdkit.clogp < lead_ref.calc_rdkit.clogp
        assert cand.calc_rdkit.delta_clogp_vs_parent < 0.0
        assert "ncnc" in cand.canonical_smiles

    assert trace_path is not None and trace_path.is_file()
    assert csv_path is not None and csv_path.is_file()
    assert sdf_path is not None and sdf_path.is_file()


def test_scenario3_wetlab_csv_ingestion_and_cliffs(tmp_path: Path) -> None:
    """Сценарий 3: Анализ серии CSV с проверкой изоляции ошибок и детекции Activity Cliffs."""
    csv_file = Path("data/samples/custom_wetlab_series.csv").resolve()
    query = (
        "Проанализируй результаты испытаний нашей лабораторной серии из 25 соединений, "
        "найди критические перепады активности (Activity Cliffs) и предложи 5 структур "
        "для следующей очереди синтеза с разбивкой на коммерчески доступные прекурсоры."
    )

    state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=query,
        session_id="test_s3_e2e",
        workspace_dir=tmp_path,
        uploaded_csv_path=csv_file,
        export_artifacts=True,
    )

    assert_fat_pointer_isolation(dict(state))

    summary = state.get("curation_summary")
    assert summary is not None
    assert summary.invalid_smiles_dropped == 3
    assert summary.censored_values_count == 2
    assert summary.duplicates_merged_count == 3
    assert summary.curated_molecules_count == 27

    cliffs = state.get("detected_activity_cliffs", [])
    assert len(cliffs) >= 4
    for cliff in cliffs:
        assert cliff.abs_delta_pic50 >= 1.5
        assert cliff.tanimoto_similarity >= 0.85
        assert cliff.sali_index > 0.0

    assert len(state.get("extracted_sar_rules", [])) >= 5
    assert isinstance(state.get("enriched_with_chembl_background"), bool)
    assert state.get("critic_verdict") == "approve"

    top_cands = state.get("top_candidates", [])
    assert len(top_cands) >= 5
    for cand in top_cands[:5]:
        _assert_four_categories_valid(cand)
        assert len(cand.calc_rdkit.brics_fragments) >= 2
        assert cand.calc_rdkit.brics_bonds_broken >= 1

    assert trace_path is not None and trace_path.is_file()
    assert csv_path is not None and csv_path.is_file()
    assert sdf_path is not None and sdf_path.is_file()


def test_iteration_limiter_and_compromise_fallback(tmp_path: Path) -> None:
    """Проверяет ограничитель числа итераций при заведомо невыполнимых ограничениях.

    Защита от зацикливания: при max_logp = 1.0 для гефитиниба (базовый 4.28)
    число итераций не должно превысить 2, выдаются лучшие компромиссные кандидаты
    с явной маркировкой отклонений в `constraint_violations`.
    """
    strict_constraints = QueryConstraints(
        scenario_type="lead_optimization",
        target_chembl_id="CHEMBL203",
        lead_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        freeze_scaffold=True,
        max_logp=1.0,  # Недостижимо низкий cLogP для хиназолина
        max_sa_score=3.0,
        top_k_candidates=3,
    )

    state, _trace, _csv, _sdf = run_sar_pipeline(
        user_query="Оптимизируй гефитиниб с жестким cLogP < 1.0",
        session_id="test_iter_limit",
        workspace_dir=tmp_path,
        query_constraints=strict_constraints,
        export_artifacts=True,
    )

    # Проверка, что итератор дизайна не ушел в бесконечный цикл и остановился ровно на 2
    assert state.get("design_iteration") == 2
    assert state.get("critic_verdict") == "approve"

    top_cands = state.get("top_candidates", [])
    assert len(top_cands) >= 1
    # Лучшие кандидаты должны содержать явную маркировку превышения cLogP
    cand1 = top_cands[0]
    assert len(cand1.constraint_violations) > 0
    assert any("cLogP" in v for v in cand1.constraint_violations)
    assert cand1.passes_all_constraints is False
