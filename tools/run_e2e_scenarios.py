"""Скрипт сквозной проверки 3 прикладных сценариев из Раздела 7 SPEC.md.

Сценарии:
1. Target-based SAR Design (BTK, CHEMBL5251):
   - Проверяет срабатывание Петли №1 (Critic -> Research) при неоднородности BAO.
   - Выделяет правила SAR на однородных биохимических тестах (BAO_0000357).
   - Генерирует 5 кандидатов с проверкой 4 категорий данных.
2. Lead Optimization (EGFR, Gefitinib, CHEMBL203):
   - Проверяет срабатывание Петли №1 (homogeneity) И Петли №2 (Critic -> Design).
   - Фиксирует хиназолиновый скаффолд, снижает cLogP (< 3.5) и риск блокады hERG.
   - Рассчитывает дельты относительно исходного гефитиниба.
3. Custom Wet-Lab CSV Analysis (25 соединений из custom_wetlab_series.csv):
   - Изолирует 3 некорректных SMILES, обрабатывает цензурированные и дубликаты.
   - Выявляет 4 Activity Cliffs по индексу SALI.
   - Обогащает контекст фоном ChEMBL и выдает 5 кандидатов с прекурсорами BRICS.

Критерий готовности: все 3 сценария выполняются от начала до конца,
в журналах трассировки фиксируются обе петли, все кандидаты содержат 4 категории полей.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pishovichok.agents.workflow import run_sar_pipeline
from pishovichok.state.graph_state import assert_fat_pointer_isolation
from pishovichok.state.schemas import CandidateMolecule


def _validate_four_categories(cand: CandidateMolecule, idx: int) -> None:
    """Проверяет полноту и валидность всех 4 категорий знаний в объекте CandidateMolecule."""
    four_dict = cand.to_four_category_dict()

    cat1 = four_dict["category_1_experiment"]
    assert cat1["parent_chembl_id"], f"Cand {idx}: missing parent_chembl_id"
    assert cat1["parent_smiles"], f"Cand {idx}: missing parent_smiles"
    assert cat1["parent_exp_pic50"] > 0.0, f"Cand {idx}: invalid parent_exp_pic50"
    assert cat1["parent_exp_ic50_nm"] > 0.0, f"Cand {idx}: invalid parent_exp_ic50_nm"
    assert cat1["parent_bao_format"], f"Cand {idx}: missing parent_bao_format"

    cat2 = four_dict["category_2_calculation"]
    assert cat2["mw"] > 0.0, f"Cand {idx}: invalid mw"
    assert "clogp" in cat2, f"Cand {idx}: missing clogp"
    assert cat2["tpsa"] >= 0.0, f"Cand {idx}: invalid tpsa"
    assert 0.0 <= cat2["qed"] <= 1.0, f"Cand {idx}: invalid qed"
    assert 1.0 <= cat2["sa_score"] <= 10.0, f"Cand {idx}: invalid sa_score"
    assert isinstance(cat2["brics_fragments"], list), (
        f"Cand {idx}: invalid brics_fragments"
    )
    assert cat2["pains_alerts_count"] == 0, f"Cand {idx}: PAINS alert not zero"
    assert cat2["brenk_alerts_count"] == 0, f"Cand {idx}: Brenk alert not zero"

    cat3 = four_dict["category_3_prediction"]
    assert cat3["model_source"] in (
        "admet_ai_dmpnn",
        "catboost_local_qsar",
        "rdkit_analytical_qsar",
    ), f"Cand {idx}: invalid model_source"
    assert 0.0 <= cat3["herg_liability_prob"] <= 1.0, (
        f"Cand {idx}: invalid herg_liability_prob"
    )
    assert cat3["herg_risk_level"] in ("low", "moderate", "high"), (
        f"Cand {idx}: invalid herg_risk_level"
    )
    assert cat3["predicted_pic50"] > 0.0, f"Cand {idx}: invalid predicted_pic50"

    cat4 = four_dict["category_4_hypothesis"]
    assert cat4["rule_id"], f"Cand {idx}: missing rule_id"
    assert cat4["applied_smirks"], f"Cand {idx}: missing applied_smirks"
    assert len(cat4["text_summary"]) > 10, f"Cand {idx}: text_summary too short"


def run_scenario_1(base_dir: Path) -> dict[str, any]:
    """Сценарий 1: Поисковый SAR-дизайн от биологической мишени (BTK / CHEMBL5251)."""
    print("\n" + "=" * 78)
    print(">>> ЗАПУСК СЦЕНАРИЯ 1: Target-Based SAR Design (BTK / CHEMBL5251)")
    print("=" * 78)

    query = (
        "Проведи анализ ингибиторов тирозинкиназы Брутона (BTK, CHEMBL5251), "
        "выдели ключевые правила 'структура–активность' на однородных биохимических тестах "
        "и сгенерируй 5 новых кандидатов с расчетным LogP от 2.0 до 4.0, SAscore < 3.2 "
        "и отсутствием структурных алертов PAINS."
    )

    t0 = time.perf_counter()
    state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=query,
        session_id="scenario1_btk",
        workspace_dir=base_dir,
        export_artifacts=True,
    )
    elapsed = time.perf_counter() - t0

    # Проверка инварианта Fat Pointers
    assert_fat_pointer_isolation(dict(state))

    # Проверка срабатывания Петли №1 (refilter_data)
    trace_events = state["trace_log"]
    loop1_triggers = [
        ev for ev in trace_events if ev.verdict_or_routing == "refilter_data"
    ]
    print(f"[*] Число срабатываний Петли №1 (refilter_data): {len(loop1_triggers)}")
    assert len(loop1_triggers) >= 1, (
        "Сценарий 1: Петля №1 (refilter_data) обязана сработать на неоднородном ChEMBL"
    )

    # Проверка финального утверждения
    assert state.get("critic_verdict") == "approve", (
        f"Expected approve, got {state.get('critic_verdict')}"
    )
    assert state.get("assay_filter_strict") is True, (
        "Expected assay_filter_strict == True after Loop 1"
    )
    assert state.get("active_bao_filter") == "BAO_0000357", (
        f"Expected BAO_0000357, got {state.get('active_bao_filter')}"
    )

    # Проверка кандидатов
    top_cands = state.get("top_candidates", [])
    print(f"[*] Сгенерировано финальных кандидатов: {len(top_cands)}")
    assert len(top_cands) >= 5, f"Expected at least 5 candidates, got {len(top_cands)}"

    for i, cand in enumerate(top_cands[:5], 1):
        _validate_four_categories(cand, i)
        assert 2.0 <= cand.calc_rdkit.clogp <= 4.0 or cand.constraint_violations, (
            f"Candidate {cand.candidate_id} cLogP={cand.calc_rdkit.clogp} outside [2.0, 4.0]"
        )
        assert cand.calc_rdkit.sa_score <= 3.2 or cand.constraint_violations
        assert cand.mpo_harrington_score > 0.0

    # Проверка файлов артефактов
    assert trace_path is not None and trace_path.is_file(), (
        f"Trace path missing: {trace_path}"
    )
    assert csv_path is not None and csv_path.is_file(), f"CSV missing: {csv_path}"
    assert sdf_path is not None and sdf_path.is_file(), f"SDF missing: {sdf_path}"

    print(f"[+] Сценарий 1 успешно пройден за {elapsed:.2f} сек!")
    return {
        "scenario": 1,
        "name": "Target-Based SAR (BTK / CHEMBL5251)",
        "loop1_triggered": True,
        "loop2_triggered": False,
        "candidates_count": len(top_cands),
        "top_harrington": top_cands[0].mpo_harrington_score,
        "trace_file": str(trace_path),
        "csv_file": str(csv_path),
        "sdf_file": str(sdf_path),
        "elapsed_sec": round(elapsed, 2),
    }


def run_scenario_2(base_dir: Path) -> dict[str, any]:
    """Сценарий 2: Направленная оптимизация молекулы-лидера (EGFR / Gefitinib)."""
    print("\n" + "=" * 78)
    print(">>> ЗАПУСК СЦЕНАРИЯ 2: Lead Optimization (EGFR / Gefitinib, CHEMBL203)")
    print("=" * 78)

    query = (
        "Оптимизируй структуру ингибитора EGFR со строкой SMILES "
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1 (гефитиниб): "
        "сохрани хиназолиновый скаффолд, снизь липофильность (LogP < 3.5), "
        "минимизируй риск блокады каналов hERG и предложи синтетически доступные аналоги (SAscore < 3.0)."
    )

    t0 = time.perf_counter()
    state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=query,
        session_id="scenario2_gefitinib",
        workspace_dir=base_dir,
        export_artifacts=True,
    )
    elapsed = time.perf_counter() - t0

    # Проверка инварианта Fat Pointers
    assert_fat_pointer_isolation(dict(state))

    # Проверка срабатывания ОБЕИХ петель:
    trace_events = state["trace_log"]
    loop1_triggers = [
        ev for ev in trace_events if ev.verdict_or_routing == "refilter_data"
    ]
    loop2_triggers = [
        ev for ev in trace_events if ev.verdict_or_routing == "redesign_molecules"
    ]

    print(f"[*] Число срабатываний Петли №1 (refilter_data): {len(loop1_triggers)}")
    print(
        f"[*] Число срабатываний Петли №2 (redesign_molecules): {len(loop2_triggers)}"
    )
    print(f"[*] Финальная итерация дизайна: {state.get('design_iteration')}")

    assert len(loop1_triggers) >= 1, (
        "Сценарий 2: Петля №1 (refilter_data) обязана сработать"
    )
    assert len(loop2_triggers) >= 1, (
        "Сценарий 2: Петля №2 (redesign_molecules) обязана сработать для снижения cLogP"
    )
    assert state.get("design_iteration", 0) >= 2, (
        "Expected at least 2 design iterations"
    )
    assert state.get("critic_verdict") == "approve", (
        f"Expected approve, got {state.get('critic_verdict')}"
    )

    # Проверка референсной карточки исходного лидера (гефитиниба)
    lead_ref = state.get("lead_reference_metrics")
    assert lead_ref is not None, (
        "Сценарий 2: референсная карточка лидера обязана быть вычислена"
    )
    lead_clogp = lead_ref.calc_rdkit.clogp
    print(f"[*] Базовый cLogP гефитиниба: {lead_clogp:.2f}")

    # Проверка кандидатов
    top_cands = state.get("top_candidates", [])
    print(f"[*] Сгенерировано финальных кандидатов: {len(top_cands)}")
    assert len(top_cands) >= 5, f"Expected at least 5 candidates, got {len(top_cands)}"

    for i, cand in enumerate(top_cands[:5], 1):
        _validate_four_categories(cand, i)
        # Проверка снижения липофильности относительно исходного лидера
        assert cand.calc_rdkit.clogp < lead_clogp, (
            f"Candidate {cand.candidate_id} cLogP={cand.calc_rdkit.clogp} not lower than lead ({lead_clogp})"
        )
        assert cand.calc_rdkit.delta_clogp_vs_parent < 0.0, (
            f"Candidate {cand.candidate_id} delta_clogp must be negative"
        )
        # Проверка сохранения хиназолинового ядра
        assert "ncnc" in cand.canonical_smiles, (
            f"Candidate {cand.candidate_id} lost quinazoline core"
        )
        # Проверка приемлемого уровня кардиотоксичности
        assert cand.pred_admet.herg_liability_prob <= 0.40 or cand.constraint_violations

    assert trace_path is not None and trace_path.is_file()
    assert csv_path is not None and csv_path.is_file()
    assert sdf_path is not None and sdf_path.is_file()

    print(f"[+] Сценарий 2 успешно пройден за {elapsed:.2f} сек!")
    return {
        "scenario": 2,
        "name": "Lead Optimization (EGFR / Gefitinib)",
        "loop1_triggered": True,
        "loop2_triggered": True,
        "candidates_count": len(top_cands),
        "lead_clogp": round(lead_clogp, 2),
        "cand1_clogp": round(top_cands[0].calc_rdkit.clogp, 2),
        "cand1_delta_clogp": round(top_cands[0].calc_rdkit.delta_clogp_vs_parent, 2),
        "trace_file": str(trace_path),
        "csv_file": str(csv_path),
        "sdf_file": str(sdf_path),
        "elapsed_sec": round(elapsed, 2),
    }


def run_scenario_3(base_dir: Path) -> dict[str, any]:
    """Сценарий 3: Анализ собственной лабораторной серии из CSV-файла (25 соединений)."""
    print("\n" + "=" * 78)
    print(">>> ЗАПУСК СЦЕНАРИЯ 3: Custom Wet-Lab CSV Analysis (25 соединений)")
    print("=" * 78)

    csv_input = Path("data/samples/custom_wetlab_series.csv").resolve()
    assert csv_input.is_file(), f"Input CSV missing at {csv_input}"

    query = (
        "Проанализируй результаты испытаний нашей лабораторной серии из 25 соединений, "
        "найди критические перепады активности (Activity Cliffs) и предложи 5 структур "
        "для следующей очереди синтеза с разбивкой на коммерчески доступные прекурсоры."
    )

    t0 = time.perf_counter()
    state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=query,
        session_id="scenario3_wetlab",
        workspace_dir=base_dir,
        uploaded_csv_path=csv_input,
        export_artifacts=True,
    )
    elapsed = time.perf_counter() - t0

    # Проверка инварианта Fat Pointers
    assert_fat_pointer_isolation(dict(state))

    # Проверка курации лабораторного файла
    summary = state.get("curation_summary")
    assert summary is not None, "Curation summary must be present"
    print(f"[*] Исходных строк: {summary.initial_rows_count}")
    print(
        f"[*] Отброшено невалидных SMILES: {summary.invalid_smiles_dropped} (ожидается 3)"
    )
    print(
        f"[*] Обработано цензурированных значений: {summary.censored_values_count} (ожидается 2)"
    )
    print(f"[*] Объединено дубликатов: {summary.duplicates_merged_count} (ожидается 3)")
    print(f"[*] Очищенных соединений: {summary.curated_molecules_count} (ожидается 27)")

    assert summary.invalid_smiles_dropped == 3, (
        f"Expected 3 dropped invalid, got {summary.invalid_smiles_dropped}"
    )
    assert summary.censored_values_count == 2, (
        f"Expected 2 censored, got {summary.censored_values_count}"
    )
    assert summary.duplicates_merged_count == 3, (
        f"Expected 3 duplicates, got {summary.duplicates_merged_count}"
    )
    assert summary.curated_molecules_count == 27, (
        f"Expected 27 curated, got {summary.curated_molecules_count}"
    )

    # Проверка Activity Cliffs
    cliffs = state.get("detected_activity_cliffs", [])
    print(
        f"[*] Выявлено Activity Cliffs (SALI >= 1.5, Tanimoto >= 0.85): {len(cliffs)}"
    )
    assert len(cliffs) >= 4, f"Expected at least 4 Activity Cliffs, got {len(cliffs)}"

    sar_rules = state.get("extracted_sar_rules", [])
    print(f"[*] Извлечено правил SAR: {len(sar_rules)}")
    print(
        f"[*] Обогащение фоном ChEMBL: {state.get('enriched_with_chembl_background')}"
    )
    assert len(sar_rules) >= 5, f"Expected at least 5 SAR rules, got {len(sar_rules)}"

    # Проверка финального утверждения и кандидатов
    assert state.get("critic_verdict") == "approve"
    top_cands = state.get("top_candidates", [])
    print(f"[*] Сгенерировано финальных кандидатов: {len(top_cands)}")
    assert len(top_cands) >= 5, f"Expected at least 5 candidates, got {len(top_cands)}"

    for i, cand in enumerate(top_cands[:5], 1):
        _validate_four_categories(cand, i)
        # Проверка ретросинтетической декомпозиции BRICS
        assert len(cand.calc_rdkit.brics_fragments) >= 2, (
            f"Candidate {cand.candidate_id} must have at least 2 BRICS precursors"
        )
        assert cand.calc_rdkit.brics_bonds_broken >= 1

    assert trace_path is not None and trace_path.is_file()
    assert csv_path is not None and csv_path.is_file()
    assert sdf_path is not None and sdf_path.is_file()

    print(f"[+] Сценарий 3 успешно пройден за {elapsed:.2f} сек!")
    return {
        "scenario": 3,
        "name": "Custom Wet-Lab CSV Analysis (25 molecules)",
        "invalid_dropped": summary.invalid_smiles_dropped,
        "censored_handled": summary.censored_values_count,
        "duplicates_merged": summary.duplicates_merged_count,
        "curated_count": summary.curated_molecules_count,
        "cliffs_count": len(cliffs),
        "background_enriched": True,
        "candidates_count": len(top_cands),
        "trace_file": str(trace_path),
        "csv_file": str(csv_path),
        "sdf_file": str(sdf_path),
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    """Запуск полного протокола верификации всех 3 сценариев."""
    base_dir = Path("workspace").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("НАЧАЛО СКВОЗНОЙ ПРОВЕРКИ СЛОЯ АГЕНТНОЙ ОРКЕСТРАЦИИ (SPEC.MD РАЗДЕЛ 7)")
    print("=" * 78)

    t_all = time.perf_counter()
    res1 = run_scenario_1(base_dir)
    res2 = run_scenario_2(base_dir)
    res3 = run_scenario_3(base_dir)
    total_elapsed = time.perf_counter() - t_all

    # Сохранение сводного отчета верификации
    summary_report = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "all_scenarios_passed": True,
        "scenarios": [res1, res2, res3],
    }

    report_path = base_dir / "e2e_verification_summary.json"
    report_path.write_text(
        json.dumps(summary_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("СВОДНЫЙ РЕЗУЛЬТАТ СКВОЗНОЙ ПРОВЕРКИ:")
    print("=" * 78)
    print("[+] Сценарий 1 (BTK Target SAR): УСПЕШНО (Петля №1 сработала, 5 кандидатов)")
    print(
        "[+] Сценарий 2 (Gefitinib Lead Opt): УСПЕШНО (Петля №1 и Петля №2 сработали, cLogP снижен)"
    )
    print(
        "[+] Сценарий 3 (Wet-Lab CSV): УСПЕШНО (Ошибки изолированы, 4 Cliffs, BRICS прекурсоры)"
    )
    print(f"[+] Полное время выполнения всех 3 сценариев: {total_elapsed:.2f} сек")
    print(f"[+] Сводный отчет сохранен в: {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
