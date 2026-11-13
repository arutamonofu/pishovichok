"""Модульные тесты детерминированного вычислительного ядра (Слой 1):
курация данных (`core/curation.py`) и анализ SAR / Activity Cliffs (`core/sar_analysis.py`).

Проверяют критерий готовности на эталонных датасетах из Шага 1:
1. Предварительная интроспекция API (`tools/verify_api.py`).
2. Химическая стандартизация, удаление солей, нейтрализация, канонизация таутомеров,
   приведение к шкале pChEMBL, агрегация дубликатов и выделение каркасов Bemis-Murcko.
3. Загрузка из локального хранилища с фолбэком на `fallback_targets.parquet` и парсинг
   пользовательского CSV (`custom_wetlab_series.csv`) с изоляцией ошибок в `validation_errors_log`.
4. На сырой выборке (`assay_filter_strict=False`) система детектирует смешение протоколов
   испытаний (`BAO_0000357` и `BAO_0000219`, `assay_homogeneous == False`), а после включения
   строгого фильтра (`assay_filter_strict=True`) формирует на 100% методологически однородные
   правила SAR (`assay_homogeneous == True`) и находит критические перепады активности (Activity Cliffs).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from pishovichok.config import settings
from pishovichok.core import (
    CURATED_BIOISOSTERE_CATALOG,
    build_catboost_feature_vector,
    calculate_harrington_mpo_score,
    calculate_rdkit_metrics,
    calculate_sali,
    canonicalize_substituent_smiles,
    convert_to_pchembl,
    curate_chembl_or_csv,
    evaluate_and_rank_candidates,
    evaluate_lead_reference,
    extract_mmp_and_cliffs,
    extract_mmp_with_background_fallback,
    fragment_molecule_mmpa,
    generate_candidate_structures,
    parse_activity_cell,
    predict_admet_metrics,
    select_parent_seeds_from_curated,
    standardize_molecule_detailed,
    standardize_smiles,
)
from pishovichok.state import (
    QueryConstraints,
    create_session_workspace,
    query_parquet_pointer_df,
    read_candidates_parquet,
)
from tools.verify_api import (
    verify_generation_and_evaluation_apis,
    verify_tabular_and_schema_apis,
)


def test_runtime_api_introspection_passes() -> None:
    """Подтверждает прохождение 4-этапного протокола интроспекции установленных библиотек."""
    report = verify_tabular_and_schema_apis()
    assert report["pydantic_v2"]["model_validate_ok"] is True
    assert report["pydantic_v2"]["extra_forbid_enforced"] is True
    assert report["parquet_and_duckdb"]["duckdb_filtered_rows"] == 1
    assert report["rdkit_core"]["mmpa_fragments_count"] > 0
    for iface in report["rdkit_core"]["interfaces"]:
        assert iface["sample_run_success"] is True, f"Failed interface check: {iface}"


def test_molecule_standardization_and_error_isolation() -> None:
    """Проверяет удаление солей, нейтрализацию зарядов, канонизацию таутомеров, скаффолды и лог ошибок."""
    errors: list[str] = []

    # 1. Солевая форма гефитиниба (.Cl) и свободное основание дают идентичный канонический SMILES и InChIKey
    free_base = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    hcl_salt = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1.Cl"
    mesylate_salt = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1.CS(=O)(=O)O"

    res_free = standardize_molecule_detailed(
        free_base, errors_log=errors, row_label="free"
    )
    res_hcl = standardize_molecule_detailed(
        hcl_salt, errors_log=errors, row_label="hcl"
    )
    res_mes = standardize_molecule_detailed(
        mesylate_salt, errors_log=errors, row_label="mes"
    )

    assert res_free is not None
    assert res_hcl is not None
    assert res_mes is not None
    assert res_free.salts_stripped is False
    assert res_hcl.salts_stripped is True
    assert res_mes.salts_stripped is True
    assert (
        res_free.canonical_smiles
        == res_hcl.canonical_smiles
        == res_mes.canonical_smiles
    )
    assert res_free.inchi_key == res_hcl.inchi_key == res_mes.inchi_key
    assert "ncnc" in res_free.scaffold_smiles

    # 2. Нейтрализация заряженного карбоксилата натрия -> нейтральная карбоновая кислота
    sodium_benzoate = "O=C([O-])c1ccccc1.[Na+]"
    res_benz = standardize_smiles(sodium_benzoate, errors_log=errors)
    assert res_benz is not None
    canon_smi, _ik, scaf_smi = res_benz
    assert canon_smi == "O=C(O)c1ccccc1"
    assert scaf_smi == "c1ccccc1"
    assert len(errors) == 0

    # 3. Изоляция невалидных SMILES в лог ошибок без прерывания выполнения
    assert (
        standardize_smiles(
            "INVALID_SMILES_XYZ", errors_log=errors, row_label="bad_token"
        )
        is None
    )
    assert (
        standardize_smiles(
            "C(C)(C)(C)(C)c1ccccc1", errors_log=errors, row_label="pentavalent_c"
        )
        is None
    )
    assert standardize_smiles("   ", errors_log=errors, row_label="empty") is None
    assert len(errors) == 3
    assert "bad_token" in errors[0]
    assert "pentavalent_c" in errors[1]


def test_pchembl_conversion_and_censored_values() -> None:
    """Проверяет приведение концентраций (nM, uM, mM, M) к шкале pChEMBL и обработку цензурированных записей."""
    # 10 nM -> 10^-8 M -> pChEMBL = 8.0
    p_10nm, cens_10nm = convert_to_pchembl(10.0, "nM", "=")
    assert math.isclose(p_10nm, 8.0, abs_tol=1e-4)
    assert cens_10nm is False

    # 1 uM -> 10^-6 M -> pChEMBL = 6.0
    p_1um, cens_1um = convert_to_pchembl(1.0, "uM", "=")
    assert math.isclose(p_1um, 6.0, abs_tol=1e-4)
    assert cens_1um is False

    # Цензурированное измерение '> 100' uM -> is_censored=True, pChEMBL=4.0
    parsed = parse_activity_cell("> 100")
    assert parsed == (">", 100.0)
    p_cens, is_cens = convert_to_pchembl(parsed[1], "uM", parsed[0])
    assert is_cens is True
    assert math.isclose(p_cens, 4.0, abs_tol=1e-4)

    # Цензурированное измерение '>50' uM -> is_censored=True, pChEMBL=4.0
    parsed_50 = parse_activity_cell(">50")
    assert parsed_50 == (">", 50.0)
    p_cens50, is_cens50 = convert_to_pchembl(parsed_50[1], "uM", parsed_50[0])
    assert is_cens50 is True
    assert math.isclose(p_cens50, 4.0, abs_tol=1e-4)

    # Невалидная ячейка активности
    assert parse_activity_cell("not_a_number") is None
    assert parse_activity_cell("-5.0") is None


@pytest.mark.parametrize("target_id", ["CHEMBL203", "CHEMBL5251", "CHEMBL262"])
def test_protocol_mixing_detection_and_strict_filtering_sar_rules(
    target_id: str,
    tmp_path: Path,
) -> None:
    """Главный тест критерия готовности на всех 3 эталонных датасетах мишеней ChEMBL из Шага 1:

    1. На сырой выборке (`assay_filter_strict=False`) система детектирует смешение
       экспериментальных протоколов (`BAO_0000357` и `BAO_0000219`):
       - `curation_summary.is_dataset_assay_homogeneous == False`
       - Среди топ-правил SAR присутствуют межпротокольные правила с `assay_homogeneous == False`
         (включая ложно-высокое правило `[*:1]OC>>[*:1]S(=O)(=O)C` со сдвигом ~ +1.40).
    2. После включения строгого фильтра (`assay_filter_strict=True`):
       - `curation_summary.is_dataset_assay_homogeneous == True`
       - Все 100% извлеченных правил SAR становятся методологически однородными
         (`assay_homogeneous == True`, `participating_bao_formats == ['BAO_0000357']`).
       - Система извлекает достоверные правила трансформации (`[*:1]OC>>[*:1]C(=O)NC`,
         `[*:1]OC>>[*:1]C#N`, `[*:1]OC>>[*:1]Cl` с `support_pairs_count == 6`)
         и детектирует критические перепады активности (`ActivityCliff` с `Tanimoto >= 0.85`
         и `|ΔpIC50| >= 1.5`).
    """
    constraints = QueryConstraints(
        scenario_type="target_based_sar",
        target_chembl_id=target_id,
    )

    # --- ЭТАП 1: Базовый режим (сырая гетерогенная выборка, assay_filter_strict=False) ---
    _, raw_session_dir = create_session_workspace(
        base_workspace_dir=tmp_path,
        session_id=f"{target_id}_raw_mode",
    )
    raw_ptr_1, curated_ptr_1, summary_1 = curate_chembl_or_csv(
        session_dir=raw_session_dir,
        constraints=constraints,
        assay_filter_strict=False,
    )

    assert raw_ptr_1.row_count >= 69
    assert curated_ptr_1.row_count >= 60
    assert summary_1.salts_stripped_count >= 4
    assert summary_1.duplicates_merged_count >= 6
    assert summary_1.censored_values_count == 1
    assert summary_1.dominant_bao_format == "BAO_0000357"
    assert summary_1.dominant_standard_type == "IC50"

    # Подтверждаем детектирование смешения протоколов в выборке
    assert summary_1.is_dataset_assay_homogeneous is False
    assert set(summary_1.bao_distribution.keys()) == {"BAO_0000357", "BAO_0000219"}

    mmp_ptr_1, rules_1, cliffs_1 = extract_mmp_and_cliffs(
        curated_pointer=curated_ptr_1,
        session_dir=raw_session_dir,
        min_support_pairs=2,
    )
    assert mmp_ptr_1.row_count > 100
    assert len(rules_1) >= 15
    assert len(cliffs_1) >= 2

    # В нефильтрованной выборке присутствуют правила со смешанными протоколами BAO (assay_homogeneous == False)
    heterogeneous_rules = [r for r in rules_1 if not r.assay_homogeneous]
    assert len(heterogeneous_rules) > 0, (
        f"Expected heterogeneous SAR rules on raw {target_id}"
    )
    # Топ-1 правило по ΔpIC50 в сырой выборке — межпротокольная замена на метилсульфонил [*:1]S(=O)(=O)C
    top_raw_rule = rules_1[0]
    assert top_raw_rule.assay_homogeneous is False
    assert set(top_raw_rule.participating_bao_formats) == {"BAO_0000219", "BAO_0000357"}
    assert "[*:1]S(=O)(=O)C" in top_raw_rule.smirks
    assert top_raw_rule.mean_delta_pic50 >= 1.30

    # --- ЭТАП 2: Строгий режим фильтрации протоколов (assay_filter_strict=True) ---
    _, strict_session_dir = create_session_workspace(
        base_workspace_dir=tmp_path,
        session_id=f"{target_id}_strict_mode",
    )
    _raw_ptr_2, curated_ptr_2, summary_2 = curate_chembl_or_csv(
        session_dir=strict_session_dir,
        constraints=constraints,
        assay_filter_strict=True,
    )

    # После строгой фильтрации остаются только записи BAO_0000357 (IC50)
    assert summary_2.is_dataset_assay_homogeneous is True
    assert list(summary_2.bao_distribution.keys()) == ["BAO_0000357"]
    assert curated_ptr_2.row_count < curated_ptr_1.row_count
    assert curated_ptr_2.row_count >= 48

    mmp_ptr_2, rules_2, cliffs_2 = extract_mmp_and_cliffs(
        curated_pointer=curated_ptr_2,
        session_dir=strict_session_dir,
        min_support_pairs=2,
    )
    assert mmp_ptr_2.row_count > 50
    assert len(rules_2) >= 10

    # 100% правил после строгой фильтрации методологически однородны!
    assert all(r.assay_homogeneous is True for r in rules_2)
    assert all(r.participating_bao_formats == ["BAO_0000357"] for r in rules_2)

    # Проверяем наличие ключевых однородных правил трансформации с support == 6
    rules_by_smirks = {r.smirks: r for r in rules_2}
    assert "[*:1]OC>>[*:1]Cl" in rules_by_smirks
    assert "[*:1]OC>>[*:1]C#N" in rules_by_smirks
    assert "[*:1]OC>>[*:1]C(=O)NC" in rules_by_smirks

    rule_oc_cl = rules_by_smirks["[*:1]OC>>[*:1]Cl"]
    assert rule_oc_cl.support_pairs_count == 6
    assert rule_oc_cl.assay_homogeneous is True
    assert rule_oc_cl.mean_delta_pic50 > 0.50

    # Проверяем найденные Activity Cliffs
    assert len(cliffs_2) >= 2
    for cliff in cliffs_2:
        assert cliff.tanimoto_similarity >= 0.85
        assert cliff.abs_delta_pic50 >= 1.50
        assert cliff.sali_index > 10.0
        assert cliff.pic50_active > cliff.pic50_inactive
        assert cliff.assay_homogeneous is True
        assert ">>" in cliff.smirks_difference


def test_custom_wetlab_csv_curation_and_activity_cliffs(tmp_path: Path) -> None:
    """Проверяет курацию и SAR-анализ лабораторной серии из `custom_wetlab_series.csv` (Сценарий 3)."""
    _, session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="csv_test"
    )
    csv_path = settings.samples_dir / "custom_wetlab_series.csv"

    constraints = QueryConstraints(scenario_type="custom_csv_analysis")
    raw_ptr, curated_ptr, summary = curate_chembl_or_csv(
        session_dir=session_dir,
        constraints=constraints,
        uploaded_csv_path=csv_path,
        assay_filter_strict=False,
    )

    # В файле 33 строки: 25 уникальных валидных + 3 дубликата + 2 цензурированных + 3 невалидных SMILES
    assert raw_ptr.row_count == 33
    assert summary.source_type == "custom_csv"
    assert summary.initial_rows_count == 33
    assert summary.invalid_smiles_dropped == 3
    assert len(summary.validation_errors_log) == 3
    assert summary.salts_stripped_count >= 1
    assert summary.duplicates_merged_count == 3
    assert summary.censored_values_count == 2
    # 25 валидных активных + 2 цензурированных = 27 уникальных стандартизированных записей
    assert summary.curated_molecules_count == 27
    assert curated_ptr.row_count == 27

    # Проверяем корректность агрегации дубликатов WL-001 (3 измерения: 3.8, 4.2, 3.5 nM)
    df_curated = query_parquet_pointer_df(curated_ptr, where_clause="mol_id = 'WL-001'")
    assert len(df_curated) == 1
    assert int(df_curated.iloc[0]["replicate_count"]) == 3
    assert float(df_curated.iloc[0]["pchembl_std"]) < 0.1

    # Запускаем анализ SAR и поиск Activity Cliffs на курированной лабораторной серии
    mmp_ptr, sar_rules, cliffs = extract_mmp_and_cliffs(
        curated_pointer=curated_ptr,
        session_dir=session_dir,
        min_support_pairs=2,
    )
    assert mmp_ptr.row_count >= 30
    assert len(sar_rules) >= 5
    assert all(r.assay_homogeneous is True for r in sar_rules)

    # Проверяем обнаружение всех 4 пар Activity Cliffs, заложенных в custom_wetlab_series.csv
    assert len(cliffs) >= 4
    cliff_active_ids = {c.mol_id_active for c in cliffs}
    assert {"WL-001", "WL-003", "WL-005", "WL-007"}.issubset(cliff_active_ids)


def test_chembl_target_fallback_and_sparse_csv_background_enrichment(
    tmp_path: Path,
) -> None:
    """Проверяет:
    1. Фолбэк на `fallback_targets.parquet` при запросе отсутствующего локального файла мишени.
    2. Автоматическое обогащение фоновыми правилами ChEMBL для малой серии CSV (< 3 правил).
    """
    # 1. Тестируем фолбэк на fallback_targets.parquet для неизвестного ID мишени
    _, fb_session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="fallback_target"
    )
    fb_constraints = QueryConstraints(
        scenario_type="target_based_sar",
        target_chembl_id="CHEMBL999999_NONEXISTENT",
    )
    raw_ptr, curated_ptr, summary = curate_chembl_or_csv(
        session_dir=fb_session_dir,
        constraints=fb_constraints,
        assay_filter_strict=True,
    )
    assert raw_ptr.row_count >= 60
    assert curated_ptr.row_count >= 45
    assert summary.is_dataset_assay_homogeneous is True

    # 2. Тестируем малочисленный CSV из 2 молекул -> автоматическое обогащение из ChEMBL
    sparse_csv = tmp_path / "sparse_series.csv"
    with sparse_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["mol_id", "SMILES", "IC50_nM", "units"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "mol_id": "S1",
                    "SMILES": "COc1cc2ncnc(Nc3ccc(OC)cc3)c2cc1OC",
                    "IC50_nM": "200.0",
                    "units": "nM",
                },
                {
                    "mol_id": "S2",
                    "SMILES": "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OC",
                    "IC50_nM": "40.0",
                    "units": "nM",
                },
            ]
        )

    _, sparse_session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="sparse_csv"
    )
    _, sparse_curated_ptr, _ = curate_chembl_or_csv(
        session_dir=sparse_session_dir,
        constraints=QueryConstraints(
            scenario_type="custom_csv_analysis", target_chembl_id="CHEMBL203"
        ),
        uploaded_csv_path=sparse_csv,
    )
    _mmp_ptr, enriched_rules, _cliffs, was_enriched = (
        extract_mmp_with_background_fallback(
            curated_pointer=sparse_curated_ptr,
            session_dir=sparse_session_dir,
            background_target_id="CHEMBL203",
        )
    )
    assert was_enriched is True
    assert len(enriched_rules) >= 10
    assert all(r.assay_homogeneous is True for r in enriched_rules)


def test_scenario2_lead_neighborhood_and_fixed_scaffold_mmp(tmp_path: Path) -> None:
    """Проверяет фильтрацию окрестности Танимото >= 0.60 вокруг гефитиниба и фиксацию скаффолда (Сценарий 2)."""
    gefitinib_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    constraints = QueryConstraints(
        scenario_type="lead_optimization",
        target_chembl_id="CHEMBL203",
        lead_smiles=gefitinib_smiles,
        freeze_scaffold=True,
    )
    _, session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="lead_opt"
    )
    _raw_ptr, curated_ptr, summary = curate_chembl_or_csv(
        session_dir=session_dir,
        constraints=constraints,
        assay_filter_strict=True,
    )
    assert summary.is_dataset_assay_homogeneous is True
    df_cur = query_parquet_pointer_df(curated_ptr)
    assert (df_cur["tanimoto_to_lead"] >= 0.60).all()

    _mmp_ptr, rules, _cliffs = extract_mmp_and_cliffs(
        curated_pointer=curated_ptr,
        session_dir=session_dir,
        fixed_scaffold_smiles=gefitinib_smiles,
        min_support_pairs=2,
    )
    assert len(rules) >= 5
    assert all(r.assay_homogeneous is True for r in rules)


def test_sali_and_substituent_canonicalization_helpers() -> None:
    """Проверяет чистые функции расчета SALI, фрагментации rdMMPA и канонизации заместителей."""
    # SALI(8.5, 6.5, sim=0.90, eps=0.001) = 2.0 / (0.10 + 0.001) = 19.80198...
    sali = calculate_sali(8.5, 6.5, tanimoto=0.90, epsilon=0.001)
    assert math.isclose(sali, 2.0 / 0.101, rel_tol=1e-5)

    assert canonicalize_substituent_smiles("CO[*:1]") == "[*:1]OC"
    assert canonicalize_substituent_smiles("Cl[*:1]") == "[*:1]Cl"
    assert canonicalize_substituent_smiles("N#C[*:1]") == "[*:1]C#N"
    assert canonicalize_substituent_smiles("CS(=O)(=O)[*:1]") == "[*:1]S(=O)(=O)C"

    frags = fragment_molecule_mmpa("COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OCCCN1CCOCC1")
    rg_set = {rg for _core, rg, _ring, _ch in frags}
    assert "[*:1]Cl" in rg_set
    assert "[*:1]OC" in rg_set


def test_generation_and_evaluation_api_introspection() -> None:
    """Проверяет программную верификацию API генерации, SAscore, FilterCatalog, BRICS и ADMET."""
    gen_eval_report = verify_generation_and_evaluation_apis()

    rxn_block = gen_eval_report["reaction_and_fragment_engine"]
    assert rxn_block["reaction_type"] == "ChemicalReaction"
    assert rxn_block["run_reactants_return_type"] == "tuple"
    assert rxn_block["sample_product_smiles"] == "N#Cc1ccccc1"
    assert rxn_block["local_crem_db_exists"] is True
    assert rxn_block["local_crem_db_rows"] == 10
    assert "from_smarts" in rxn_block["local_crem_db_columns"]
    assert "to_smarts" in rxn_block["local_crem_db_columns"]
    assert len(CURATED_BIOISOSTERE_CATALOG) >= 45

    eval_block = gen_eval_report["evaluation_filters_and_brics"]
    assert eval_block["sascorer_available"] is True
    assert 2.0 <= eval_block["gefitinib_sa_score"] <= 3.0
    assert eval_block["quinone_alert_detected"] == "quinone_A(370)"
    assert eval_block["brics_return_type"] == "set"
    assert eval_block["brics_bonds_count"] >= 4
    for iface in eval_block["interfaces"]:
        assert iface["sample_run_success"] is True, (
            f"Failed evaluation API check: {iface}"
        )

    admet_block = gen_eval_report["admet_models"]
    assert admet_block["active_cascade_tier"] in (
        "admet_ai_dmpnn",
        "catboost_local_qsar",
        "rdkit_analytical_qsar",
    )


def test_scenario2_gefitinib_generation_evaluation_brics_and_veto(
    tmp_path: Path,
) -> None:
    """Проверяет на примере гефитиниба (Сценарий 2 SPEC.md):
    1. Генерацию 100% химически валидных аналогов с сохранением хиназолинового скаффолда Bemis-Murcko.
    2. Корректное заполнение всех расчетных (`calc_rdkit`), прогностических (`pred_admet`) и
       гипотетических (`hypothesis_rationale`) полей `CandidateMolecule`.
    3. Ретросинтетическую декомпозицию молекул на валидные прекурсоры BRICS без фиктивных меток.
    4. Работу подструктурных фильтров PAINS/Brenk и жесткого вето в индексе Харрингтона (`D = 0.0`).
    """
    from rdkit import Chem

    gefitinib_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    constraints = QueryConstraints(
        scenario_type="lead_optimization",
        target_chembl_id="CHEMBL203",
        lead_smiles=gefitinib_smiles,
        freeze_scaffold=True,
        max_logp=4.5,
        max_sa_score=3.2,
        max_herg_prob=0.60,
        require_no_pains=True,
        top_k_candidates=5,
    )

    _, session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="s2_eval_test"
    )
    _raw_ptr, curated_ptr, _summary = curate_chembl_or_csv(
        session_dir=session_dir,
        constraints=constraints,
        assay_filter_strict=True,
    )
    _mmp_ptr, sar_rules, _cliffs = extract_mmp_and_cliffs(
        curated_pointer=curated_ptr,
        session_dir=session_dir,
        fixed_scaffold_smiles=gefitinib_smiles,
        min_support_pairs=2,
    )

    parent_seeds = select_parent_seeds_from_curated(curated_ptr, constraints)
    assert len(parent_seeds) == 1
    lead_seed = parent_seeds[0]
    assert lead_seed.mol_id == "CHEMBL939"
    assert lead_seed.canonical_smiles == gefitinib_smiles

    # Оценка референсной карточки самого лидера (гефитиниба)
    lead_ref = evaluate_lead_reference(lead_seed, constraints)
    assert lead_ref.parent_chembl_id == "CHEMBL939"
    assert math.isclose(lead_ref.calc_rdkit.clogp, 4.2756, abs_tol=1e-3)
    assert math.isclose(lead_ref.calc_rdkit.tpsa, 68.74, abs_tol=1e-2)
    assert 2.0 <= lead_ref.calc_rdkit.sa_score <= 3.0
    assert lead_ref.calc_rdkit.brics_bonds_broken >= 4
    assert len(lead_ref.calc_rdkit.brics_fragments) >= 3

    # Проверка вектора признаков для CatBoost (ровно 2056 чисел)
    gef_mol = Chem.MolFromSmiles(gefitinib_smiles)
    assert gef_mol is not None
    cb_vec = build_catboost_feature_vector(gef_mol)
    assert cb_vec.shape == (2056,)

    # Направленная генерация аналогов гефитиниба
    raw_candidates = generate_candidate_structures(
        parent_molecules=parent_seeds,
        sar_rules=sar_rules,
        constraints=constraints,
        generation_iteration=1,
        max_candidates_pool=25,
    )
    assert len(raw_candidates) >= 10

    seen_smiles: set[str] = set()
    for rc in raw_candidates:
        mol = Chem.MolFromSmiles(rc.smiles, sanitize=True)
        assert mol is not None, f"Generated invalid SMILES: {rc.smiles}"
        assert rc.smiles != gefitinib_smiles
        assert rc.smiles not in seen_smiles
        seen_smiles.add(rc.smiles)
        # Проверка сохранения скаффолда Bemis-Murcko
        assert rc.scaffold_smiles == lead_seed.scaffold_smiles
        assert len(rc.highlighted_atom_indices) >= 1

    # Пакетная оценка, декомпозиция BRICS, прогноз ADMET и скоринг Харрингтона
    cand_ptr, top_candidates = evaluate_and_rank_candidates(
        raw_candidates=raw_candidates,
        constraints=constraints,
        session_dir=session_dir,
    )
    assert cand_ptr.row_count == len(raw_candidates)
    assert len(top_candidates) == 5

    # Проверка обратимого чтения из ParquetPointer
    restored_all = read_candidates_parquet(cand_ptr)
    assert len(restored_all) == len(raw_candidates)

    for cand in top_candidates:
        # Категория 1: Экспериментальный факт родителя
        assert cand.parent_chembl_id == "CHEMBL939"
        assert cand.parent_smiles == gefitinib_smiles
        assert cand.parent_exp_pic50 > 7.0
        assert cand.parent_bao_format == "BAO_0000357"

        # Категория 2: Детерминированный расчет RDKit + прекурсоры BRICS
        assert cand.calc_rdkit.mw > 350.0
        assert 1.0 <= cand.calc_rdkit.sa_score <= 3.2
        assert 0.0 < cand.calc_rdkit.qed <= 1.0
        assert cand.calc_rdkit.pains_alerts_count == 0
        assert cand.calc_rdkit.brenk_alerts_count == 0
        assert cand.calc_rdkit.brics_bonds_broken >= 4
        assert len(cand.calc_rdkit.brics_fragments) >= 3
        for frag_smi in cand.calc_rdkit.brics_fragments:
            assert "*" not in frag_smi, (
                f"Unstripped BRICS dummy label in precursor: {frag_smi}"
            )
            frag_mol = Chem.MolFromSmiles(frag_smi, sanitize=True)
            assert frag_mol is not None, f"Invalid BRICS precursor SMILES: {frag_smi}"

        # Категория 3: Модельный прогноз ADMET
        assert cand.pred_admet.model_source in (
            "admet_ai_dmpnn",
            "catboost_local_qsar",
            "rdkit_analytical_qsar",
        )
        assert -10.0 <= cand.pred_admet.log_s_aq <= 0.0
        assert 0.0 <= cand.pred_admet.bbb_permeability_prob <= 1.0
        assert 0.0 <= cand.pred_admet.herg_liability_prob <= 1.0
        assert 0.0 <= cand.pred_admet.cyp3a4_inhibition_prob <= 1.0
        assert cand.pred_admet.predicted_pic50 > cand.parent_exp_pic50

        # Категория 4: Научная гипотеза SAR
        assert ">>" in cand.hypothesis_rationale.applied_smirks
        assert len(cand.hypothesis_rationale.text_summary) > 30
        assert 0.40 <= cand.mpo_harrington_score <= 1.0

    # Проверка срабатывания PAINS/Brenk и жесткого права вето Харрингтона (D = 0.0)
    pains_quinone_smiles = "O=C1C=CC(=O)c2ccccc21"
    pains_metrics = calculate_rdkit_metrics(pains_quinone_smiles)
    assert pains_metrics.pains_alerts_count >= 1
    assert "quinone_A(370)" in pains_metrics.pains_matched_names
    pains_admet = predict_admet_metrics(
        pains_quinone_smiles,
        rdkit_metrics=pains_metrics,
        parent_exp_pic50=7.5,
        historical_mean_delta_pic50=0.8,
    )
    veto_score = calculate_harrington_mpo_score(
        calc_rdkit=pains_metrics,
        pred_admet=pains_admet,
        parent_exp_pic50=7.5,
        constraints=constraints,
    )
    assert veto_score == 0.0


def test_scenario2_corrective_constraints_shift_molecular_profile(
    tmp_path: Path,
) -> None:
    """Главный тест критерия готовности для управляемой генерации (Сценарий 2 SPEC.md):

    Проверяет, что при подаче корректирующих ограничений во второй итерации дизайна
    (`max_logp=3.5`, `max_sa_score=3.0`, `max_herg_prob=0.35`,
    `preferred_substituent_types=['polar', 'less_lipophilic', 'exclude_halogens']`, `min_tpsa=85.0`)
    модуль генерации и ранжирования детерминированно смещает профиль кандидатов относительно
    исходного гефитиниба (`cLogP = 4.28`, `TPSA = 68.74`, `hERG prob = 0.58`) и Итерации 1:
    - все 5 лучших кандидатов Итерации 2 имеют `cLogP < 3.5`, `SAscore < 3.0`, `TPSA >= 85.0`;
    - риск блокады каналов `hERG` снижается до уровня `< 0.30` (`low`);
    - вводимые заместители на Итерации 2 не содержат галогенов и имеют `delta_clogp_vs_parent < -0.70`
      и `delta_tpsa_vs_parent > +19.0`.
    """
    gefitinib_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    _, session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="s2_loop2_test"
    )

    # Итерация 1: Первичный дизайн без корректирующих ограничений Критика (фокус на ΔpIC50)
    constraints_iter1 = QueryConstraints(
        scenario_type="lead_optimization",
        target_chembl_id="CHEMBL203",
        lead_smiles=gefitinib_smiles,
        freeze_scaffold=True,
        max_logp=4.5,
        max_sa_score=3.2,
        max_herg_prob=0.60,
        preferred_substituent_types=["any"],
        top_k_candidates=5,
    )
    _raw_ptr, curated_ptr, _summary = curate_chembl_or_csv(
        session_dir=session_dir,
        constraints=constraints_iter1,
        assay_filter_strict=True,
    )
    _mmp_ptr, sar_rules, _cliffs = extract_mmp_and_cliffs(
        curated_pointer=curated_ptr,
        session_dir=session_dir,
        fixed_scaffold_smiles=gefitinib_smiles,
        min_support_pairs=2,
    )
    parent_seeds = select_parent_seeds_from_curated(curated_ptr, constraints_iter1)
    lead_ref = evaluate_lead_reference(parent_seeds[0], constraints_iter1)

    raw_iter1 = generate_candidate_structures(
        parent_molecules=parent_seeds,
        sar_rules=sar_rules,
        constraints=constraints_iter1,
        generation_iteration=1,
        max_candidates_pool=15,
    )
    # На 1-й итерации в пуле присутствуют галогеновые замены из топ-правил MMP (например, [*:1]OC>>[*:1]Cl с cLogP ~ 4.92)
    iter1_introduced = {c.introduced_group_smiles for c in raw_iter1}
    assert "[*:1]Cl" in iter1_introduced or "[*:1]F" in iter1_introduced

    _ptr_iter1, all_eval_iter1 = evaluate_and_rank_candidates(
        raw_candidates=raw_iter1,
        constraints=constraints_iter1,
        session_dir=session_dir,
        filename="evaluated_iter1.parquet",
    )
    mean_clogp_iter1 = sum(c.calc_rdkit.clogp for c in all_eval_iter1) / len(
        all_eval_iter1
    )
    mean_tpsa_iter1 = sum(c.calc_rdkit.tpsa for c in all_eval_iter1) / len(
        all_eval_iter1
    )
    mean_herg_iter1 = sum(
        c.pred_admet.herg_liability_prob for c in all_eval_iter1
    ) / len(all_eval_iter1)

    # Итерация 2: Подача корректирующих ограничений из Сценария 2 SPEC.md
    # (сохранить хиназолиновый скаффолд, снизить LogP < 3.5, минимизировать hERG, SAscore < 3.0,
    #  исключить галогеновые замены и сместить фокус на полярные заместители)
    constraints_iter2 = QueryConstraints(
        scenario_type="lead_optimization",
        target_chembl_id="CHEMBL203",
        lead_smiles=gefitinib_smiles,
        freeze_scaffold=True,
        max_logp=3.5,
        max_sa_score=3.0,
        min_tpsa=85.0,
        max_herg_prob=0.35,
        preferred_substituent_types=["polar", "less_lipophilic", "exclude_halogens"],
        top_k_candidates=5,
    )

    raw_iter2 = generate_candidate_structures(
        parent_molecules=parent_seeds,
        sar_rules=sar_rules,
        constraints=constraints_iter2,
        generation_iteration=2,
        max_candidates_pool=15,
    )
    assert len(raw_iter2) >= 5

    # На 2-й итерации все галогеновые замены ([*:1]Cl, [*:1]F, [*:1]C(F)(F)F) полностью исключены
    for rc in raw_iter2:
        assert "Cl" not in rc.introduced_group_smiles
        assert "F" not in rc.introduced_group_smiles
        assert "Br" not in rc.introduced_group_smiles

    _ptr_iter2, top_iter2 = evaluate_and_rank_candidates(
        raw_candidates=raw_iter2,
        constraints=constraints_iter2,
        session_dir=session_dir,
        filename="evaluated_iter2.parquet",
    )
    assert len(top_iter2) == 5

    mean_clogp_iter2 = sum(c.calc_rdkit.clogp for c in top_iter2) / len(top_iter2)
    mean_tpsa_iter2 = sum(c.calc_rdkit.tpsa for c in top_iter2) / len(top_iter2)
    mean_herg_iter2 = sum(c.pred_admet.herg_liability_prob for c in top_iter2) / len(
        top_iter2
    )

    # Доказываем направленный сдвиг профиля молекул во 2-й итерации относительно Итерации 1 и исходного гефитиниба
    assert mean_clogp_iter2 < mean_clogp_iter1 - 0.35
    assert mean_clogp_iter2 < lead_ref.calc_rdkit.clogp - 1.00
    assert mean_tpsa_iter2 > mean_tpsa_iter1 + 8.0
    assert mean_tpsa_iter2 > lead_ref.calc_rdkit.tpsa + 20.0
    assert mean_herg_iter2 < mean_herg_iter1 - 0.10
    assert mean_herg_iter2 < lead_ref.pred_admet.herg_liability_prob - 0.25

    # Все топ-5 кандидатов Итерации 2 полностью удовлетворяют целевым ограничениям Сценария 2
    for cand in top_iter2:
        assert cand.passes_all_constraints is True, (
            f"Violations in {cand.candidate_id}: {cand.constraint_violations}"
        )
        assert cand.constraint_violations == []
        assert cand.scaffold_smiles == lead_ref.scaffold_smiles
        assert cand.calc_rdkit.clogp < 3.50
        assert cand.calc_rdkit.sa_score < 3.00
        assert cand.calc_rdkit.tpsa >= 85.0
        assert cand.calc_rdkit.delta_clogp_vs_parent < -0.75
        assert cand.calc_rdkit.delta_tpsa_vs_parent > 19.0
        assert cand.pred_admet.herg_liability_prob < 0.35
        assert (
            cand.pred_admet.herg_liability_prob
            < lead_ref.pred_admet.herg_liability_prob
        )
        assert cand.mpo_harrington_score >= 0.70


def test_clean_brics_fragment_smiles_no_isolated_h_warning(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Очистка BRICS-фрагментов не должна вызывать RDKit-предупреждение об изолированном H."""
    from pishovichok.core.evaluation import _clean_brics_fragment_smiles

    assert _clean_brics_fragment_smiles("[1*]c1ccccc1") == "c1ccccc1"
    assert _clean_brics_fragment_smiles("[9*]n1cccc1") == "c1cc[nH]c1"
    assert _clean_brics_fragment_smiles("[5*]N1CCN([5*])CC1") == "C1CNCCN1"
    assert _clean_brics_fragment_smiles("[*]") is None
    assert _clean_brics_fragment_smiles("[1*][H]") is None

    captured = capfd.readouterr()
    assert "not removing hydrogen atom without neighbors" not in captured.err
    assert "not removing hydrogen atom without neighbors" not in captured.out


def test_tool_config_controls_core_modules(tmp_path: Path) -> None:
    """Проверяет реальное влияние параметров ToolConfig на все вычислительные модули core/*.py."""
    from pishovichok.core.rendering import generate_3d_conformer_molblock
    from pishovichok.state import ToolConfig

    # 1. Curation & Standardization: strip_salts=False и scaffold_mode="generic_framework"
    salt_smi = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1.Cl"
    no_strip_cfg = ToolConfig(strip_salts=False, scaffold_mode="generic_framework")
    res_no_strip = standardize_molecule_detailed(salt_smi, tool_config=no_strip_cfg)
    assert res_no_strip is not None
    assert res_no_strip.salts_stripped is False
    assert "." in res_no_strip.canonical_smiles
    assert "N" not in res_no_strip.scaffold_smiles  # generic framework replaces heteroatoms with C

    # 2. Curation CSV: relation_mode="exact_only" отбрасывает цензурированные записи
    _, sess_csv = create_session_workspace(base_workspace_dir=tmp_path, session_id="cfg_csv")
    csv_path = settings.samples_dir / "custom_wetlab_series.csv"
    cfg_exact = ToolConfig(relation_mode="exact_only", duplicate_agg_method="max")
    _, curated_exact_ptr, sum_exact = curate_chembl_or_csv(
        session_dir=sess_csv,
        constraints=QueryConstraints(scenario_type="custom_csv_analysis"),
        uploaded_csv_path=csv_path,
        tool_config=cfg_exact,
    )
    assert sum_exact.curated_molecules_count == 25  # 27 - 2 censored
    assert curated_exact_ptr.row_count == 25

    # 3. Fingerprints & SAR: смена метрики (dice), fp_type (fcfp), sali_epsilon и max_cuts
    cfg_sar = ToolConfig(
        fp_type="fcfp",
        similarity_metric="dice",
        sali_epsilon=0.01,
        mmpa_max_cuts=1,
        min_support_pairs=1,
        cliff_similarity_cutoff=0.80,
    )
    _, rules_dice, cliffs_dice = extract_mmp_and_cliffs(
        curated_pointer=curated_exact_ptr,
        session_dir=sess_csv,
        tool_config=cfg_sar,
    )
    assert len(rules_dice) >= 5
    assert len(cliffs_dice) >= 1

    # 4. Evaluation, custom_ban_smarts, extra_rdkit_descriptors, admet_backend & soft_penalty
    nitro_alert_smi = "O=[N+]([O-])c1ccc(Nc2ncnc3ccccc23)cc1"
    cfg_eval = ToolConfig(
        active_filter_catalogs=["PAINS_A", "Glaxo"],
        custom_ban_smarts=["[N+](=O)[O-]"],
        extra_rdkit_descriptors=["RingCount", "HeavyAtomCount"],
        admet_backend="empirical_only",
        alert_penalty_mode="soft_penalty",
        alert_soft_multiplier=0.5,
        brics_min_fragment_size=4,
    )
    rdkit_m = calculate_rdkit_metrics(nitro_alert_smi, tool_config=cfg_eval)
    assert rdkit_m.extra_alerts_count >= 1
    assert any(n.startswith("CUSTOM_") for n in rdkit_m.extra_alerts_matched_names)
    assert "RingCount" in rdkit_m.extra_descriptors
    assert rdkit_m.extra_descriptors["RingCount"] >= 3.0
    assert "HeavyAtomCount" in rdkit_m.extra_descriptors

    admet_m = predict_admet_metrics(
        nitro_alert_smi,
        rdkit_m,
        parent_exp_pic50=7.0,
        tool_config=cfg_eval,
    )
    assert admet_m.model_source == "rdkit_analytical_qsar"

    mpo_soft = calculate_harrington_mpo_score(
        rdkit_m,
        admet_m,
        parent_exp_pic50=7.0,
        constraints=QueryConstraints(scenario_type="target_based_sar"),
        tool_config=cfg_eval,
    )
    assert mpo_soft > 0.0

    cfg_veto = cfg_eval.model_copy(update={"alert_penalty_mode": "veto"})
    mpo_veto = calculate_harrington_mpo_score(
        rdkit_m,
        admet_m,
        parent_exp_pic50=7.0,
        constraints=QueryConstraints(scenario_type="target_based_sar"),
        tool_config=cfg_veto,
    )
    assert mpo_veto == 0.0

    # 5. Generation engine="crem_only" & protected_smarts
    seeds = select_parent_seeds_from_curated(
        curated_exact_ptr,
        QueryConstraints(scenario_type="custom_csv_analysis"),
        max_parents=2,
        tool_config=cfg_eval,
    )
    cfg_crem = ToolConfig(
        generation_engine="crem_only",
        crem_mode="mutate",
        crem_context_radius=3,
        crem_max_replacements=10,
        max_candidates_pool=10,
    )
    crem_cands = generate_candidate_structures(
        parent_molecules=seeds,
        sar_rules=rules_dice,
        constraints=QueryConstraints(scenario_type="custom_csv_analysis"),
        tool_config=cfg_crem,
    )
    assert len(crem_cands) >= 1
    assert all(
        c.generation_origin == "crem_fragment_replacement" for c in crem_cands
    )

    # 6. 3D Rendering: ETKDGv2 + UFF / none
    mb_uff, highlights_uff = generate_3d_conformer_molblock(
        nitro_alert_smi,
        tool_config=ToolConfig(etkdg_variant="ETKDGv2", force_field="UFF", num_conformers=2),
    )
    assert isinstance(highlights_uff, list) and mb_uff and "M  END" in mb_uff
    mb_none, highlights_none = generate_3d_conformer_molblock(
        nitro_alert_smi,
        tool_config=ToolConfig(etkdg_variant="ETKDGv2", force_field="none", num_conformers=2),
    )
    assert isinstance(highlights_none, list) and mb_none and "M  END" in mb_none


def test_agent_nodes_track_param_sources_and_telemetry(tmp_path: Path) -> None:
    """Проверяет запись node_telemetry и param_sources ('nl_query' и 'critic_loop') в узлах графа."""
    from pishovichok.agents.critic_node import critic_node
    from pishovichok.agents.design_node import design_node
    from pishovichok.agents.evaluation_node import evaluation_node
    from pishovichok.agents.research_node import research_node
    from pishovichok.agents.sar_node import sar_node
    from pishovichok.state import create_initial_graph_state

    _, session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="node_tele_test"
    )
    state = create_initial_graph_state(
        user_query=(
            "Найди ингибиторы CHEMBL5251 (BTK) через FCFP4, метрика dice, "
            "empirical only, ETKDGv2, топ 4"
        ),
        session_id="node_tele_test",
        workspace_dir=session_dir,
    )
    state.update(research_node(state))
    assert state["param_sources"].get("fp_type") == "nl_query"
    assert state["param_sources"].get("similarity_metric") == "nl_query"
    assert state["param_sources"].get("admet_backend") == "nl_query"
    assert state["param_sources"].get("top_k_output") == "nl_query"
    assert "research" in state["node_telemetry"]
    assert state["node_telemetry"]["research"]["active_params"]["fp_type"] == "fcfp"

    state.update(sar_node(state))
    assert "sar" in state["node_telemetry"]
    assert state["node_telemetry"]["sar"]["active_params"]["similarity_metric"] == "dice"

    state.update(critic_node(state))
    assert "critic" in state["node_telemetry"]
    if state["critic_verdict"] == "refilter_data":
        assert state["param_sources"].get("allowed_bao_formats") == "critic_loop"
        state.update(research_node(state))
        state.update(sar_node(state))
        state.update(critic_node(state))

    state.update(design_node(state))
    assert "design" in state["node_telemetry"]

    state.update(evaluation_node(state))
    assert "evaluation" in state["node_telemetry"]
    assert state["node_telemetry"]["evaluation"]["active_params"]["admet_backend"] == "empirical_only"
    assert len(state["top_candidates"]) <= 4


def test_generate_graph_ga_candidates_respects_constraints() -> None:
    """Проверяет GraphGA + CReM grow/mutate на seed=42: валидность RDKit, родитель, rule_id и logP <= max_logp."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from pishovichok.config import settings
    from pishovichok.core import generate_graph_ga_candidates

    seed_molecules = [
        {
            "chembl_id": "CHEMBL913",
            "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "exp_pic50": 8.1,
            "bao_format": "BAO_0000357",
        },
        {
            "chembl_id": "CHEMBL553",
            "smiles": "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
            "exp_pic50": 7.9,
            "bao_format": "BAO_0000357",
        },
    ]
    constraints = {
        "max_logp": 3.5,
        "min_tpsa": 65.0,
        "max_sascore": 4.5,
        "max_mw": 550.0,
    }

    candidates = generate_graph_ga_candidates(
        seed_molecules=seed_molecules,
        constraints=constraints,
        crem_db_path=settings.crem_db_path,
        population_size=12,
        generations=2,
        enable_crem_grow=True,
        protect_scaffold=False,
        seed=42,
    )
    assert len(candidates) > 0

    seed_ids = {s["chembl_id"] for s in seed_molecules}
    rule_ids = set()
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand["smiles"])
        assert mol is not None, f"Невалидный SMILES: {cand['smiles']}"
        assert cand["parent_chembl_id"] in seed_ids
        assert cand["rule_id"] in ("GEN-GA", "CREM-GROW", "CREM-MUT")
        assert "| MCS: " in cand["applied_smirks"]
        logp = float(Descriptors.MolLogP(mol))
        assert logp <= 3.5, f"Превышен max_logp=3.5: {logp:.2f} для {cand['smiles']}"
        rule_ids.add(cand["rule_id"])

    assert rule_ids.intersection({"GEN-GA", "CREM-GROW"})


def test_applicability_domain_tanimoto_and_generation_method_calculation() -> None:
    """Проверяет расчёт tanimoto_to_parent (Morgan) и метки generation_method в CalcRDKitMetrics."""
    parent = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    analog = "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCCCN1CCOCC1"
    unrelated = "CC(=O)Oc1ccccc1C(=O)O"

    same_metrics = calculate_rdkit_metrics(
        parent, parent, generation_origin="mmp_smirks_rule"
    )
    assert same_metrics.tanimoto_to_parent == pytest.approx(1.0)
    assert same_metrics.generation_method == "MMP SMIRKS"

    analog_metrics = calculate_rdkit_metrics(
        analog, parent, generation_origin="crem_fragment_replacement"
    )
    assert analog_metrics.tanimoto_to_parent is not None
    assert 0.5 <= analog_metrics.tanimoto_to_parent < 1.0
    assert analog_metrics.generation_method == "Изостера CReM"

    ref_metrics = calculate_rdkit_metrics(
        analog,
        None,
        reference_mols_or_smiles=[unrelated, parent],
        generation_origin="graph_ga",
    )
    assert ref_metrics.tanimoto_to_parent == pytest.approx(
        analog_metrics.tanimoto_to_parent
    )
    assert ref_metrics.generation_method == "Генеративная модель"


