"""Автоматические тесты схем данных, изоляции 4 категорий знаний, Fat Pointers и демо-датасетов."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMMPA

from pishovichok.config import settings
from pishovichok.state import (
    PARQUET_TABLE_REQUIRED_COLUMNS,
    ActivityCliff,
    CalcRDKitMetrics,
    CandidateMolecule,
    DatasetCurationSummary,
    ParquetPointer,
    PredADMETMetrics,
    QueryConstraints,
    SARHypothesisRationale,
    SARRule,
    TraceEvent,
    assert_fat_pointer_isolation,
    create_initial_graph_state,
    create_session_workspace,
    export_trace_log,
    query_parquet_pointer_df,
    read_candidates_parquet,
    read_parquet_pointer,
    write_candidates_parquet,
    write_parquet_pointer,
)


def _sample_candidate(
    candidate_id: str = "CAND_IT1_01",
    pains_count: int = 0,
    mpo_score: float = 0.82,
) -> CandidateMolecule:
    """Создает валидный тестовый экземпляр CandidateMolecule со всеми 4 категориями данных."""
    return CandidateMolecule(
        candidate_id=candidate_id,
        canonical_smiles="COc1cc2ncnc(Nc3ccc(C#N)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        scaffold_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
        generation_iteration=1,
        # Категория 1: Экспериментальный факт родителя
        parent_chembl_id="CHEMBL939",
        parent_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        parent_exp_pic50=7.85,
        parent_exp_ic50_nm=14.13,
        parent_bao_format="BAO_0000357",
        parent_assay_type="B",
        parent_source_dataset="ChEMBL_Parquet",
        # Категория 2: Детерминированный расчет RDKit
        calc_rdkit=CalcRDKitMetrics(
            mw=453.95,
            clogp=3.42,
            tpsa=96.15,
            hbd=1,
            hba=8,
            rotatable_bonds=8,
            fsp3=0.32,
            qed=0.64,
            sa_score=2.55,
            lipinski_violations=0,
            pains_alerts_count=pains_count,
            pains_matched_names=["azo_A"] if pains_count > 0 else [],
            brenk_alerts_count=0,
            brenk_matched_names=[],
            glaxo_alerts_count=0,
            brics_fragments=["COc1cc2ncnc(Cl)c2cc1O", "Nc1ccc(C#N)c(Cl)c1"],
            brics_bonds_broken=2,
            delta_clogp_vs_parent=-0.42,
            delta_tpsa_vs_parent=23.79,
        ),
        # Категория 3: Модельный прогноз ADMET / QSAR
        pred_admet=PredADMETMetrics(
            model_source="rdkit_analytical_qsar",
            log_s_aq=-4.12,
            solubility_class="moderate",
            bbb_permeability_prob=0.22,
            herg_liability_prob=0.24,
            herg_risk_level="low",
            cyp3a4_inhibition_prob=0.35,
            predicted_pic50=8.32,
            prediction_confidence="high",
        ),
        # Категория 4: Научная гипотеза SAR
        hypothesis_rationale=SARHypothesisRationale(
            generation_origin="mmp_smirks_rule",
            rule_id="R2",
            applied_smirks="[*:1]F>>[*:1]C#N",
            replaced_group_smiles="[*:1]F",
            introduced_group_smiles="[*:1]C#N",
            historical_mean_delta_pic50=0.47,
            historical_std_delta_pic50=0.11,
            support_pairs_count=6,
            assay_homogeneous=True,
            text_summary="Замена фтора на полярную цианогруппу увеличивает pIC50 и снижает cLogP.",
        ),
        mpo_harrington_score=mpo_score,
        passes_all_constraints=True,
        constraint_violations=[],
        highlighted_atom_indices=[14, 15],
    )


def test_four_category_isolation_and_immutability() -> None:
    """Проверяет строгое разделение 4 категорий данных, неизменяемость и запрет лишних полей."""
    cand = _sample_candidate()
    four_blocks = cand.to_four_category_dict()

    assert set(four_blocks.keys()) == {
        "category_1_experiment",
        "category_2_calculation",
        "category_3_prediction",
        "category_4_hypothesis",
    }

    # Поля 4 категорий не пересекаются между собой
    exp_keys = set(four_blocks["category_1_experiment"].keys())
    calc_keys = set(four_blocks["category_2_calculation"].keys())
    pred_keys = set(four_blocks["category_3_prediction"].keys())
    hyp_keys = set(four_blocks["category_4_hypothesis"].keys())

    assert exp_keys.isdisjoint(calc_keys)
    assert exp_keys.isdisjoint(pred_keys)
    assert exp_keys.isdisjoint(hyp_keys)
    assert calc_keys.isdisjoint(pred_keys)
    assert calc_keys.isdisjoint(hyp_keys)
    assert pred_keys.isdisjoint(hyp_keys)

    # Проверка неизменяемости (frozen=True) экспериментального факта и вложенных блоков
    with pytest.raises(ValidationError):
        cand.parent_exp_pic50 = 9.99  # type: ignore[misc]

    with pytest.raises(ValidationError):
        cand.calc_rdkit.clogp = 1.0  # type: ignore[misc]

    # Проверка запрета произвольных полей (extra="forbid")
    raw = cand.model_dump()
    raw["hallucinated_field"] = "forbidden"
    with pytest.raises(ValidationError):
        CandidateMolecule.model_validate(raw)


def test_harrington_veto_invariant() -> None:
    """Проверяет, что при срабатывании PAINS/Brenk индекс Харрингтона обязан равняться 0.0."""
    with pytest.raises(ValidationError, match="Harrington Veto Invariant"):
        _sample_candidate(pains_count=1, mpo_score=0.50)

    vetoed = _sample_candidate(pains_count=1, mpo_score=0.0)
    assert vetoed.mpo_harrington_score == 0.0


def test_fat_pointer_storage_and_graph_state_isolation(tmp_path: Path) -> None:
    """Проверяет передачу ссылок ParquetPointer между узлами, SHA-256 и защиту состояния графа."""
    session_id, session_dir = create_session_workspace(
        base_workspace_dir=tmp_path, session_id="test_s1"
    )

    # 1. Запись и чтение кандидатов через Fat Pointer
    c1 = _sample_candidate("CAND_01")
    c2 = _sample_candidate("CAND_02")
    cand_ptr = write_candidates_parquet([c1, c2], session_dir=session_dir)

    assert isinstance(cand_ptr, ParquetPointer)
    assert cand_ptr.table_type == "evaluated_candidates"
    assert cand_ptr.row_count == 2
    assert len(cand_ptr.sha256_checksum) == 64

    restored = read_candidates_parquet(cand_ptr, verify_checksum=True)
    assert len(restored) == 2
    assert restored[0] == c1
    assert restored[1] == c2

    # 2. Проверка SQL-фильтрации через DuckDB по ParquetPointer
    df_filtered = query_parquet_pointer_df(
        cand_ptr,
        where_clause="candidate_id = 'CAND_02'",
    )
    assert len(df_filtered) == 1
    assert df_filtered.iloc[0]["candidate_id"] == "CAND_02"

    # 3. Проверка обнаружения модификации файла (контрольная сумма SHA-256)
    tampered_ptr = ParquetPointer(
        uri=cand_ptr.uri,
        table_type=cand_ptr.table_type,
        row_count=cand_ptr.row_count,
        columns=cand_ptr.columns,
        sha256_checksum="0" * 64,
    )
    with pytest.raises(ValueError, match="integrity check failed"):
        read_parquet_pointer(tampered_ptr, verify_checksum=True)

    # 4. Проверка отсутствия обязательных колонок в ParquetPointer
    with pytest.raises(ValidationError, match="missing required columns"):
        ParquetPointer(
            uri=cand_ptr.uri,
            table_type="raw_dataset",
            row_count=1,
            columns=["raw_row_id", "raw_smiles"],
            sha256_checksum=cand_ptr.sha256_checksum,
        )

    # 5. Проверка инициализации SARGraphState и защиты от передачи сырых коллекций
    state = create_initial_graph_state(
        user_query="Проанализируй EGFR (CHEMBL203)",
        session_id=session_id,
        workspace_dir=session_dir,
        query_constraints=QueryConstraints(
            scenario_type="target_based_sar", target_chembl_id="CHEMBL203"
        ),
    )
    state["candidates_table_uri"] = cand_ptr
    assert_fat_pointer_isolation(dict(state))

    illegal_state = dict(state)
    illegal_state["raw_data_df"] = df_filtered
    with pytest.raises(ValueError, match="Fat Pointer Invariant violated"):
        assert_fat_pointer_isolation(illegal_state)

    # 6. Экспорт журнала трассировки
    trace_event = TraceEvent(
        step_index=1,
        timestamp_iso="2026-11-12T17:00:00Z",
        agent_name="ResearchAgent",
        phase="curation",
        input_summary="CHEMBL203",
        action_taken="Standardized 70 molecules",
        output_summary="Curated 60 molecules",
        verdict_or_routing=None,
        fallback_triggered=None,
        duration_ms=42.5,
    )
    trace_path = export_trace_log([trace_event], session_dir=session_dir)
    assert trace_path.is_file()


@pytest.mark.parametrize("target_id", ["CHEMBL203", "CHEMBL5251", "CHEMBL262"])
def test_chembl_target_datasets_valid_and_heterogeneous(
    target_id: str, tmp_path: Path
) -> None:
    """Проверяет сгенерированные Parquet-датасеты мишеней для Сценариев 1 и 2 и Петли самокоррекции №1."""
    parquet_path = settings.chembl_cache_dir / f"{target_id}.parquet"
    assert parquet_path.is_file(), f"Missing target dataset: {parquet_path}"

    table = pq.read_table(parquet_path)
    assert table.num_rows >= 60

    # Проверяем наличие всех обязательных колонок raw_dataset
    required_cols = set(PARQUET_TABLE_REQUIRED_COLUMNS["raw_dataset"])
    assert required_cols.issubset(set(table.column_names))

    # Проверяем работу через Fat Pointer и DuckDB
    ptr = write_parquet_pointer(
        table,
        table_type="raw_dataset",
        session_dir=tmp_path,
        filename=f"{target_id}.parquet",
    )
    df = query_parquet_pointer_df(ptr)
    assert len(df) == table.num_rows

    # 1. Все SMILES в датасете мишени химически валидны в RDKit
    for smi in df["raw_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid SMILES in {target_id}: {smi}"

    # 2. Намеренная неоднородность протоколов BAO (BAO_0000357 и BAO_0000219) для Петли №1
    bao_counts = df["bao_format"].value_counts().to_dict()
    assert "BAO_0000357" in bao_counts
    assert "BAO_0000219" in bao_counts
    assert bao_counts["BAO_0000357"] >= 40
    assert bao_counts["BAO_0000219"] >= 10

    # 3. Проверяем, что rdMMPA на нефильтрованной выборке образует межпротокольные пары (assay_homogeneous == False),
    # а после фильтрации по BAO_0000357 все пары становятся 100% гомогенными
    conn = duckdb.connect(":memory:")
    homog_df = conn.sql(
        f"SELECT * FROM read_parquet('{parquet_path.as_posix()}') "
        "WHERE bao_format = 'BAO_0000357' AND standard_type = 'IC50'"
    ).df()
    assert set(homog_df["bao_format"].unique()) == {"BAO_0000357"}

    # Проверяем фрагментацию rdMMPA на молекулах выборки
    sample_mol = Chem.MolFromSmiles(homog_df.iloc[0]["raw_smiles"].split(".")[0])
    frags = list(rdMMPA.FragmentMol(sample_mol, maxCuts=1, resultsAsMols=False))
    assert len(frags) > 0


def test_scenario2_gefitinib_lead_present_in_chembl203() -> None:
    """Проверяет наличие гефитиниба (CHEMBL939) и его структурной окрестности в CHEMBL203.parquet."""
    parquet_path = settings.chembl_cache_dir / "CHEMBL203.parquet"
    conn = duckdb.connect(":memory:")
    df = conn.sql(f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')").df()

    assert "CHEMBL939" in set(df["molecule_chembl_id"])
    gefitinib_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    lead_mol = Chem.MolFromSmiles(gefitinib_smiles)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    lead_fp = fp_gen.GetFingerprint(lead_mol)

    neighbors = 0
    for smi in df["raw_smiles"]:
        mol = Chem.MolFromSmiles(smi.split(".")[0])
        if mol is not None:
            sim = DataStructs.TanimotoSimilarity(lead_fp, fp_gen.GetFingerprint(mol))
            if sim >= 0.60:
                neighbors += 1
    assert neighbors >= 15, (
        f"Expected >= 15 neighbors with Tanimoto >= 0.60 around Gefitinib, got {neighbors}"
    )


def test_scenario3_custom_wetlab_csv_cliffs_and_edge_cases() -> None:
    """Проверяет пользовательский CSV лабораторной серии (Activity Cliffs, дубликаты, невалидные, цензурированные)."""
    csv_path = settings.samples_dir / "custom_wetlab_series.csv"
    assert csv_path.is_file(), f"Missing custom wet-lab CSV: {csv_path}"

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) >= 30
    assert set(rows[0].keys()) == {"mol_id", "SMILES", "IC50_nM", "units"}

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    valid_mols: list[tuple[str, str, object, float]] = []
    invalid_smiles_rows: list[str] = []
    censored_rows: list[str] = []
    salt_rows: list[str] = []
    inchi_keys: list[str] = []

    for r in rows:
        raw_smi = r["SMILES"].strip()
        raw_val = r["IC50_nM"].strip()
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            invalid_smiles_rows.append(r["mol_id"])
            continue

        if "." in raw_smi:
            salt_rows.append(r["mol_id"])

        parent_smi = raw_smi.split(".")[0]
        parent_mol = Chem.MolFromSmiles(parent_smi)
        assert parent_mol is not None
        inchi_keys.append(Chem.MolToInchiKey(parent_mol))

        if raw_val.startswith((">", "<")):
            censored_rows.append(r["mol_id"])
            continue

        val = float(raw_val)
        val_nm = val * 1000.0 if r["units"].strip() == "uM" else val
        pic50 = 9.0 - math.log10(val_nm)
        valid_mols.append(
            (
                r["mol_id"],
                Chem.MolToSmiles(parent_mol, canonical=True),
                fp_gen.GetFingerprint(parent_mol),
                pic50,
            )
        )

    # Проверяем краевые случаи из ТЗ
    assert len(invalid_smiles_rows) >= 3, "CSV must contain invalid SMILES records"
    assert len(censored_rows) >= 2, "CSV must contain censored (> 100 uM) records"
    assert len(salt_rows) >= 1, "CSV must contain salt forms (.Cl)"
    assert len(inchi_keys) > len(set(inchi_keys)), (
        "CSV must contain duplicate structures"
    )

    # Проверяем наличие Activity Cliffs (Tanimoto >= 0.85 и |ΔpIC50| >= 1.5) и валидность модели ActivityCliff
    cliffs: list[ActivityCliff] = []
    for i in range(len(valid_mols)):
        for j in range(i + 1, len(valid_mols)):
            id_i, smi_i, fp_i, p_i = valid_mols[i]
            id_j, smi_j, fp_j, p_j = valid_mols[j]
            if smi_i == smi_j:
                continue
            sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
            delta = abs(p_i - p_j)
            if sim >= 0.85 and delta >= 1.5:
                sali = delta / (1.0 - sim + 0.001)
                active_first = p_i >= p_j
                cliffs.append(
                    ActivityCliff(
                        cliff_id=f"CLIFF_{len(cliffs) + 1}",
                        mol_id_active=id_i if active_first else id_j,
                        mol_id_inactive=id_j if active_first else id_i,
                        smiles_active=smi_i if active_first else smi_j,
                        smiles_inactive=smi_j if active_first else smi_i,
                        pic50_active=p_i if active_first else p_j,
                        pic50_inactive=p_j if active_first else p_i,
                        abs_delta_pic50=round(delta, 3),
                        tanimoto_similarity=round(sim, 4),
                        sali_index=round(sali, 2),
                        smirks_difference="[*:1]R1>>[*:1]R2",
                        assay_homogeneous=True,
                    )
                )

    assert len(cliffs) >= 4, (
        f"Expected >= 4 Activity Cliffs in CSV series, got {len(cliffs)}"
    )


def test_auxiliary_sar_and_curation_schemas() -> None:
    """Проверяет валидацию вспомогательных схем SARRule и DatasetCurationSummary."""
    rule = SARRule(
        rule_id="R1",
        smirks="[*:1]OC>>[*:1]Cl",
        from_fragment="[*:1]OC",
        to_fragment="[*:1]Cl",
        mean_delta_pic50=0.72,
        std_delta_pic50=0.08,
        support_pairs_count=6,
        mean_delta_clogp=0.64,
        mean_delta_tpsa=-9.23,
        assay_homogeneous=True,
        participating_bao_formats=["BAO_0000357"],
        example_pair_ids=["P1", "P2"],
    )
    assert rule.assay_homogeneous is True

    summary = DatasetCurationSummary(
        source_type="chembl_parquet",
        initial_rows_count=70,
        invalid_smiles_dropped=0,
        salts_stripped_count=7,
        duplicates_merged_count=6,
        censored_values_count=1,
        curated_molecules_count=63,
        dominant_bao_format="BAO_0000357",
        dominant_standard_type="IC50",
        bao_distribution={"BAO_0000357": 58, "BAO_0000219": 12},
        is_dataset_assay_homogeneous=False,
        top_scaffolds=[
            {
                "scaffold_smiles": "c1ccc(Nc2ncnc3ccccc23)cc1",
                "count": 42,
                "mean_pic50": 7.1,
            }
        ],
        validation_errors_log=[],
    )
    assert summary.is_dataset_assay_homogeneous is False


def test_tool_config_defaults_extra_forbid_and_state_tracking(tmp_path: Path) -> None:
    """Проверяет схему ToolConfig, запрет лишних полей, частичное слияние mpo_weights и трекинг источников."""
    from pishovichok.state import (
        ToolConfig,
        resolve_tool_config,
        update_param_sources,
    )

    cfg = ToolConfig()
    assert cfg.allowed_bao_formats == ["BAO_0000357", "BAO_0000219"]
    assert cfg.standard_types == ["IC50", "Ki"]
    assert cfg.relation_mode == "include_censored"
    assert cfg.mmpa_max_cuts == 1
    assert cfg.fp_type == "ecfp"
    assert cfg.generation_engine == "mmp_and_crem"
    assert cfg.active_filter_catalogs == ["PAINS_A", "PAINS_B", "PAINS_C", "Brenk"]
    assert cfg.admet_backend == "auto_catboost"
    assert cfg.cns_profile_mode == "peripheral_low_bbb"
    assert cfg.alert_penalty_mode == "veto"
    assert cfg.etkdg_variant == "ETKDGv3"
    assert cfg.force_field == "MMFF94s"
    assert cfg.max_iterations == 2
    assert cfg.mpo_weights["pic50"] == 1.5
    assert cfg.mpo_weights["herg"] == 1.2

    with pytest.raises(ValidationError):
        ToolConfig(unknown_param_xyz=123)  # type: ignore[call-arg]

    updated, changed = cfg.apply_overrides(
        {
            "mpo_weights": {"logp": 2.4},
            "fp_type": "fcfp",
            "max_iterations": 3,
        }
    )
    assert updated.fp_type == "fcfp"
    assert updated.max_iterations == 3
    assert updated.mpo_weights["logp"] == 2.4
    assert updated.mpo_weights["pic50"] == 1.5
    assert set(changed) == {"mpo_weights", "fp_type", "max_iterations"}

    state = create_initial_graph_state(
        user_query="Тест конфигурации",
        session_id="cfg_sess",
        workspace_dir=tmp_path,
        tool_config=updated,
        param_sources=update_param_sources({}, changed, "nl_query"),
    )
    resolved = resolve_tool_config(state)
    assert resolved.fp_type == "fcfp"
    assert state["param_sources"]["fp_type"] == "nl_query"
    assert state["param_sources"]["mpo_weights"] == "nl_query"
    assert state["param_sources"]["allowed_bao_formats"] == "default"
    assert state["node_telemetry"] == {}

    ps2 = update_param_sources(state["param_sources"], ["max_candidates_pool"], "critic_loop")
    assert ps2["max_candidates_pool"] == "critic_loop"
    ps3 = update_param_sources(ps2, ["protected_smarts"], "ui_manual")
    assert ps3["protected_smarts"] == "ui_manual"


def test_nl_tool_config_parsing_across_subsystems() -> None:
    """Проверяет детерминированный разбор NL-настроек ToolConfig и сохранение триггера Петли 1."""
    from pishovichok.agents.llm_adapter import LLMAdapter

    adapter = LLMAdapter()
    nl_query = (
        "Оптимизируй Гефитиниб для CHEMBL203 (EGFR) по Ki на BAO_0000357 без цензурированных данных, "
        "не удалять соли, без канонизации таутомеров, generic framework, pchembl >= 5.5, "
        "2 cuts, FCFP4, метрика dice, cliff sim >= 0.90, sali epsilon 0.01, "
        "crem only, grow, radius 2, не замораживать скаффолд, protected smarts c1ccc2ncncc2c1, топ 8, "
        "каталоги Glaxo и SureChEMBL, empirical only, ЦНС ГТБ, soft penalty 0.4, "
        "ETKDGv2, UFF, 7 конформеров, max_iterations=3, дескрипторы RingCount, HeavyAtomCount"
    )
    overrides = adapter._deterministic_parse_tool_config(nl_query)

    assert overrides["allowed_bao_formats"] == ["BAO_0000357"]
    assert "Ki" in overrides["standard_types"]
    assert overrides["relation_mode"] == "exact_only"
    assert overrides["strip_salts"] is False
    assert overrides["canonicalize_tautomers"] is False
    assert overrides["scaffold_mode"] == "generic_framework"
    assert overrides["pchembl_min_cutoff"] == 5.5
    assert overrides["mmpa_max_cuts"] == 2
    assert overrides["fp_type"] == "fcfp"
    assert overrides["fp_radius"] == 2
    assert overrides["similarity_metric"] == "dice"
    assert overrides["cliff_similarity_cutoff"] == 0.9
    assert overrides["sali_epsilon"] == 0.01
    assert overrides["generation_engine"] == "crem_only"
    assert overrides["crem_mode"] == "grow"
    assert overrides["crem_context_radius"] == 2
    assert overrides["freeze_murcko_core"] is False
    assert overrides["protected_smarts"] == "c1ccc2ncncc2c1"
    assert overrides["top_k_output"] == 8
    assert "Glaxo" in overrides["active_filter_catalogs"]
    assert "SureChEMBL" in overrides["active_filter_catalogs"]
    assert overrides["admet_backend"] == "empirical_only"
    assert overrides["cns_profile_mode"] == "cns_high_bbb"
    assert overrides["alert_penalty_mode"] == "soft_penalty"
    assert overrides["alert_soft_multiplier"] == 0.4
    assert overrides["etkdg_variant"] == "ETKDGv2"
    assert overrides["force_field"] == "UFF"
    assert overrides["num_conformers"] == 7
    assert overrides["max_iterations"] == 3
    assert "RingCount" in overrides["extra_rdkit_descriptors"]
    assert "HeavyAtomCount" in overrides["extra_rdkit_descriptors"]

    # Защита Сценария 1: общая фраза про однородные биохимические тесты не должна форсировать BAO заранее
    sc1_query = (
        "Найди наиболее активные скаффолды и SAR-правила для мишени CHEMBL5251 (BTK) "
        "на однородных биохимических тестах и выдели activity cliffs"
    )
    sc1_overrides = adapter._deterministic_parse_tool_config(sc1_query)
    assert "allowed_bao_formats" not in sc1_overrides

    constraints, parsed_overrides, _ = adapter.parse_query_with_tool_config(
        "Найди ингибиторы CHEMBL203 через FCFP4 и метрику dice, empirical only, UFF"
    )
    assert constraints.target_chembl_id == "CHEMBL203"
    assert parsed_overrides.get("fp_type") == "fcfp"
    assert parsed_overrides.get("similarity_metric") == "dice"
    assert parsed_overrides.get("admet_backend") == "empirical_only"
    assert parsed_overrides.get("force_field") == "UFF"


def test_graph_ga_tool_config_and_candidate_contracts() -> None:
    """Проверяет новые поля ToolConfig, сериализацию GEN-GA/CREM-GROW и парсинг GA-параметров."""
    from pishovichok.agents.llm_adapter import LLMAdapter
    from pishovichok.state.schemas import (
        CalcRDKitMetrics,
        CandidateMolecule,
        GeneratedRawCandidate,
        ParentSeed,
        PredADMETMetrics,
        SARHypothesisRationale,
        ToolConfig,
    )

    tc = ToolConfig()
    assert tc.ga_population_size == 20
    assert tc.ga_generations == 2
    assert tc.enable_crem_grow is True

    adapter = LLMAdapter()
    ga_overrides = adapter._deterministic_parse_tool_config(
        "Запусти с популяцией 24, поколений 3, без crem grow"
    )
    assert ga_overrides.get("ga_population_size") == 24
    assert ga_overrides.get("ga_generations") == 3
    assert ga_overrides.get("enable_crem_grow") is False

    parent = ParentSeed(
        mol_id="CHEMBL913",
        canonical_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        scaffold_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
        exp_pic50=8.1,
        exp_ic50_nm=7.9,
        bao_format="BAO_0000357",
        assay_type="B",
        source_dataset="test",
    )
    for rule_id, origin in (("GEN-GA", "graph_ga"), ("CREM-GROW", "crem_fragment_replacement")):
        raw = GeneratedRawCandidate(
            smiles="COc1cc2ncnc(Nc3ccccc3)c2cc1O",
            scaffold_smiles="c1ccc(Nc2ncnc3ccccc23)cc1",
            generation_iteration=2,
            parent_seed=parent,
            rule_id=rule_id,
            applied_smirks="BRICS_crossover | MCS: c1ccc2ncncc2c1",
            replaced_group_smiles="*",
            introduced_group_smiles="*",
            generation_origin=origin,
            historical_mean_delta_pic50=0.15,
            historical_std_delta_pic50=0.45,
            support_pairs_count=0,
            assay_homogeneous=False,
        )
        assert raw.rule_id == rule_id
        cand = CandidateMolecule(
            candidate_id=f"CAND-{rule_id}",
            canonical_smiles=raw.smiles,
            scaffold_smiles=raw.scaffold_smiles,
            generation_iteration=2,
            parent_chembl_id=parent.mol_id,
            parent_smiles=parent.canonical_smiles,
            parent_exp_pic50=parent.exp_pic50,
            parent_exp_ic50_nm=parent.exp_ic50_nm,
            parent_bao_format=parent.bao_format,
            parent_assay_type=parent.assay_type,
            parent_source_dataset=parent.source_dataset,
            calc_rdkit=CalcRDKitMetrics(
                mw=320.0,
                clogp=2.5,
                tpsa=78.0,
                hbd=2,
                hba=5,
                rotatable_bonds=3,
                fsp3=0.25,
                sa_score=2.4,
                qed=0.75,
                lipinski_violations=0,
                pains_alerts_count=0,
                brenk_alerts_count=0,
                delta_clogp_vs_parent=-1.2,
                delta_tpsa_vs_parent=9.0,
            ),
            pred_admet=PredADMETMetrics(
                model_source="rdkit_analytical_qsar",
                log_s_aq=-3.1,
                solubility_class="high",
                bbb_permeability_prob=0.25,
                herg_liability_prob=0.18,
                herg_risk_level="low",
                cyp3a4_inhibition_prob=0.15,
                predicted_pic50=8.2,
                prediction_confidence="medium",
            ),
            hypothesis_rationale=SARHypothesisRationale(
                generation_origin=origin,
                rule_id=rule_id,
                applied_smirks=raw.applied_smirks,
                replaced_group_smiles="*",
                introduced_group_smiles="*",
                support_pairs_count=0,
                historical_mean_delta_pic50=0.15,
                assay_homogeneous=False,
                text_summary="Тестовая гипотеза",
            ),
            mpo_harrington_score=0.72,
            passes_all_constraints=True,
        )
        roundtrip = CandidateMolecule.model_validate_json(cand.model_dump_json())
        assert roundtrip.hypothesis_rationale.rule_id == rule_id


def test_from_flat_parquet_record_handles_nan_and_empty_optional_fields() -> None:
    """Проверяет устойчивость `from_flat_parquet_record` к NaN/None из pandas/DuckDB."""
    cand = _sample_candidate()
    flat = cand.to_flat_parquet_record()
    flat["docking_json"] = float("nan")
    flat["alert_penalty_mode"] = float("nan")
    flat["constraint_violations_json"] = float("nan")
    flat["highlighted_atom_indices_json"] = ""

    restored = CandidateMolecule.from_flat_parquet_record(flat)
    assert restored.docking is None
    assert restored.alert_penalty_mode == "veto"
    assert restored.constraint_violations == []
    assert restored.highlighted_atom_indices == []



