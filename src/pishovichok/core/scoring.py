"""Модуль многопараметрического скоринга Харрингтона с правом вето и ранжирования кандидатов (Слой 1).

Реализует Раздел 5 SPEC.md и Раздел 6.1.4 ARCHITECTURE.md без вызовов LLM:
1. Частные функции желательности Харрингтона $d_k = \\exp(-\\exp(-Y_k)) \\in (0, 1]$ для 5 критериев:
   - $d_1(\\widehat{\\text{pIC}}_{50})$ — максимизация прогнозной активности;
   - $d_2(\\text{cLogP})$ — удержание липофильности в целевом коридоре $[\\text{min\\_logp}, \\text{max\\_logp}]$;
   - $d_3(\\text{SAscore})$ — минимизация синтетической сложности ($\\le \\text{max\\_sa\\_score}$);
   - $d_4(\\text{QED})$ — максимизация индекса лекарственного сходства;
   - $d_5(P_{\\text{hERG}})$ — минимизация риска блокады каналов hERG ($\\le \\text{max\\_herg\\_prob}$).
2. Свертка в обобщенный индекс Харрингтона $D = I_{\\text{veto}} \\cdot (\\prod_{k=1}^5 d_k)^{1/5}$
   с жестким правом вето ($D = 0.0$ при `pains_alerts_count > 0` или `brenk_alerts_count > 0`).
3. Проверка соответствия всем ограничениям пользователя (`passes_all_constraints`, `constraint_violations`).
4. Сквозная оценка пула `GeneratedRawCandidate`, сохранение в `evaluated_candidates.parquet`
   и ранжирование топ-$K$ кандидатов.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from rdkit import Chem

from pishovichok.core.admet import predict_admet_metrics
from pishovichok.core.evaluation import calculate_rdkit_metrics
from pishovichok.core.generation import verify_scaffold_preservation
from pishovichok.state.schemas import (
    CalcRDKitMetrics,
    CandidateMolecule,
    GeneratedRawCandidate,
    ParentSeed,
    ParquetPointer,
    PredADMETMetrics,
    QueryConstraints,
    SARHypothesisRationale,
)
from pishovichok.state.storage import write_candidates_parquet


def harrington_one_sided_desirability(
    value: float,
    bad_limit: float,
    good_target: float,
) -> float:
    """Вычисляет одностороннюю частную функцию желательности Харрингтона $d = \\exp(-\\exp(-Y))$.

    Линейно отображает:
    - `bad_limit` -> $Y = 0.0$ ($d = e^{-1} \\approx 0.3679$)
    - `good_target` -> $Y = 2.0$ ($d = \\exp(-e^{-2}) \\approx 0.8734$)
    Работает как для максимизации (`good_target > bad_limit`), так и для минимизации (`good_target < bad_limit`).
    """
    span = good_target - bad_limit
    if abs(span) < 1e-9:
        return 0.5
    y_coord = 2.0 * (value - bad_limit) / span
    clamped_y = max(-6.0, min(8.0, float(y_coord)))
    d_val = math.exp(-math.exp(-clamped_y))
    return max(1e-6, min(1.0, d_val))


def calculate_harrington_mpo_score(
    calc_rdkit: CalcRDKitMetrics,
    pred_admet: PredADMETMetrics,
    parent_exp_pic50: float,
    constraints: QueryConstraints,
) -> float:
    """Вычисляет обобщенный многопараметрический индекс желательности Харрингтона $D \\in [0, 1]$.

    Формула (Раздел 6.1.4 ARCHITECTURE.md):
        D = I_veto * (d_1 * d_2 * d_3 * d_4 * d_5) ** (1/5)
    При срабатывании любого структурного алерта PAINS или Brenk накладывается жесткое вето: D = 0.0.
    """
    if calc_rdkit.pains_alerts_count > 0 or calc_rdkit.brenk_alerts_count > 0:
        return 0.0

    # d1: Максимизация прогнозной активности pIC50 относительно родителя
    d1_pic50 = harrington_one_sided_desirability(
        value=pred_admet.predicted_pic50,
        bad_limit=parent_exp_pic50 - 0.20,
        good_target=parent_exp_pic50 + 0.55,
    )

    # d2: Липофильность cLogP в целевом коридоре [min_logp, max_logp]
    d2_upper = harrington_one_sided_desirability(
        value=calc_rdkit.clogp,
        bad_limit=constraints.max_logp + 0.60,
        good_target=constraints.max_logp - 0.50,
    )
    if constraints.min_logp is not None:
        d2_lower = harrington_one_sided_desirability(
            value=calc_rdkit.clogp,
            bad_limit=constraints.min_logp - 0.50,
            good_target=constraints.min_logp + 0.40,
        )
        d2_clogp = min(d2_upper, d2_lower)
    else:
        d2_clogp = d2_upper

    # d3: Минимизация индекса синтетической доступности SAscore (<= max_sa_score)
    d3_sa = harrington_one_sided_desirability(
        value=calc_rdkit.sa_score,
        bad_limit=constraints.max_sa_score + 0.80,
        good_target=max(1.50, constraints.max_sa_score - 0.90),
    )

    # d4: Максимизация индекса лекарственного сходства QED (>= 0.55)
    d4_qed = harrington_one_sided_desirability(
        value=calc_rdkit.qed,
        bad_limit=0.35,
        good_target=0.65,
    )

    # d5: Минимизация вероятности кардиотоксичности hERG (<= max_herg_prob)
    d5_herg = harrington_one_sided_desirability(
        value=pred_admet.herg_liability_prob,
        bad_limit=min(0.90, constraints.max_herg_prob + 0.15),
        good_target=max(0.15, constraints.max_herg_prob - 0.22),
    )

    product = d1_pic50 * d2_clogp * d3_sa * d4_qed * d5_herg
    geom_mean = product ** (1.0 / 5.0)
    return round(max(0.0, min(1.0, geom_mean)), 4)


def check_candidate_constraints(
    calc_rdkit: CalcRDKitMetrics,
    pred_admet: PredADMETMetrics,
    candidate_mol: Chem.Mol,
    parent_mol: Chem.Mol,
    constraints: QueryConstraints,
) -> tuple[bool, list[str]]:
    """Проверяет выполнение всех целевых ограничений пользователя из `QueryConstraints`.

    Возвращает `(passes_all_constraints, constraint_violations)`.
    """
    violations: list[str] = []

    if calc_rdkit.clogp > constraints.max_logp:
        violations.append(
            f"cLogP = {calc_rdkit.clogp:.2f} > порога max_logp ({constraints.max_logp:.2f})"
        )

    if constraints.min_logp is not None and calc_rdkit.clogp < constraints.min_logp:
        violations.append(
            f"cLogP = {calc_rdkit.clogp:.2f} < порога min_logp ({constraints.min_logp:.2f})"
        )

    if calc_rdkit.sa_score > constraints.max_sa_score:
        violations.append(
            f"SAscore = {calc_rdkit.sa_score:.2f} > порога max_sa_score ({constraints.max_sa_score:.2f})"
        )

    if constraints.min_tpsa is not None and calc_rdkit.tpsa < constraints.min_tpsa:
        violations.append(
            f"TPSA = {calc_rdkit.tpsa:.2f} < порога min_tpsa ({constraints.min_tpsa:.2f})"
        )

    if calc_rdkit.mw > constraints.max_mw:
        violations.append(
            f"MW = {calc_rdkit.mw:.1f} > порога max_mw ({constraints.max_mw:.1f})"
        )

    if pred_admet.herg_liability_prob > constraints.max_herg_prob:
        violations.append(
            f"hERG prob = {pred_admet.herg_liability_prob:.2f} > порога max_herg_prob ({constraints.max_herg_prob:.2f})"
        )

    if constraints.require_no_pains:
        if calc_rdkit.pains_alerts_count > 0:
            violations.append(
                f"PAINS alerts = {calc_rdkit.pains_alerts_count} ({', '.join(calc_rdkit.pains_matched_names)})"
            )
        if calc_rdkit.brenk_alerts_count > 0:
            violations.append(
                f"Brenk alerts = {calc_rdkit.brenk_alerts_count} ({', '.join(calc_rdkit.brenk_matched_names)})"
            )

    if constraints.freeze_scaffold:
        preserved, _scaf = verify_scaffold_preservation(
            parent_mol=parent_mol,
            candidate_mol=candidate_mol,
            freeze_scaffold=True,
        )
        if not preserved:
            violations.append("Нарушено сохранение базового скаффолда Bemis-Murcko")

    return (len(violations) == 0, violations)


def build_hypothesis_rationale(
    raw_cand: GeneratedRawCandidate,
    calc_rdkit: CalcRDKitMetrics,
    pred_admet: PredADMETMetrics,
) -> SARHypothesisRationale:
    """Формирует объект Категории 4 знаний (`SARHypothesisRationale`) с детерминированным обоснованием."""
    origin_label = {
        "mmp_smirks_rule": f"правила MMP {raw_cand.rule_id} ({raw_cand.support_pairs_count} пар в однородном тесте)",
        "crem_fragment_replacement": f"контекстной замены фрагментов CReM (r=3, {raw_cand.rule_id})",
        "bioisostere_catalog": f"медико-химического каталога биоизостеров ({raw_cand.rule_id})",
    }[raw_cand.generation_origin]

    logp_dir = (
        f"снижает cLogP на {abs(calc_rdkit.delta_clogp_vs_parent):.2f}"
        if calc_rdkit.delta_clogp_vs_parent <= 0.0
        else f"повышает cLogP на {calc_rdkit.delta_clogp_vs_parent:.2f}"
    )
    tpsa_dir = (
        f"увеличивает полярную поверхность TPSA на +{calc_rdkit.delta_tpsa_vs_parent:.1f} Å²"
        if calc_rdkit.delta_tpsa_vs_parent >= 0.0
        else f"изменяет TPSA на {calc_rdkit.delta_tpsa_vs_parent:.1f} Å²"
    )
    text_summary = (
        f"Модификация '{raw_cand.applied_smirks}' вокруг родителя {raw_cand.parent_seed.mol_id} "
        f"на базе {origin_label} дает ожидаемый прирост активности ΔpIC50 = "
        f"{raw_cand.historical_mean_delta_pic50:+.2f} (прогноз pIC50 = {pred_admet.predicted_pic50:.2f}), "
        f"{logp_dir}, {tpsa_dir} и обеспечивает риск блокады hERG на уровне "
        f"{pred_admet.herg_liability_prob:.2f} ({pred_admet.herg_risk_level})."
    )

    return SARHypothesisRationale(
        generation_origin=raw_cand.generation_origin,
        rule_id=raw_cand.rule_id,
        applied_smirks=raw_cand.applied_smirks,
        replaced_group_smiles=raw_cand.replaced_group_smiles,
        introduced_group_smiles=raw_cand.introduced_group_smiles,
        historical_mean_delta_pic50=round(raw_cand.historical_mean_delta_pic50, 4),
        historical_std_delta_pic50=round(raw_cand.historical_std_delta_pic50, 4),
        support_pairs_count=raw_cand.support_pairs_count,
        assay_homogeneous=raw_cand.assay_homogeneous,
        text_summary=text_summary,
    )


def evaluate_lead_reference(
    lead_seed: ParentSeed,
    constraints: QueryConstraints,
    *,
    preferred_admet_backend: Literal[
        "auto",
        "admet_ai_dmpnn",
        "catboost_local_qsar",
        "rdkit_analytical_qsar",
    ]
    | None = None,
) -> CandidateMolecule:
    """Формирует референсную карточку `CandidateMolecule` для исходного соединения-лидера (Сценарий 2)."""
    lead_mol = Chem.MolFromSmiles(lead_seed.canonical_smiles)
    if lead_mol is None:
        raise ValueError(f"Invalid lead SMILES: {lead_seed.canonical_smiles}")

    calc_metrics = calculate_rdkit_metrics(lead_mol, parent_mol_or_smiles=lead_mol)
    admet_metrics = predict_admet_metrics(
        mol_or_smiles=lead_mol,
        rdkit_metrics=calc_metrics,
        parent_exp_pic50=lead_seed.exp_pic50,
        historical_mean_delta_pic50=0.0,
        support_pairs_count=1,
        assay_homogeneous=True,
        generation_origin="mmp_smirks_rule",
        preferred_backend=preferred_admet_backend,
    )
    mpo_score = calculate_harrington_mpo_score(
        calc_rdkit=calc_metrics,
        pred_admet=admet_metrics,
        parent_exp_pic50=lead_seed.exp_pic50,
        constraints=constraints,
    )
    passes_all, violations = check_candidate_constraints(
        calc_rdkit=calc_metrics,
        pred_admet=admet_metrics,
        candidate_mol=lead_mol,
        parent_mol=lead_mol,
        constraints=constraints,
    )
    rationale = SARHypothesisRationale(
        generation_origin="mmp_smirks_rule",
        rule_id="LEAD_REF",
        applied_smirks="[*:1]H>>[*:1]H",
        replaced_group_smiles="[*:1]H",
        introduced_group_smiles="[*:1]H",
        historical_mean_delta_pic50=0.0,
        historical_std_delta_pic50=0.0,
        support_pairs_count=1,
        assay_homogeneous=True,
        text_summary=f"Исходная структура соединения-лидера ({lead_seed.mol_id}) до направленной оптимизации.",
    )

    return CandidateMolecule(
        candidate_id=f"LEAD_{lead_seed.mol_id}",
        canonical_smiles=lead_seed.canonical_smiles,
        scaffold_smiles=lead_seed.scaffold_smiles,
        generation_iteration=1,
        parent_chembl_id=lead_seed.mol_id,
        parent_smiles=lead_seed.canonical_smiles,
        parent_exp_pic50=lead_seed.exp_pic50,
        parent_exp_ic50_nm=lead_seed.exp_ic50_nm,
        parent_bao_format=lead_seed.bao_format,
        parent_assay_type=lead_seed.assay_type,
        parent_source_dataset=lead_seed.source_dataset,
        calc_rdkit=calc_metrics,
        pred_admet=admet_metrics,
        hypothesis_rationale=rationale,
        mpo_harrington_score=mpo_score,
        passes_all_constraints=passes_all,
        constraint_violations=violations,
        highlighted_atom_indices=[],
    )


def evaluate_and_rank_candidates(
    raw_candidates: list[GeneratedRawCandidate],
    constraints: QueryConstraints,
    session_dir: Path | str,
    *,
    filename: str = "evaluated_candidates.parquet",
    preferred_admet_backend: Literal[
        "auto",
        "admet_ai_dmpnn",
        "catboost_local_qsar",
        "rdkit_analytical_qsar",
    ]
    | None = None,
) -> tuple[ParquetPointer, list[CandidateMolecule]]:
    """Оценивает пул сгенерированных кандидатов, сохраняет `evaluated_candidates.parquet` и ранжирует их.

    Порядок ранжирования:
    1. `passes_all_constraints` (`True` выше `False`)
    2. `mpo_harrington_score` (по убыванию)
    3. `pred_admet.predicted_pic50` (по убыванию)
    4. `-calc_rdkit.clogp` (при равном скоре предпочтение более гидрофильным структурам)
    """
    resolved_session_dir = Path(session_dir).resolve()
    resolved_session_dir.mkdir(parents=True, exist_ok=True)

    evaluated_list: list[CandidateMolecule] = []

    for idx, raw_cand in enumerate(raw_candidates, start=1):
        cand_mol = Chem.MolFromSmiles(raw_cand.smiles)
        parent_mol = Chem.MolFromSmiles(raw_cand.parent_seed.canonical_smiles)
        if cand_mol is None or parent_mol is None:
            continue

        calc_metrics = calculate_rdkit_metrics(
            mol_or_smiles=cand_mol,
            parent_mol_or_smiles=parent_mol,
        )
        admet_metrics = predict_admet_metrics(
            mol_or_smiles=cand_mol,
            rdkit_metrics=calc_metrics,
            parent_exp_pic50=raw_cand.parent_seed.exp_pic50,
            historical_mean_delta_pic50=raw_cand.historical_mean_delta_pic50,
            support_pairs_count=raw_cand.support_pairs_count,
            assay_homogeneous=raw_cand.assay_homogeneous,
            generation_origin=raw_cand.generation_origin,
            preferred_backend=preferred_admet_backend,
        )
        mpo_score = calculate_harrington_mpo_score(
            calc_rdkit=calc_metrics,
            pred_admet=admet_metrics,
            parent_exp_pic50=raw_cand.parent_seed.exp_pic50,
            constraints=constraints,
        )
        passes_all, violations = check_candidate_constraints(
            calc_rdkit=calc_metrics,
            pred_admet=admet_metrics,
            candidate_mol=cand_mol,
            parent_mol=parent_mol,
            constraints=constraints,
        )
        rationale = build_hypothesis_rationale(
            raw_cand=raw_cand,
            calc_rdkit=calc_metrics,
            pred_admet=admet_metrics,
        )

        cand_id = f"CAND_IT{raw_cand.generation_iteration}_{idx:02d}"
        evaluated_list.append(
            CandidateMolecule(
                candidate_id=cand_id,
                canonical_smiles=raw_cand.smiles,
                scaffold_smiles=raw_cand.scaffold_smiles,
                generation_iteration=raw_cand.generation_iteration,
                parent_chembl_id=raw_cand.parent_seed.mol_id,
                parent_smiles=raw_cand.parent_seed.canonical_smiles,
                parent_exp_pic50=raw_cand.parent_seed.exp_pic50,
                parent_exp_ic50_nm=raw_cand.parent_seed.exp_ic50_nm,
                parent_bao_format=raw_cand.parent_seed.bao_format,
                parent_assay_type=raw_cand.parent_seed.assay_type,
                parent_source_dataset=raw_cand.parent_seed.source_dataset,
                calc_rdkit=calc_metrics,
                pred_admet=admet_metrics,
                hypothesis_rationale=rationale,
                mpo_harrington_score=mpo_score,
                passes_all_constraints=passes_all,
                constraint_violations=violations,
                highlighted_atom_indices=raw_cand.highlighted_atom_indices,
            )
        )

    evaluated_list.sort(
        key=lambda c: (
            bool(c.passes_all_constraints),
            float(c.mpo_harrington_score),
            float(c.pred_admet.predicted_pic50),
            -float(c.calc_rdkit.clogp),
        ),
        reverse=True,
    )

    candidates_pointer = write_candidates_parquet(
        candidates=evaluated_list,
        session_dir=resolved_session_dir,
        filename=filename,
    )

    top_k = evaluated_list[: constraints.top_k_candidates]
    return (candidates_pointer, top_k)
