"""Слой 1: Детерминированное хемоинформатическое ядро (без участия LLM)."""

from __future__ import annotations

from pishovichok.core.admet import (
    build_catboost_feature_vector,
    classify_herg_risk,
    classify_solubility,
    determine_prediction_confidence,
    predict_admet_metrics,
)
from pishovichok.core.bioisosteres import (
    CURATED_BIOISOSTERE_CATALOG,
    BioisostereRule,
)
from pishovichok.core.curation import (
    StandardizedMoleculeResult,
    convert_to_pchembl,
    curate_chembl_or_csv,
    parse_activity_cell,
    standardize_molecule_detailed,
    standardize_smiles,
)
from pishovichok.core.evaluation import (
    calculate_rdkit_metrics,
    calculate_sa_score,
    check_structural_alerts,
    count_lipinski_violations,
    decompose_brics_precursors,
)
from pishovichok.core.generation import (
    apply_smirks_to_molecule,
    build_constrained_reaction_smarts,
    compute_highlighted_atoms,
    fragment_contains_halogen,
    generate_candidate_structures,
    select_parent_seeds_from_curated,
    verify_scaffold_preservation,
)
from pishovichok.core.sar_analysis import (
    calculate_sali,
    canonicalize_substituent_smiles,
    extract_mmp_and_cliffs,
    extract_mmp_with_background_fallback,
    fragment_molecule_mmpa,
)
from pishovichok.core.scoring import (
    build_hypothesis_rationale,
    calculate_harrington_mpo_score,
    check_candidate_constraints,
    evaluate_and_rank_candidates,
    evaluate_lead_reference,
    harrington_one_sided_desirability,
)

__all__ = [
    "CURATED_BIOISOSTERE_CATALOG",
    "BioisostereRule",
    "StandardizedMoleculeResult",
    "apply_smirks_to_molecule",
    "build_catboost_feature_vector",
    "build_constrained_reaction_smarts",
    "build_hypothesis_rationale",
    "calculate_harrington_mpo_score",
    "calculate_rdkit_metrics",
    "calculate_sa_score",
    "calculate_sali",
    "canonicalize_substituent_smiles",
    "check_candidate_constraints",
    "check_structural_alerts",
    "classify_herg_risk",
    "classify_solubility",
    "compute_highlighted_atoms",
    "convert_to_pchembl",
    "count_lipinski_violations",
    "curate_chembl_or_csv",
    "decompose_brics_precursors",
    "determine_prediction_confidence",
    "evaluate_and_rank_candidates",
    "evaluate_lead_reference",
    "extract_mmp_and_cliffs",
    "extract_mmp_with_background_fallback",
    "fragment_contains_halogen",
    "fragment_molecule_mmpa",
    "generate_candidate_structures",
    "harrington_one_sided_desirability",
    "parse_activity_cell",
    "predict_admet_metrics",
    "select_parent_seeds_from_curated",
    "standardize_molecule_detailed",
    "standardize_smiles",
    "verify_scaffold_preservation",
]
