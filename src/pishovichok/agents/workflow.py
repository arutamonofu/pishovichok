"""Оркестратор мультиагентного графа LangGraph (Слой 3).

Объединяет узлы конечного автомата в единый граф состояний:
  START -> research -> sar -> critic --(refilter_data)--> research
                               |
                    (proceed_to_design)
                               |
                               v
                            design -> evaluation -> critic --(redesign_molecules)--> design
                                                       |
                                                   (approve)
                                                       |
                                                       v
                                                      END

Содержит:
- `build_sar_graph()`: сборка и компиляция графа `StateGraph`.
- `run_sar_pipeline(...)`: сквозной запуск с экспортом артефактов (`trace_log.json`, `candidates.csv`, `candidates.sdf`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from rdkit import Chem
from rdkit.Chem import AllChem

from pishovichok.agents.critic_node import critic_node, route_after_critic
from pishovichok.agents.design_node import design_node
from pishovichok.agents.evaluation_node import evaluation_node
from pishovichok.agents.research_node import research_node
from pishovichok.agents.sar_node import sar_node
from pishovichok.state.graph_state import SARGraphState, create_initial_graph_state
from pishovichok.state.schemas import CandidateMolecule, QueryConstraints, ToolConfig
from pishovichok.state.storage import (
    create_session_workspace,
    export_session_meta,
    export_trace_log,
)


def build_sar_graph(
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> Any:
    """Создает и компилирует конечный автомат мультиагентного скрининга на базе LangGraph."""
    workflow = StateGraph(state_schema=SARGraphState)

    # Добавление функциональных узлов
    workflow.add_node("research", research_node)
    workflow.add_node("sar", sar_node)
    workflow.add_node("design", design_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("critic", critic_node)

    # Детерминированные линейные ребра
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "sar")
    workflow.add_edge("sar", "critic")
    workflow.add_edge("design", "evaluation")
    workflow.add_edge("evaluation", "critic")

    # Условные переходы под управлением узла-критика
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "research": "research",
            "design": "design",
            "end": END,
        },
    )

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )


def export_candidates_to_csv(
    candidates: list[CandidateMolecule],
    target_path: str | Path,
) -> Path:
    """Экспортирует список кандидатов в структурированный CSV-файл."""
    path = Path(target_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate_id",
        "canonical_smiles",
        "scaffold_smiles",
        "generation_iteration",
        # Категория 1: Экспериментальный факт
        "parent_chembl_id",
        "parent_smiles",
        "parent_exp_pic50",
        "parent_exp_ic50_nm",
        "parent_bao_format",
        "parent_assay_type",
        "parent_source_dataset",
        # Категория 2: Расчет RDKit
        "mw",
        "clogp",
        "tpsa",
        "qed",
        "sa_score",
        "lipinski_violations",
        "pains_alerts_count",
        "brenk_alerts_count",
        "delta_clogp_vs_parent",
        "delta_tpsa_vs_parent",
        "brics_fragments",
        # Категория 3: Прогноз ADMET
        "model_source",
        "predicted_pic50",
        "log_s_aq",
        "herg_liability_prob",
        "herg_risk_level",
        "bbb_permeability_prob",
        "prediction_confidence",
        # Категория 4: Гипотеза SAR
        "rule_id",
        "applied_smirks",
        "historical_mean_delta_pic50",
        "support_pairs_count",
        "assay_homogeneous",
        "hypothesis_summary",
        # Интегральный скоринг
        "mpo_harrington_score",
        "passes_all_constraints",
        "constraint_violations",
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            writer.writerow(
                {
                    "candidate_id": cand.candidate_id,
                    "canonical_smiles": cand.canonical_smiles,
                    "scaffold_smiles": cand.scaffold_smiles,
                    "generation_iteration": cand.generation_iteration,
                    "parent_chembl_id": cand.parent_chembl_id,
                    "parent_smiles": cand.parent_smiles,
                    "parent_exp_pic50": cand.parent_exp_pic50,
                    "parent_exp_ic50_nm": cand.parent_exp_ic50_nm,
                    "parent_bao_format": cand.parent_bao_format,
                    "parent_assay_type": cand.parent_assay_type,
                    "parent_source_dataset": cand.parent_source_dataset,
                    "mw": cand.calc_rdkit.mw,
                    "clogp": cand.calc_rdkit.clogp,
                    "tpsa": cand.calc_rdkit.tpsa,
                    "qed": cand.calc_rdkit.qed,
                    "sa_score": cand.calc_rdkit.sa_score,
                    "lipinski_violations": cand.calc_rdkit.lipinski_violations,
                    "pains_alerts_count": cand.calc_rdkit.pains_alerts_count,
                    "brenk_alerts_count": cand.calc_rdkit.brenk_alerts_count,
                    "delta_clogp_vs_parent": cand.calc_rdkit.delta_clogp_vs_parent,
                    "delta_tpsa_vs_parent": cand.calc_rdkit.delta_tpsa_vs_parent,
                    "brics_fragments": ";".join(cand.calc_rdkit.brics_fragments),
                    "model_source": cand.pred_admet.model_source,
                    "predicted_pic50": cand.pred_admet.predicted_pic50,
                    "log_s_aq": cand.pred_admet.log_s_aq,
                    "herg_liability_prob": cand.pred_admet.herg_liability_prob,
                    "herg_risk_level": cand.pred_admet.herg_risk_level,
                    "bbb_permeability_prob": cand.pred_admet.bbb_permeability_prob,
                    "prediction_confidence": cand.pred_admet.prediction_confidence,
                    "rule_id": cand.hypothesis_rationale.rule_id,
                    "applied_smirks": cand.hypothesis_rationale.applied_smirks,
                    "historical_mean_delta_pic50": cand.hypothesis_rationale.historical_mean_delta_pic50,
                    "support_pairs_count": cand.hypothesis_rationale.support_pairs_count,
                    "assay_homogeneous": cand.hypothesis_rationale.assay_homogeneous,
                    "hypothesis_summary": cand.hypothesis_rationale.text_summary,
                    "mpo_harrington_score": cand.mpo_harrington_score,
                    "passes_all_constraints": cand.passes_all_constraints,
                    "constraint_violations": "; ".join(cand.constraint_violations),
                }
            )
    return path


def export_candidates_to_sdf(
    candidates: list[CandidateMolecule],
    target_path: str | Path,
) -> Path:
    """Экспортирует молекулы в SDF с 2D-координатами и метаданными 4 категорий знаний."""
    path = Path(target_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(path))
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand.canonical_smiles)
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mol.SetProp("_Name", cand.candidate_id)
        mol.SetProp("Canonical_SMILES", cand.canonical_smiles)
        # Категория 1: Экспериментальный факт
        mol.SetProp("Cat1_Parent_ChEMBL_ID", cand.parent_chembl_id)
        mol.SetProp("Cat1_Parent_SMILES", cand.parent_smiles)
        mol.SetProp("Cat1_Parent_Exp_pIC50", str(cand.parent_exp_pic50))
        mol.SetProp("Cat1_Parent_Exp_IC50_nM", str(cand.parent_exp_ic50_nm))
        mol.SetProp("Cat1_Parent_BAO_Format", cand.parent_bao_format)
        mol.SetProp("Cat1_Parent_Assay_Type", cand.parent_assay_type)
        # Категория 2: Расчет RDKit
        mol.SetProp("Cat2_MW", str(cand.calc_rdkit.mw))
        mol.SetProp("Cat2_cLogP", str(cand.calc_rdkit.clogp))
        mol.SetProp("Cat2_TPSA", str(cand.calc_rdkit.tpsa))
        mol.SetProp("Cat2_QED", str(cand.calc_rdkit.qed))
        mol.SetProp("Cat2_SAscore", str(cand.calc_rdkit.sa_score))
        mol.SetProp("Cat2_PAINS_Alerts", str(cand.calc_rdkit.pains_alerts_count))
        mol.SetProp("Cat2_Brenk_Alerts", str(cand.calc_rdkit.brenk_alerts_count))
        mol.SetProp("Cat2_BRICS_Fragments", ";".join(cand.calc_rdkit.brics_fragments))
        # Категория 3: Прогноз ADMET
        mol.SetProp("Cat3_Model_Source", cand.pred_admet.model_source)
        mol.SetProp("Cat3_Predicted_pIC50", str(cand.pred_admet.predicted_pic50))
        mol.SetProp("Cat3_LogS_Water", str(cand.pred_admet.log_s_aq))
        mol.SetProp(
            "Cat3_hERG_Liability_Prob", str(cand.pred_admet.herg_liability_prob)
        )
        mol.SetProp("Cat3_hERG_Risk_Level", cand.pred_admet.herg_risk_level)
        # Категория 4: Гипотеза SAR
        mol.SetProp("Cat4_Rule_ID", cand.hypothesis_rationale.rule_id)
        mol.SetProp("Cat4_Applied_SMIRKS", cand.hypothesis_rationale.applied_smirks)
        mol.SetProp(
            "Cat4_Historical_Mean_Delta_pIC50",
            str(cand.hypothesis_rationale.historical_mean_delta_pic50),
        )
        mol.SetProp(
            "Cat4_Assay_Homogeneous", str(cand.hypothesis_rationale.assay_homogeneous)
        )
        mol.SetProp("Cat4_Hypothesis_Text", cand.hypothesis_rationale.text_summary)
        # Интегральный скоринг
        mol.SetProp("MPO_Harrington_Score", str(cand.mpo_harrington_score))
        mol.SetProp("Passes_All_Constraints", str(cand.passes_all_constraints))
        writer.write(mol)

    writer.close()
    return path


def run_sar_pipeline(
    user_query: str,
    session_id: str | None = None,
    workspace_dir: str | Path | None = None,
    uploaded_csv_path: str | Path | None = None,
    query_constraints: QueryConstraints | None = None,
    tool_config: ToolConfig | None = None,
    export_artifacts: bool = True,
) -> tuple[SARGraphState, Path | None, Path | None, Path | None]:
    """Выполняет сквозной скрининг лекарственных кандидатов в графе LangGraph.

    Возвращает `(final_state, trace_log_path, csv_path, sdf_path)`.
    """
    actual_session_id, session_dir = create_session_workspace(
        base_workspace_dir=workspace_dir,
        session_id=session_id,
    )

    initial_state = create_initial_graph_state(
        user_query=user_query,
        session_id=actual_session_id,
        workspace_dir=session_dir,
        query_constraints=query_constraints,
        uploaded_csv_path=uploaded_csv_path,
        tool_config=tool_config,
    )

    compiled_graph = build_sar_graph()
    final_state_raw = compiled_graph.invoke(initial_state)
    final_state: SARGraphState = final_state_raw  # type: ignore[assignment]

    trace_path: Path | None = None
    csv_path: Path | None = None
    sdf_path: Path | None = None

    if export_artifacts:
        trace_path = export_trace_log(final_state.get("trace_log", []), session_dir)
        export_session_meta(dict(final_state), session_dir)
        top_candidates = final_state.get("top_candidates", [])
        if top_candidates:
            csv_path = export_candidates_to_csv(
                top_candidates, session_dir / "candidates.csv"
            )
            sdf_path = export_candidates_to_sdf(
                top_candidates, session_dir / "candidates.sdf"
            )
        if final_state.get("final_report_markdown"):
            (session_dir / "report.md").write_text(
                final_state["final_report_markdown"], encoding="utf-8"
            )

    return final_state, trace_path, csv_path, sdf_path
