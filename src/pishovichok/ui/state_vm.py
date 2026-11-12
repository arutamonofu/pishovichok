"""View-Model над `SARGraphState` для реактивного обновления MPO, таблицы и графиков (Слой 4).

Обеспечивает:
1. Хранение плоского буфера кандидатов (без SVG и 3D-координат) для `ui.aggrid` и `ui.echart`.
2. Мгновенный векторный пересчёт интегрального индекса Харрингтона и статусов ограничений
   при изменении весов и порогов на ползунках без перезапуска `agents/workflow.py`.
3. Инкрементальную выдачу новых строк для `applyTransaction({'add': [...]})`.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from pishovichok.core.scoring import harrington_one_sided_desirability
from pishovichok.state.schemas import CandidateMolecule, QueryConstraints


class MPOWeightsAndThresholds(BaseModel):
    """Интерактивные веса и пороги MPO-панели (управляются ползунками без перезапуска графа)."""

    w_activity: float = Field(default=1.0, ge=0.0, le=5.0)
    w_logp: float = Field(default=1.0, ge=0.0, le=5.0)
    w_sa: float = Field(default=1.0, ge=0.0, le=5.0)
    w_qed: float = Field(default=1.0, ge=0.0, le=5.0)
    w_herg: float = Field(default=1.0, ge=0.0, le=5.0)

    min_logp: float | None = Field(default=None)
    max_logp: float = Field(default=4.5, ge=-2.0, le=8.0)
    max_sa_score: float = Field(default=3.2, ge=1.0, le=10.0)
    max_herg_prob: float = Field(default=0.5, ge=0.05, le=1.0)
    max_mw: float = Field(default=550.0, ge=150.0, le=1000.0)

    @classmethod
    def from_constraints(cls, constraints: QueryConstraints) -> MPOWeightsAndThresholds:
        """Инициализирует пороги View-Model из `QueryConstraints`."""
        return cls(
            min_logp=constraints.min_logp,
            max_logp=constraints.max_logp,
            max_sa_score=constraints.max_sa_score,
            max_herg_prob=constraints.max_herg_prob,
            max_mw=constraints.max_mw,
        )


class CandidateRowVM(BaseModel):
    """Лёгкая строковая проекция кандидата для `ui.aggrid` и `mpo_canvas` (без SVG/3D)."""

    candidate_id: str
    canonical_smiles: str
    scaffold_smiles: str
    parent_chembl_id: str
    parent_exp_pic50: float
    predicted_pic50: float
    delta_activity: float
    clogp: float
    delta_clogp: float
    tpsa: float
    delta_tpsa: float
    sa_score: float
    qed: float
    mw: float
    herg_prob: float
    herg_risk_level: str
    delta_herg: float
    pains_count: int
    brenk_count: int
    rule_id: str
    applied_smirks: str
    generation_origin: str
    mpo_score: float
    passes_all: bool
    violations: list[str] = Field(default_factory=list)
    desirability_d1_activity: float = 0.0
    desirability_d2_clogp: float = 0.0
    desirability_d3_sa: float = 0.0
    desirability_d4_qed: float = 0.0
    desirability_d5_herg: float = 0.0
    pinned: bool = False
    excluded: bool = False
    pca_x: float = 0.0
    pca_y: float = 0.0

    @property
    def status_label(self) -> str:
        if self.pains_count > 0 or self.brenk_count > 0:
            return "⛔ Veto"
        if self.passes_all:
            return "✅ OK"
        return "⚠️ Компромисс"

    def to_aggrid_row(self, session_id: str) -> dict[str, Any]:
        """Формирует лёгкий словарь строки для `ui.aggrid` (без SVG и 3D-координат)."""
        svg_url = f"/api/sessions/{session_id}/mol/{self.candidate_id}/svg"
        return {
            "candidate_id": self.candidate_id,
            "preview_html": (
                f'<img src="{svg_url}" loading="lazy" '
                f'alt="{self.candidate_id}" '
                f'style="height:54px;width:auto;display:block;margin:2px auto;border-radius:4px;background:#fff;" />'
            ),
            "parent_chembl_id": self.parent_chembl_id,
            "predicted_pic50": round(self.predicted_pic50, 2),
            "delta_activity": round(self.delta_activity, 2),
            "clogp": round(self.clogp, 2),
            "delta_clogp": round(self.delta_clogp, 2),
            "tpsa": round(self.tpsa, 1),
            "delta_tpsa": round(self.delta_tpsa, 1),
            "sa_score": round(self.sa_score, 2),
            "herg_prob": round(self.herg_prob, 2),
            "delta_herg": round(self.delta_herg, 2),
            "critic_status": self.status_label,
            "mpo_score": round(self.mpo_score, 3),
            "passes_all": self.passes_all,
            "pinned": self.pinned,
            "excluded": self.excluded,
            "rule_id": self.rule_id,
        }


class ReferencePointVM(BaseModel):
    """Точка референсного соединения (ChEMBL или Wet-Lab CSV) для проекции химпространства."""

    mol_id: str
    canonical_smiles: str
    pchembl_value: float
    clogp: float
    tpsa: float
    source_group: Literal["chembl", "wetlab", "lead"]
    pca_x: float = 0.0
    pca_y: float = 0.0


class StateViewModel:
    """View-Model над `SARGraphState` с векторным пересчётом MPO и буфером транзакций."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.weights = MPOWeightsAndThresholds()
        self.candidates_by_id: dict[str, CandidateMolecule] = {}
        self.rows_by_id: dict[str, CandidateRowVM] = {}
        self.row_order: list[str] = []
        self.emitted_to_grid_ids: set[str] = set()
        self.reference_points: list[ReferencePointVM] = []
        self.lead_reference: CandidateMolecule | None = None
        self.lead_pca_coords: tuple[float, float] | None = None
        self.pinned_ids: set[str] = set()
        self.excluded_ids: set[str] = set()
        self.selected_candidate_id: str | None = None

    def sync_constraints(self, constraints: QueryConstraints | None) -> None:
        """Синхронизирует пороги слайдеров при обновлении `QueryConstraints` графом."""
        if constraints is None:
            return
        self.weights.min_logp = constraints.min_logp
        self.weights.max_logp = constraints.max_logp
        self.weights.max_sa_score = constraints.max_sa_score
        self.weights.max_herg_prob = constraints.max_herg_prob
        self.weights.max_mw = constraints.max_mw

    def upsert_candidates(
        self,
        candidates: list[CandidateMolecule],
        lead_reference: CandidateMolecule | None = None,
    ) -> list[str]:
        """Добавляет или обновляет кандидатов и возвращает ID впервые появившихся строк."""
        if lead_reference is not None:
            self.lead_reference = lead_reference

        newly_added_ids: list[str] = []
        lead_herg = (
            self.lead_reference.pred_admet.herg_liability_prob
            if self.lead_reference is not None
            else None
        )

        for cand in candidates:
            is_new = cand.candidate_id not in self.candidates_by_id
            self.candidates_by_id[cand.candidate_id] = cand
            if is_new:
                self.row_order.append(cand.candidate_id)
                newly_added_ids.append(cand.candidate_id)

            delta_act = round(cand.pred_admet.predicted_pic50 - cand.parent_exp_pic50, 4)
            delta_herg = (
                round(cand.pred_admet.herg_liability_prob - lead_herg, 4)
                if lead_herg is not None
                else 0.0
            )
            existing_row = self.rows_by_id.get(cand.candidate_id)
            row_vm = CandidateRowVM(
                candidate_id=cand.candidate_id,
                canonical_smiles=cand.canonical_smiles,
                scaffold_smiles=cand.scaffold_smiles,
                parent_chembl_id=cand.parent_chembl_id,
                parent_exp_pic50=cand.parent_exp_pic50,
                predicted_pic50=cand.pred_admet.predicted_pic50,
                delta_activity=delta_act,
                clogp=cand.calc_rdkit.clogp,
                delta_clogp=cand.calc_rdkit.delta_clogp_vs_parent,
                tpsa=cand.calc_rdkit.tpsa,
                delta_tpsa=cand.calc_rdkit.delta_tpsa_vs_parent,
                sa_score=cand.calc_rdkit.sa_score,
                qed=cand.calc_rdkit.qed,
                mw=cand.calc_rdkit.mw,
                herg_prob=cand.pred_admet.herg_liability_prob,
                herg_risk_level=cand.pred_admet.herg_risk_level,
                delta_herg=delta_herg,
                pains_count=cand.calc_rdkit.pains_alerts_count,
                brenk_count=cand.calc_rdkit.brenk_alerts_count,
                rule_id=cand.hypothesis_rationale.rule_id,
                applied_smirks=cand.hypothesis_rationale.applied_smirks,
                generation_origin=cand.hypothesis_rationale.generation_origin,
                mpo_score=cand.mpo_harrington_score,
                passes_all=cand.passes_all_constraints,
                violations=list(cand.constraint_violations),
                pinned=cand.candidate_id in self.pinned_ids,
                excluded=cand.candidate_id in self.excluded_ids,
                pca_x=existing_row.pca_x if existing_row else 0.0,
                pca_y=existing_row.pca_y if existing_row else 0.0,
            )
            self.rows_by_id[cand.candidate_id] = row_vm

        self.recalculate_mpo(self.weights)
        if self.selected_candidate_id is None and self.row_order:
            self.selected_candidate_id = self.row_order[0]
        return newly_added_ids

    def update_pca_coordinates(
        self,
        candidate_coords: dict[str, tuple[float, float]],
        ref_coords: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Обновляет 2D-координаты проекции химпространства для кандидатов и референсов."""
        for cid, (px, py) in candidate_coords.items():
            if cid in self.rows_by_id:
                self.rows_by_id[cid].pca_x = px
                self.rows_by_id[cid].pca_y = py
        if ref_coords:
            for ref in self.reference_points:
                if ref.mol_id in ref_coords:
                    ref.pca_x, ref.pca_y = ref_coords[ref.mol_id]

    def recalculate_mpo(
        self, weights: MPOWeightsAndThresholds | None = None
    ) -> list[CandidateRowVM]:
        """Мгновенно векторно пересчитывает MPO-скоры и статусы ограничений без вызова графа агентов."""
        if weights is not None:
            self.weights = weights

        if not self.row_order:
            return []

        w_vec = np.array(
            [
                self.weights.w_activity,
                self.weights.w_logp,
                self.weights.w_sa,
                self.weights.w_qed,
                self.weights.w_herg,
            ],
            dtype=np.float64,
        )
        w_sum = float(np.sum(w_vec))
        if w_sum <= 1e-9:
            w_vec = np.ones(5, dtype=np.float64)
            w_sum = 5.0

        rows = [self.rows_by_id[cid] for cid in self.row_order if cid in self.rows_by_id]
        n = len(rows)
        d_matrix = np.zeros((n, 5), dtype=np.float64)
        veto_mask = np.ones(n, dtype=np.float64)

        for i, row in enumerate(rows):
            if row.pains_count > 0 or row.brenk_count > 0:
                veto_mask[i] = 0.0

            d1 = harrington_one_sided_desirability(
                row.predicted_pic50,
                row.parent_exp_pic50 - 0.20,
                row.parent_exp_pic50 + 0.55,
            )
            d2_up = harrington_one_sided_desirability(
                row.clogp,
                self.weights.max_logp + 0.60,
                self.weights.max_logp - 0.50,
            )
            if self.weights.min_logp is not None:
                d2_low = harrington_one_sided_desirability(
                    row.clogp,
                    self.weights.min_logp - 0.50,
                    self.weights.min_logp + 0.40,
                )
                d2 = min(d2_up, d2_low)
            else:
                d2 = d2_up

            d3 = harrington_one_sided_desirability(
                row.sa_score,
                self.weights.max_sa_score + 0.80,
                max(1.50, self.weights.max_sa_score - 0.90),
            )
            d4 = harrington_one_sided_desirability(row.qed, 0.35, 0.65)
            d5 = harrington_one_sided_desirability(
                row.herg_prob,
                min(0.90, self.weights.max_herg_prob + 0.15),
                max(0.15, self.weights.max_herg_prob - 0.22),
            )

            d_matrix[i] = [d1, d2, d3, d4, d5]
            row.desirability_d1_activity = round(d1, 4)
            row.desirability_d2_clogp = round(d2, 4)
            row.desirability_d3_sa = round(d3, 4)
            row.desirability_d4_qed = round(d4, 4)
            row.desirability_d5_herg = round(d5, 4)

            # Пересчёт нарушений по текущим порогам слайдеров
            violations: list[str] = []
            if row.clogp > self.weights.max_logp:
                violations.append(
                    f"cLogP = {row.clogp:.2f} > {self.weights.max_logp:.2f}"
                )
            if self.weights.min_logp is not None and row.clogp < self.weights.min_logp:
                violations.append(
                    f"cLogP = {row.clogp:.2f} < {self.weights.min_logp:.2f}"
                )
            if row.sa_score > self.weights.max_sa_score:
                violations.append(
                    f"SAscore = {row.sa_score:.2f} > {self.weights.max_sa_score:.2f}"
                )
            if row.herg_prob > self.weights.max_herg_prob:
                violations.append(
                    f"hERG = {row.herg_prob:.2f} > {self.weights.max_herg_prob:.2f}"
                )
            if row.mw > self.weights.max_mw:
                violations.append(f"MW = {row.mw:.1f} > {self.weights.max_mw:.1f}")
            if row.pains_count > 0:
                violations.append(f"PAINS = {row.pains_count}")
            if row.brenk_count > 0:
                violations.append(f"Brenk = {row.brenk_count}")

            row.violations = violations
            row.passes_all = len(violations) == 0

        # Векторизованное взвешенное среднее геометрическое в log-пространстве
        safe_d = np.clip(d_matrix, 1e-9, 1.0)
        log_scores = np.sum(np.log(safe_d) * w_vec, axis=1) / w_sum
        scores = np.where(veto_mask > 0.0, np.exp(log_scores), 0.0)

        for i, row in enumerate(rows):
            val = float(scores[i])
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            row.mpo_score = round(max(0.0, min(1.0, val)), 4)

        return rows

    def set_pinned(self, candidate_id: str, pinned: bool) -> None:
        """Фиксирует или снимает фиксацию кандидата."""
        if pinned:
            self.pinned_ids.add(candidate_id)
            self.excluded_ids.discard(candidate_id)
        else:
            self.pinned_ids.discard(candidate_id)
        if candidate_id in self.rows_by_id:
            self.rows_by_id[candidate_id].pinned = pinned
            if pinned:
                self.rows_by_id[candidate_id].excluded = False

    def set_excluded(self, candidate_id: str, excluded: bool) -> None:
        """Исключает или восстанавливает кандидата."""
        if excluded:
            self.excluded_ids.add(candidate_id)
            self.pinned_ids.discard(candidate_id)
        else:
            self.excluded_ids.discard(candidate_id)
        if candidate_id in self.rows_by_id:
            self.rows_by_id[candidate_id].excluded = excluded
            if excluded:
                self.rows_by_id[candidate_id].pinned = False

    def get_all_grid_rows(self) -> list[dict[str, Any]]:
        """Возвращает все строки таблицы в порядке сортировки по MPO."""
        ordered = sorted(
            (self.rows_by_id[cid] for cid in self.row_order if cid in self.rows_by_id),
            key=lambda r: (not r.excluded, r.pinned, r.passes_all, r.mpo_score),
            reverse=True,
        )
        return [r.to_aggrid_row(self.session_id) for r in ordered]

    def pop_new_grid_rows(self) -> list[dict[str, Any]]:
        """Возвращает только ещё не отправленные в `ui.aggrid` строки для `applyTransaction`."""
        new_rows: list[dict[str, Any]] = []
        for cid in self.row_order:
            if cid not in self.emitted_to_grid_ids and cid in self.rows_by_id:
                self.emitted_to_grid_ids.add(cid)
                new_rows.append(self.rows_by_id[cid].to_aggrid_row(self.session_id))
        return new_rows

    def reset_grid_tracking(self) -> None:
        """Сбрасывает трекер отправленных строк при переподключении клиента (F5)."""
        self.emitted_to_grid_ids.clear()

    def clear_candidates(self) -> None:
        """Очищает буфер кандидатов при перезапуске ветви."""
        self.candidates_by_id.clear()
        self.rows_by_id.clear()
        self.row_order.clear()
        self.emitted_to_grid_ids.clear()
        self.lead_reference = None
        self.lead_pca_coords = None
        self.selected_candidate_id = None
