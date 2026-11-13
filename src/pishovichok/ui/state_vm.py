"""View-Model над `SARGraphState` для реактивного обновления MPO, таблицы и графиков (Слой 4).

Обеспечивает:
1. Хранение плоского буфера кандидатов (без SVG и 3D-координат) для `ui.aggrid` и `ui.echart`.
2. Мгновенный векторный пересчёт интегрального индекса Харрингтона и статусов ограничений
   при изменении весов и порогов на ползунках без перезапуска `agents/workflow.py`.
3. Инкрементальную выдачу новых строк для `applyTransaction({'add': [...]})`.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from pishovichok.config import DEMO_RECEPTOR_PRESETS, RECEPTORS_DIR
from pishovichok.core.evaluation import resolve_generation_method_label
from pishovichok.core.scoring import compute_harrington_mpo_breakdown
from pishovichok.state.schemas import (
    CandidateMolecule,
    ParamSourceLiteral,
    QueryConstraints,
    ToolConfig,
)

ApplyModeLiteral = Literal["instant_mpo", "evaluation", "design", "sar", "research"]

PARAM_SOURCE_BADGE_META: dict[ParamSourceLiteral, tuple[str, str]] = {
    "default": ("", "slate"),
    "nl_query": ("Из запроса", "sky"),
    "ui_manual": ("Изменено", "blue-7"),
    "critic_loop": ("Авто-фильтр верификации", "amber"),
}

DOCKING_STRONG_BENCHMARK_KCAL: float = -7.0
AD_TANIMOTO_THRESHOLD: float = 0.5


def resolve_receptor_display_name(
    pdb_id: str | None,
    receptor_name: str | None = None,
    receptors_dir: Path | str | None = None,
) -> str | None:
    """Возвращает человекочитаемое название рецептора из DockingResult или data/receptors/*.json."""
    if receptor_name and str(receptor_name).strip():
        return str(receptor_name).strip()
    pdb_clean = (pdb_id or "").strip().upper()
    if not pdb_clean or pdb_clean in ("-", "NONE", "N/A"):
        return None

    root = Path(receptors_dir) if receptors_dir is not None else RECEPTORS_DIR
    if root.is_dir():
        for json_file in sorted(root.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(data.get("pdb_id", "")).strip().upper() == pdb_clean:
                tname = data.get("target_name") or data.get("receptor_name") or data.get("target_id")
                if tname and str(tname).strip():
                    return str(tname).strip()

    for target_id, preset in DEMO_RECEPTOR_PRESETS.items():
        if str(preset.get("pdb_id", "")).strip().upper() == pdb_clean:
            tname = preset.get("target_name") or target_id
            return str(tname)
    return None

_RESEARCH_STAGE_KEYS: frozenset[str] = frozenset(
    {
        "allowed_bao_formats",
        "standard_types",
        "relation_mode",
        "pchembl_min_cutoff",
        "duplicate_agg_method",
        "max_duplicate_std",
        "strip_salts",
        "neutralize_charges",
        "canonicalize_tautomers",
        "keep_chirality",
        "scaffold_mode",
        "chembl_api_limit",
        "min_neighborhood_rows_fallback",
        "top_scaffolds_limit",
        "neighborhood_tanimoto",
        "strict_assay_homogeneity",
        "max_refilter_iterations",
    }
)

_SAR_STAGE_KEYS: frozenset[str] = frozenset(
    {
        "mmpa_max_cuts",
        "mmpa_max_cut_bonds",
        "mmpa_min_core_heavy_atoms",
        "min_support_pairs",
        "min_mean_delta_pic50",
        "max_rgroup_heavy_atoms",
        "fp_type",
        "fp_radius",
        "fp_nbits",
        "similarity_metric",
        "cliff_similarity_cutoff",
        "cliff_delta_pic50_cutoff",
        "sali_epsilon",
        "min_rules_before_enrichment",
        "background_target_id",
        "prioritize_positive_rules",
    }
)

_DESIGN_STAGE_KEYS: frozenset[str] = frozenset(
    {
        "generation_engine",
        "crem_mode",
        "crem_context_radius",
        "crem_min_rel_atoms",
        "crem_max_rel_atoms",
        "crem_max_replacements",
        "crem_max_fragment_heavy_atoms",
        "freeze_murcko_core",
        "protected_smarts",
        "modifiable_atom_indices",
        "max_candidates_pool",
        "max_parent_seeds",
        "use_bioisostere_catalog",
        "bioisostere_fallback_threshold",
        "ga_population_size",
        "ga_generations",
        "enable_crem_grow",
        "max_iterations",
        "strict_property_thresholds",
        "critic_fallback_min_tpsa",
        "critic_logp_step_reduction",
    }
)

_EVALUATION_STAGE_KEYS: frozenset[str] = frozenset(
    {
        "active_filter_catalogs",
        "brenk_ignore_long_chain",
        "custom_ban_smarts",
        "extra_rdkit_descriptors",
        "brics_min_fragment_size",
        "brics_only_leaf_nodes",
        "admet_backend",
        "lipinski_mw_cutoff",
        "lipinski_logp_cutoff",
        "lipinski_hbd_cutoff",
        "lipinski_hba_cutoff",
        "herg_low_cutoff",
        "herg_high_cutoff",
        "logs_soluble_cutoff",
        "logs_moderate_cutoff",
        "logs_poor_cutoff",
        "top_k_output",
        "docking_mode",
        "docking_exhaustiveness",
        "docking_top_k",
        "custom_pdb_id",
        "custom_box_center",
    }
)


def classify_tool_config_change_impact(changed_keys: Iterable[str]) -> ApplyModeLiteral:
    """Определяет минимально достаточный режим пересчёта по набору изменённых полей `ToolConfig`."""
    key_set = set(changed_keys)
    if key_set & _RESEARCH_STAGE_KEYS:
        return "research"
    if key_set & _SAR_STAGE_KEYS:
        return "sar"
    if key_set & _DESIGN_STAGE_KEYS:
        return "design"
    if key_set & _EVALUATION_STAGE_KEYS:
        return "evaluation"
    return "instant_mpo"


class MPOWeightsAndThresholds(BaseModel):
    """Интерактивные веса и пороги MPO-панели (управляются ползунками без перезапуска графа)."""

    w_activity: float = Field(default=1.5, ge=0.0, le=5.0)
    w_logp: float = Field(default=1.0, ge=0.0, le=5.0)
    w_sa: float = Field(default=1.0, ge=0.0, le=5.0)
    w_qed: float = Field(default=1.0, ge=0.0, le=5.0)
    w_tpsa: float = Field(default=0.8, ge=0.0, le=5.0)
    w_logs: float = Field(default=0.8, ge=0.0, le=5.0)
    w_herg: float = Field(default=1.2, ge=0.0, le=5.0)

    cns_profile_mode: Literal["peripheral_low_bbb", "cns_high_bbb", "neutral"] = (
        "peripheral_low_bbb"
    )
    alert_penalty_mode: Literal["veto", "soft_penalty", "ignore"] = "veto"
    alert_soft_multiplier: float = Field(default=0.5, ge=0.0, le=1.0)

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

    def to_tool_config_overrides(self) -> dict[str, Any]:
        """Преобразует UI-настройки MPO в словарь переопределений `ToolConfig`."""
        return {
            "mpo_weights": {
                "pic50": self.w_activity,
                "logp": self.w_logp,
                "sascore": self.w_sa,
                "qed": self.w_qed,
                "tpsa": self.w_tpsa,
                "logs": self.w_logs,
                "herg": self.w_herg,
            },
            "cns_profile_mode": self.cns_profile_mode,
            "alert_penalty_mode": self.alert_penalty_mode,
            "alert_soft_multiplier": self.alert_soft_multiplier,
        }


class CandidateRowVM(BaseModel):
    """Лёгкая строковая проекция кандидата для `ui.aggrid` и `mpo_canvas` (без SVG/3D)."""

    candidate_id: str
    canonical_smiles: str
    scaffold_smiles: str
    parent_chembl_id: str
    parent_exp_pic50: float
    parent_bao_format: str = "BAO_0000357"
    parent_assay_type: str = "B"
    parent_source_dataset: str = "ChEMBL"
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
    log_s_aq: float = -3.0
    bbb_permeability_prob: float = 0.5
    pains_count: int
    brenk_count: int
    extra_alerts_count: int = 0
    alert_penalty_mode: Literal["veto", "soft_penalty", "ignore"] = "veto"
    rule_id: str
    applied_smirks: str
    generation_origin: str
    tanimoto_to_parent: float | None = None
    generation_method: str = ""
    mpo_score: float
    passes_all: bool
    violations: list[str] = Field(default_factory=list)
    desirability_d1_activity: float = 0.0
    desirability_d2_clogp: float = 0.0
    desirability_d3_sa: float = 0.0
    desirability_d4_qed: float = 0.0
    desirability_d5_herg: float = 0.0
    desirability_d6_tpsa: float = 0.0
    desirability_d7_logs: float = 0.0
    docking_status: str | None = None
    docking_dg_kcal: float | None = None
    docking_pdb_id: str | None = None
    docking_receptor_name: str | None = None
    parent_docking_score_kcal: float | None = None
    docking_source: str | None = None
    docking_backend: str | None = None
    docking_interactions: dict[str, list[str]] = Field(default_factory=dict)
    docking_receptor_pdb_path: str | None = None
    docking_pose_sdf_block: str | None = None
    docking_key_residues: list[str] = Field(default_factory=list)
    docking_message: str | None = None
    pinned: bool = False
    excluded: bool = False
    pca_x: float = 0.0
    pca_y: float = 0.0

    @property
    def status_label(self) -> str:
        has_alert = (self.pains_count + self.brenk_count + self.extra_alerts_count) > 0
        if has_alert and self.alert_penalty_mode == "veto":
            return "Вето"
        if self.passes_all:
            return "Соответствует"
        return "Компромисс"

    @property
    def alerts_label(self) -> str:
        total = self.pains_count + self.brenk_count + self.extra_alerts_count
        if total == 0:
            return "Чисто"
        parts: list[str] = []
        if self.pains_count > 0:
            parts.append(f"PAINS:{self.pains_count}")
        if self.brenk_count > 0:
            parts.append(f"Brenk:{self.brenk_count}")
        if self.extra_alerts_count > 0:
            parts.append(f"Доп:{self.extra_alerts_count}")
        if self.alert_penalty_mode == "veto":
            return f"Вето ({' '.join(parts)})"
        return " ".join(parts)

    @property
    def docking_dg_display(self) -> str:
        if self.docking_status == "SUCCESS" and self.docking_dg_kcal is not None:
            return f"{self.docking_dg_kcal:.2f} ккал/моль"
        if self.docking_status == "SKIPPED":
            return "Пропущен"
        if self.docking_status == "FAILED":
            return "Ошибка"
        return "Не выполнен"

    @property
    def docking_interactions_summary(self) -> str:
        if not self.docking_interactions:
            return "-"
        parts = [
            f"{k}: {', '.join(v[:3])}"
            for k, v in self.docking_interactions.items()
            if v
        ]
        return "; ".join(parts[:3]) if parts else "-"

    @property
    def method_rule_label(self) -> str:
        if self.rule_id == "GEN-GA" or self.generation_origin == "graph_ga":
            return "GraphGA (BRICS)"
        if self.rule_id == "CREM-GROW":
            return "CReM Grow"
        if self.rule_id == "CREM-MUT":
            return "CReM Mutate"
        origin_map = {
            "mmp_smirks_rule": "MMP",
            "crem_fragment_replacement": "CReM",
            "bioisostere_catalog": "Биоизостер",
            "graph_ga": "GraphGA",
        }
        origin_short = origin_map.get(self.generation_origin, self.generation_origin)
        return f"{origin_short} · {self.rule_id}"

    @property
    def resolved_generation_method(self) -> str:
        return resolve_generation_method_label(
            generation_origin=self.generation_origin,
            explicit_method=self.generation_method,
        )

    @property
    def ad_status_label(self) -> str:
        if self.tanimoto_to_parent is None:
            return "AD: Н/Д"
        if self.tanimoto_to_parent >= AD_TANIMOTO_THRESHOLD:
            return f"Tc {self.tanimoto_to_parent:.2f} · В домене"
        return f"Tc {self.tanimoto_to_parent:.2f} · Экстраполяция"

    @property
    def docking_delta_kcal(self) -> float | None:
        if (
            self.docking_status == "SUCCESS"
            and self.docking_dg_kcal is not None
            and self.parent_docking_score_kcal is not None
        ):
            return round(self.docking_dg_kcal - self.parent_docking_score_kcal, 2)
        return None

    def to_aggrid_row(self, session_id: str) -> dict[str, Any]:
        """Формирует лёгкий словарь строки для `ui.aggrid` (без SVG и 3D-координат)."""
        svg_url = f"/api/sessions/{session_id}/mol/{self.candidate_id}/svg?variant=diff&width=320&height=170"
        herg_risk_map = {
            "low": "низкий",
            "moderate": "средний",
            "high": "высокий",
        }
        herg_risk_ru = herg_risk_map.get(self.herg_risk_level, self.herg_risk_level)
        mpo_tip = (
            "Соответствует всем порогам"
            if self.passes_all
            else f"Нарушения: {', '.join(self.violations)}"
        )
        gen_method = self.resolved_generation_method
        gen_badge_styles = {
            "MMP SMIRKS": ("#EDE9FE", "#5B21B6", "#DDD6FE"),
            "Изостера CReM": ("#ECFDF5", "#065F46", "#A7F3D0"),
            "Генеративная модель": ("#EFF6FF", "#1E40AF", "#BFDBFE"),
        }
        g_bg, g_fg, g_bd = gen_badge_styles.get(
            gen_method, ("#F1F5F9", "#334155", "#CBD5E1")
        )
        if self.tanimoto_to_parent is None:
            ad_badge_html = (
                '<span style="font-size:10px;padding:1px 5px;border-radius:4px;'
                'background:#F8FAFC;color:#64748B;border:1px solid #E2E8F0;">Tc Н/Д</span>'
            )
        elif self.tanimoto_to_parent >= AD_TANIMOTO_THRESHOLD:
            ad_badge_html = (
                f'<span style="font-size:10px;font-weight:600;padding:1px 5px;border-radius:4px;'
                f'background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0;">'
                f"Tc {self.tanimoto_to_parent:.2f} · В домене</span>"
            )
        else:
            ad_badge_html = (
                f'<span style="font-size:10px;font-weight:600;padding:1px 5px;border-radius:4px;'
                f'background:#FFFBEB;color:#92400E;border:1px solid #FDE68A;">'
                f"Tc {self.tanimoto_to_parent:.2f} · Экстраполяция</span>"
            )
        method_rule_html = (
            '<div style="display:flex;flex-direction:column;justify-content:center;gap:3px;line-height:1.15;padding:4px 0;">'
            f'<span style="display:inline-block;width:fit-content;font-size:10px;font-weight:600;padding:1px 6px;'
            f'border-radius:4px;background:{g_bg};color:{g_fg};border:1px solid {g_bd};">'
            f"{html.escape(gen_method or self.generation_origin)}</span>"
            f'<span style="font-size:10px;color:#475569;font-family:ui-monospace,monospace;">'
            f"{html.escape(self.rule_id)}</span>"
            f"{ad_badge_html}"
            "</div>"
        )

        rec_name = resolve_receptor_display_name(
            self.docking_pdb_id, self.docking_receptor_name
        )
        dock_delta = self.docking_delta_kcal
        if self.docking_status == "SUCCESS" and self.docking_dg_kcal is not None:
            is_strong = self.docking_dg_kcal <= DOCKING_STRONG_BENCHMARK_KCAL
            q_bg, q_fg, q_bd, q_lbl = (
                ("#ECFDF5", "#065F46", "#A7F3D0", f"≤ {DOCKING_STRONG_BENCHMARK_KCAL:.1f} · сильное")
                if is_strong
                else ("#FFFBEB", "#92400E", "#FDE68A", f"> {DOCKING_STRONG_BENCHMARK_KCAL:.1f} · слабее порога")
            )
            delta_html = (
                f'<span style="font-size:10px;color:#475569;font-family:ui-monospace,monospace;">'
                f"Δ к родителю: {dock_delta:+.2f} ккал/моль</span>"
                if dock_delta is not None
                else ""
            )
            docking_dg_html = (
                '<div style="display:flex;flex-direction:column;justify-content:center;gap:3px;line-height:1.15;padding:4px 0;">'
                f'<span style="font-size:11px;font-weight:600;color:#0F172A;">{self.docking_dg_kcal:.2f} ккал/моль</span>'
                f'<span style="display:inline-block;width:fit-content;font-size:10px;font-weight:600;padding:1px 5px;'
                f'border-radius:4px;background:{q_bg};color:{q_fg};border:1px solid {q_bd};">{q_lbl}</span>'
                f"{delta_html}"
                "</div>"
            )
        else:
            docking_dg_html = (
                f'<span style="font-size:11px;color:#64748B;">{html.escape(self.docking_dg_display)}</span>'
            )

        if self.docking_pdb_id:
            short_rec = (rec_name or "").split(" (", 1)[0]
            rec_line = (
                f'<span style="font-size:10px;font-weight:600;color:#0F172A;white-space:normal;line-height:1.15;">'
                f"{html.escape(short_rec)}</span>"
                if short_rec
                else ""
            )
            docking_receptor_html = (
                '<div style="display:flex;flex-direction:column;justify-content:center;gap:2px;line-height:1.15;padding:4px 0;">'
                f"{rec_line}"
                f'<span style="font-size:10px;font-family:ui-monospace,monospace;color:#1E40AF;'
                f'background:#EFF6FF;border:1px solid #BFDBFE;border-radius:4px;padding:1px 5px;width:fit-content;">'
                f"PDB: {html.escape(self.docking_pdb_id)}</span>"
                "</div>"
            )
        else:
            docking_receptor_html = '<span style="font-size:11px;color:#94A3B8;">-</span>'
        return {
            "candidate_id": self.candidate_id,
            "preview_html": (
                f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:96px;gap:2px;" '
                f'title="{self.candidate_id}">'
                f'<img src="{svg_url}" loading="lazy" alt="{self.candidate_id}" '
                f'style="height:74px;width:156px;object-fit:contain;display:block;background:#FFFFFF;" />'
                f'<span style="font-size:10px;line-height:1.1;color:#475569;font-family:ui-monospace,monospace;white-space:nowrap;">'
                f"{self.candidate_id}</span></div>"
            ),
            "parent_chembl_id": self.parent_chembl_id,
            "parent_exp_pic50": round(self.parent_exp_pic50, 2),
            "parent_bao_tooltip": (
                f"Протокол BAO: {self.parent_bao_format} | Тип теста: {self.parent_assay_type} | Источник: {self.parent_source_dataset}"
            ),
            "predicted_pic50": round(self.predicted_pic50, 2),
            "delta_activity": round(self.delta_activity, 2),
            "pred_pic50_display": f"{self.predicted_pic50:.2f} ({self.delta_activity:+.2f})",
            "clogp": round(self.clogp, 2),
            "delta_clogp": round(self.delta_clogp, 2),
            "clogp_display": f"{self.clogp:.2f} ({self.delta_clogp:+.2f})",
            "tpsa": round(self.tpsa, 1),
            "delta_tpsa": round(self.delta_tpsa, 1),
            "sa_score": round(self.sa_score, 2),
            "qed": round(self.qed, 2),
            "alerts_display": self.alerts_label,
            "herg_prob": round(self.herg_prob, 2),
            "herg_display": f"{self.herg_prob:.2f} ({herg_risk_ru})",
            "log_s_aq": round(self.log_s_aq, 2),
            "delta_herg": round(self.delta_herg, 2),
            "critic_status": self.status_label,
            "mpo_score": round(self.mpo_score, 3),
            "mpo_tooltip": mpo_tip,
            "passes_all": self.passes_all,
            "pinned": self.pinned,
            "excluded": self.excluded,
            "rule_id": self.rule_id,
            "generation_method": gen_method,
            "tanimoto_to_parent": round(self.tanimoto_to_parent, 4)
            if self.tanimoto_to_parent is not None
            else None,
            "ad_status_display": self.ad_status_label,
            "method_rule_display": self.method_rule_label,
            "method_rule_html": method_rule_html,
            "docking_status": self.docking_status or "PENDING",
            "docking_dg_kcal": round(self.docking_dg_kcal, 2)
            if self.docking_dg_kcal is not None
            else None,
            "parent_docking_score_kcal": round(self.parent_docking_score_kcal, 2)
            if self.parent_docking_score_kcal is not None
            else None,
            "docking_delta_kcal": dock_delta,
            "docking_dg_display": self.docking_dg_display,
            "docking_dg_html": docking_dg_html,
            "docking_pdb_id": self.docking_pdb_id or "-",
            "docking_receptor_name": rec_name or "",
            "docking_receptor_html": docking_receptor_html,
            "docking_backend": self.docking_backend or "-",
            "docking_interactions_summary": self.docking_interactions_summary,
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
        self.tool_config: ToolConfig = ToolConfig()
        self.min_tpsa: float | None = None
        self.candidates_by_id: dict[str, CandidateMolecule] = {}
        self.rows_by_id: dict[str, CandidateRowVM] = {}
        self.row_order: list[str] = []
        self.emitted_to_grid_ids: set[str] = set()
        self.reference_points: list[ReferencePointVM] = []
        self.mmp_pairs_preview: list[dict[str, Any]] = []
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
        self.min_tpsa = constraints.min_tpsa

    def sync_tool_config(self, tool_config: ToolConfig | None) -> None:
        """Синхронизирует веса MPO и полную конфигурацию ядра из `ToolConfig`."""
        if tool_config is None:
            return
        self.tool_config = tool_config
        mw = tool_config.mpo_weights
        self.weights.w_activity = mw.get("pic50", self.weights.w_activity)
        self.weights.w_logp = mw.get("logp", self.weights.w_logp)
        self.weights.w_sa = mw.get("sascore", self.weights.w_sa)
        self.weights.w_qed = mw.get("qed", self.weights.w_qed)
        self.weights.w_tpsa = mw.get("tpsa", self.weights.w_tpsa)
        self.weights.w_logs = mw.get("logs", self.weights.w_logs)
        self.weights.w_herg = mw.get("herg", self.weights.w_herg)
        self.weights.cns_profile_mode = tool_config.cns_profile_mode
        self.weights.alert_penalty_mode = tool_config.alert_penalty_mode
        self.weights.alert_soft_multiplier = tool_config.alert_soft_multiplier

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
                parent_bao_format=cand.parent_bao_format,
                parent_assay_type=cand.parent_assay_type,
                parent_source_dataset=cand.parent_source_dataset,
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
                log_s_aq=cand.pred_admet.log_s_aq,
                bbb_permeability_prob=cand.pred_admet.bbb_permeability_prob,
                pains_count=cand.calc_rdkit.pains_alerts_count,
                brenk_count=cand.calc_rdkit.brenk_alerts_count,
                extra_alerts_count=cand.calc_rdkit.extra_alerts_count,
                alert_penalty_mode=self.weights.alert_penalty_mode,
                rule_id=cand.hypothesis_rationale.rule_id,
                applied_smirks=cand.hypothesis_rationale.applied_smirks,
                generation_origin=cand.hypothesis_rationale.generation_origin,
                tanimoto_to_parent=cand.calc_rdkit.tanimoto_to_parent,
                generation_method=cand.calc_rdkit.generation_method
                or resolve_generation_method_label(
                    cand.hypothesis_rationale.generation_origin
                ),
                mpo_score=cand.mpo_harrington_score,
                passes_all=cand.passes_all_constraints,
                violations=list(cand.constraint_violations),
                docking_status=cand.docking.status if cand.docking else None,
                docking_dg_kcal=cand.docking.docking_score_kcal if cand.docking else None,
                docking_pdb_id=cand.docking.pdb_id if cand.docking else None,
                docking_receptor_name=(
                    resolve_receptor_display_name(
                        cand.docking.pdb_id,
                        cand.docking.receptor_name,
                    )
                    if cand.docking
                    else None
                ),
                parent_docking_score_kcal=(
                    cand.docking.parent_docking_score_kcal if cand.docking else None
                ),
                docking_source=cand.docking.source if cand.docking else None,
                docking_backend=cand.docking.backend if cand.docking else None,
                docking_interactions=dict(cand.docking.interactions) if cand.docking else {},
                docking_receptor_pdb_path=cand.docking.receptor_pdb_path if cand.docking else None,
                docking_pose_sdf_block=cand.docking.pose_sdf_block if cand.docking else None,
                docking_key_residues=list(cand.docking.key_residues) if cand.docking else [],
                docking_message=cand.docking.message if cand.docking else None,
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
        self,
        weights: MPOWeightsAndThresholds | None = None,
        *,
        tool_config: ToolConfig | None = None,
    ) -> list[CandidateRowVM]:
        """Мгновенно пересчитывает MPO-скоры через `core/scoring.py` без вызова графа агентов."""
        if tool_config is not None:
            self.sync_tool_config(tool_config)
        if weights is not None:
            self.weights = weights

        if not self.row_order:
            return []

        effective_tc, _ = self.tool_config.apply_overrides(
            self.weights.to_tool_config_overrides()
        )
        rows = [self.rows_by_id[cid] for cid in self.row_order if cid in self.rows_by_id]

        for row in rows:
            has_alert = (row.pains_count + row.brenk_count + row.extra_alerts_count) > 0
            score, desirabilities = compute_harrington_mpo_breakdown(
                predicted_pic50=row.predicted_pic50,
                parent_exp_pic50=row.parent_exp_pic50,
                clogp=row.clogp,
                sa_score=row.sa_score,
                qed=row.qed,
                tpsa=row.tpsa,
                herg_liability_prob=row.herg_prob,
                log_s_aq=row.log_s_aq,
                has_alerts=has_alert,
                min_logp=self.weights.min_logp,
                max_logp=self.weights.max_logp,
                max_sa_score=self.weights.max_sa_score,
                max_herg_prob=self.weights.max_herg_prob,
                min_tpsa=self.min_tpsa,
                docking_score_kcal=(
                    row.docking_dg_kcal
                    if row.docking_status == "SUCCESS"
                    else None
                ),
                tool_config=effective_tc,
            )

            row.mpo_score = score
            row.alert_penalty_mode = effective_tc.alert_penalty_mode
            row.desirability_d1_activity = round(desirabilities["pic50"], 4)
            row.desirability_d2_clogp = round(desirabilities["logp"], 4)
            row.desirability_d3_sa = round(desirabilities["sascore"], 4)
            row.desirability_d4_qed = round(desirabilities["qed"], 4)
            row.desirability_d5_herg = round(desirabilities["herg"], 4)
            row.desirability_d6_tpsa = round(desirabilities["tpsa"], 4)
            row.desirability_d7_logs = round(desirabilities["logs"], 4)

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
            if self.min_tpsa is not None and row.tpsa < self.min_tpsa:
                violations.append(f"TPSA = {row.tpsa:.2f} < {self.min_tpsa:.2f}")
            if effective_tc.alert_penalty_mode != "ignore":
                if row.pains_count > 0:
                    violations.append(f"PAINS = {row.pains_count}")
                if row.brenk_count > 0:
                    violations.append(f"Brenk = {row.brenk_count}")
                if row.extra_alerts_count > 0:
                    violations.append(f"ExtraAlerts = {row.extra_alerts_count}")

            row.violations = violations
            row.passes_all = len(violations) == 0

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
