"""Модуль прогнозирования профилей фармакокинетики и безопасности ADMET (Слой 1, без участия LLM).

Реализует 3-уровневый каскад прогнозирования ADMET (Разделы 1.2, 3.7 и 6.1.4 ARCHITECTURE.md):
- Уровень 1 (`admet_ai_dmpnn`): нейросетевые модели D-MPNN пакета `admet-ai` (если установлен).
- Уровень 2 (`catboost_local_qsar`): локальные модели градиентного бустинга CatBoost (`data/models/*.cbm`)
  на векторе из 2056 признаков (2048 бит Morgan FP $r=2$ + 8 дескрипторов RDKit).
- Уровень 3 (`rdkit_analytical_qsar`): детерминированные медико-химические QSAR-модели на дескрипторах
  и структурных маркерах RDKit (Delaney ESOL для растворимости LogS, Egan BOILED-Egg для ГЭБ,
  фармакофорная модель риска блокады калиевых каналов hERG и ингибирования CYP3A4).

Каждый объект `PredADMETMetrics` содержит явную маркировку источника предсказания `model_source`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from pishovichok.config import settings
from pishovichok.state.schemas import CalcRDKitMetrics, PredADMETMetrics

_MORGAN_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# SMARTS основных алифатических аминов (протонируемых при физиологическом pH 7.4 — ключевой фармакофор hERG)
_BASIC_ALIPHATIC_AMINE_SMARTS = Chem.MolFromSmarts(
    "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(Nc);!$(n)]"
)
# SMARTS гетероароматического не-затрудненного азота (координация с гемом CYP3A4)
_CYP_HEME_NITROGEN_SMARTS = Chem.MolFromSmarts("[nX2;r5,r6]")


def _sigmoid(x: float) -> float:
    """Численно устойчивая логистическая функция."""
    clamped = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-clamped))


def build_catboost_feature_vector(mol: Chem.Mol) -> np.ndarray:
    """Формирует вектор из ровно 2056 признаков для локальных моделей CatBoost (Раздел 3.7 ARCHITECTURE.md):
    - 2048 бит отпечатка Моргана (`radius=2, fpSize=2048`);
    - 8 физико-химических дескрипторов RDKit (`MW, cLogP, TPSA, HBD, HBA, RotBonds, Fsp3, QED`).
    """
    fp_arr = _MORGAN_FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
    desc_arr = np.array(
        [
            float(Descriptors.ExactMolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(QED.qed(mol)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([fp_arr, desc_arr], axis=0)


def classify_solubility(
    log_s: float,
) -> Literal["high", "moderate", "poor", "insoluble"]:
    """Классифицирует молярную водную растворимость по шкале LogS (моль/л)."""
    if log_s >= -2.0:
        return "high"
    if log_s >= -4.0:
        return "moderate"
    if log_s >= -6.0:
        return "poor"
    return "insoluble"


def classify_herg_risk(herg_prob: float) -> Literal["low", "moderate", "high"]:
    """Классифицирует вероятность блокады каналов hERG (<0.30 low, 0.30-0.60 moderate, >0.60 high)."""
    if herg_prob < 0.30:
        return "low"
    if herg_prob <= 0.60:
        return "moderate"
    return "high"


def determine_prediction_confidence(
    support_pairs_count: int,
    assay_homogeneous: bool,
    generation_origin: str,
) -> Literal["high", "medium", "low"]:
    """Определяет уровень уверенности в прогнозе активности на основе статистики пар MMP и AD."""
    if (
        generation_origin == "mmp_smirks_rule"
        and assay_homogeneous
        and support_pairs_count >= 2
    ):
        return "high"
    if (
        generation_origin == "mmp_smirks_rule" and support_pairs_count >= 1
    ) or generation_origin == "crem_fragment_replacement":
        return "medium"
    return "low"


def _calculate_analytical_qsar_admet(
    mol: Chem.Mol,
    rdkit_metrics: CalcRDKitMetrics,
    parent_exp_pic50: float,
    historical_mean_delta_pic50: float,
    support_pairs_count: int,
    assay_homogeneous: bool,
    generation_origin: str,
) -> PredADMETMetrics:
    """Уровень 3 каскада (`rdkit_analytical_qsar`): аналитические QSAR-модели на дескрипторах RDKit."""
    mw = rdkit_metrics.mw
    clogp = rdkit_metrics.clogp
    tpsa = rdkit_metrics.tpsa
    rot_bonds = rdkit_metrics.rotatable_bonds

    heavy_atoms = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_proportion = aromatic_atoms / float(heavy_atoms)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # 1. Водная растворимость по уравнению Delaney ESOL (J. Chem. Inf. Comput. Sci. 2004, 44, 1000-1005)
    log_s_aq = round(
        0.16
        - 0.63 * clogp
        - 0.0062 * mw
        + 0.066 * rot_bonds
        - 0.74 * aromatic_proportion,
        4,
    )
    sol_class = classify_solubility(log_s_aq)

    # 2. Проницаемость ГЭБ (Egan BOILED-Egg / Clark logBB): оптимум при TPSA < 78 Å² и 1.5 <= cLogP <= 4.0
    tpsa_term = (75.0 - tpsa) / 18.0
    logp_penalty = abs(clogp - 2.6) * 0.45
    bbb_prob = round(_sigmoid(tpsa_term - logp_penalty), 4)

    # 3. Кардиотоксичность hERG (модель Aronov / Waring: липофильность + основные алифатические амины - TPSA)
    basic_amine_matches = (
        len(mol.GetSubstructMatches(_BASIC_ALIPHATIC_AMINE_SMARTS))
        if _BASIC_ALIPHATIC_AMINE_SMARTS is not None
        else 0
    )
    z_herg = (
        -0.88
        + 0.56 * (clogp - 3.0)
        + 0.45 * min(2, basic_amine_matches)
        + 0.12 * max(0, aromatic_rings - 2)
        - 0.032 * max(0.0, tpsa - 65.0)
    )
    herg_prob = round(_sigmoid(z_herg), 4)
    herg_level = classify_herg_risk(herg_prob)

    # 4. Вероятность ингибирования цитохрома CYP3A4 (зависит от MW, cLogP и гетероароматических азотов)
    heme_n_matches = (
        len(mol.GetSubstructMatches(_CYP_HEME_NITROGEN_SMARTS))
        if _CYP_HEME_NITROGEN_SMARTS is not None
        else 0
    )
    z_cyp3a4 = (
        -1.15
        + 0.38 * (clogp - 2.8)
        + 0.004 * (mw - 380.0)
        + 0.14 * min(3, heme_n_matches)
        - 0.015 * max(0.0, tpsa - 75.0)
    )
    cyp3a4_prob = round(_sigmoid(z_cyp3a4), 4)

    # 5. Ожидаемая активность pIC50 кандидата (аддитивный сдвиг MMP + мягкая поправка на экстремумы LogP/MW)
    lipophilicity_penalty = 0.15 * max(0.0, clogp - 5.0) + 0.10 * max(0.0, 1.0 - clogp)
    predicted_pic50 = round(
        parent_exp_pic50 + historical_mean_delta_pic50 - lipophilicity_penalty,
        4,
    )
    confidence = determine_prediction_confidence(
        support_pairs_count=support_pairs_count,
        assay_homogeneous=assay_homogeneous,
        generation_origin=generation_origin,
    )

    return PredADMETMetrics(
        model_source="rdkit_analytical_qsar",
        log_s_aq=log_s_aq,
        solubility_class=sol_class,
        bbb_permeability_prob=bbb_prob,
        herg_liability_prob=herg_prob,
        herg_risk_level=herg_level,
        cyp3a4_inhibition_prob=cyp3a4_prob,
        predicted_pic50=predicted_pic50,
        prediction_confidence=confidence,
    )


def _try_predict_catboost_local(
    mol: Chem.Mol,
    rdkit_metrics: CalcRDKitMetrics,
    parent_exp_pic50: float,
    historical_mean_delta_pic50: float,
    support_pairs_count: int,
    assay_homogeneous: bool,
    generation_origin: str,
    models_dir: Path,
) -> PredADMETMetrics | None:
    """Уровень 2 каскада (`catboost_local_qsar`): загрузка локальных весов `.cbm` при их наличии."""
    herg_path = models_dir / "herg_catboost.cbm"
    bbb_path = models_dir / "bbb_catboost.cbm"
    cyp_path = models_dir / "cyp3a4_catboost.cbm"
    logs_path = models_dir / "logs_catboost.cbm"

    if not (
        herg_path.is_file()
        and bbb_path.is_file()
        and cyp_path.is_file()
        and logs_path.is_file()
    ):
        return None

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        features_2d = build_catboost_feature_vector(mol).reshape(1, -1)

        herg_model = CatBoostClassifier()
        herg_model.load_model(str(herg_path), format="cbm")
        herg_prob = round(float(herg_model.predict_proba(features_2d)[0, 1]), 4)

        bbb_model = CatBoostClassifier()
        bbb_model.load_model(str(bbb_path), format="cbm")
        bbb_prob = round(float(bbb_model.predict_proba(features_2d)[0, 1]), 4)

        cyp_model = CatBoostClassifier()
        cyp_model.load_model(str(cyp_path), format="cbm")
        cyp_prob = round(float(cyp_model.predict_proba(features_2d)[0, 1]), 4)

        logs_model = CatBoostRegressor()
        logs_model.load_model(str(logs_path), format="cbm")
        log_s_val = round(float(logs_model.predict(features_2d)[0]), 4)

        lipophilicity_penalty = 0.15 * max(0.0, rdkit_metrics.clogp - 5.0)
        predicted_pic50 = round(
            parent_exp_pic50 + historical_mean_delta_pic50 - lipophilicity_penalty,
            4,
        )

        return PredADMETMetrics(
            model_source="catboost_local_qsar",
            log_s_aq=log_s_val,
            solubility_class=classify_solubility(log_s_val),
            bbb_permeability_prob=bbb_prob,
            herg_liability_prob=herg_prob,
            herg_risk_level=classify_herg_risk(herg_prob),
            cyp3a4_inhibition_prob=cyp_prob,
            predicted_pic50=predicted_pic50,
            prediction_confidence=determine_prediction_confidence(
                support_pairs_count=support_pairs_count,
                assay_homogeneous=assay_homogeneous,
                generation_origin=generation_origin,
            ),
        )
    except Exception:  # noqa: BLE001
        return None


def _try_predict_admet_ai(
    smiles: str,
    rdkit_metrics: CalcRDKitMetrics,
    parent_exp_pic50: float,
    historical_mean_delta_pic50: float,
    support_pairs_count: int,
    assay_homogeneous: bool,
    generation_origin: str,
) -> PredADMETMetrics | None:
    """Уровень 1 каскада (`admet_ai_dmpnn`): вызов пакета `admet_ai` при его наличии в окружении."""
    try:
        from admet_ai import ADMETModel  # type: ignore[import-not-found]

        model = ADMETModel()
        preds_df = model.predict(smiles=[smiles])
        row = preds_df.iloc[0].to_dict()

        log_s_val = round(float(row.get("Solubility_AqSolDB", -4.0)), 4)
        bbb_prob = round(max(0.0, min(1.0, float(row.get("BBB_Martins", 0.3)))), 4)
        herg_prob = round(max(0.0, min(1.0, float(row.get("hERG", 0.3)))), 4)
        cyp_prob = round(max(0.0, min(1.0, float(row.get("CYP3A4_Veith", 0.3)))), 4)
        predicted_pic50 = round(parent_exp_pic50 + historical_mean_delta_pic50, 4)

        return PredADMETMetrics(
            model_source="admet_ai_dmpnn",
            log_s_aq=log_s_val,
            solubility_class=classify_solubility(log_s_val),
            bbb_permeability_prob=bbb_prob,
            herg_liability_prob=herg_prob,
            herg_risk_level=classify_herg_risk(herg_prob),
            cyp3a4_inhibition_prob=cyp_prob,
            predicted_pic50=predicted_pic50,
            prediction_confidence=determine_prediction_confidence(
                support_pairs_count=support_pairs_count,
                assay_homogeneous=assay_homogeneous,
                generation_origin=generation_origin,
            ),
        )
    except Exception:  # noqa: BLE001
        return None


def predict_admet_metrics(
    mol_or_smiles: Chem.Mol | str,
    rdkit_metrics: CalcRDKitMetrics,
    parent_exp_pic50: float,
    historical_mean_delta_pic50: float = 0.0,
    support_pairs_count: int = 1,
    assay_homogeneous: bool = True,
    generation_origin: str = "mmp_smirks_rule",
    *,
    preferred_backend: Literal[
        "auto",
        "admet_ai_dmpnn",
        "catboost_local_qsar",
        "rdkit_analytical_qsar",
    ]
    | None = None,
    models_dir: Path | str | None = None,
) -> PredADMETMetrics:
    """Выполняет прогноз ADMET-профиля кандидата с автоматическим каскадным фолбэком."""
    mol = (
        Chem.MolFromSmiles(mol_or_smiles)
        if isinstance(mol_or_smiles, str)
        else mol_or_smiles
    )
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Cannot predict ADMET for invalid molecule: {mol_or_smiles}")

    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    backend = preferred_backend or settings.preferred_admet_backend
    resolved_models_dir = (
        Path(models_dir) if models_dir is not None else settings.models_dir
    )

    # Уровень 1: admet-ai (D-MPNN)
    if backend in ("auto", "admet_ai_dmpnn"):
        res_l1 = _try_predict_admet_ai(
            smiles=smiles,
            rdkit_metrics=rdkit_metrics,
            parent_exp_pic50=parent_exp_pic50,
            historical_mean_delta_pic50=historical_mean_delta_pic50,
            support_pairs_count=support_pairs_count,
            assay_homogeneous=assay_homogeneous,
            generation_origin=generation_origin,
        )
        if res_l1 is not None:
            return res_l1

    # Уровень 2: Локальные модели CatBoost (.cbm)
    if backend in ("auto", "admet_ai_dmpnn", "catboost_local_qsar"):
        res_l2 = _try_predict_catboost_local(
            mol=mol,
            rdkit_metrics=rdkit_metrics,
            parent_exp_pic50=parent_exp_pic50,
            historical_mean_delta_pic50=historical_mean_delta_pic50,
            support_pairs_count=support_pairs_count,
            assay_homogeneous=assay_homogeneous,
            generation_origin=generation_origin,
            models_dir=resolved_models_dir,
        )
        if res_l2 is not None:
            return res_l2

    # Уровень 3: Гарантированный аналитический QSAR на дескрипторах RDKit
    return _calculate_analytical_qsar_admet(
        mol=mol,
        rdkit_metrics=rdkit_metrics,
        parent_exp_pic50=parent_exp_pic50,
        historical_mean_delta_pic50=historical_mean_delta_pic50,
        support_pairs_count=support_pairs_count,
        assay_homogeneous=assay_homogeneous,
        generation_origin=generation_origin,
    )
