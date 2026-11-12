"""Строгие контракты данных (Pydantic v2) и архитектурное разделение 4 категорий знаний (Слой 2).

Реализует:
1. Категория 1 (Экспериментальный факт): поля `parent_*` и модель `ExperimentalParentFact`.
2. Категория 2 (Детерминированный расчет RDKit): `CalcRDKitMetrics`.
3. Категория 3 (Модельный прогноз ADMET / QSAR): `PredADMETMetrics` с обязательным `model_source`.
4. Категория 4 (Научная гипотеза SAR): `SARHypothesisRationale`.
5. Единый неизменяемый контракт кандидата `CandidateMolecule`.
6. Метаданные толстых указателей `ParquetPointer` и вспомогательные структуры анализа SAR.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Обязательные колонки 4 сессионных Parquet-таблиц согласно Разделу 5.1 ARCHITECTURE.md
PARQUET_TABLE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "raw_dataset": (
        "raw_row_id",
        "molecule_chembl_id",
        "raw_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
        "assay_type",
        "bao_format",
        "target_chembl_id",
        "target_organism",
    ),
    "curated_dataset": (
        "mol_id",
        "canonical_smiles",
        "inchi_key",
        "scaffold_smiles",
        "pchembl_value",
        "pchembl_std",
        "replicate_count",
        "is_censored",
        "assay_type",
        "bao_format",
        "standard_type",
        "tanimoto_to_lead",
    ),
    "mmp_pairs": (
        "pair_id",
        "mol_id_lhs",
        "mol_id_rhs",
        "smiles_lhs",
        "smiles_rhs",
        "core_smiles",
        "smirks_transformation",
        "pic50_lhs",
        "pic50_rhs",
        "delta_pic50",
        "tanimoto_similarity",
        "sali_index",
        "is_activity_cliff",
        "bao_lhs",
        "bao_rhs",
        "pair_assay_homogeneous",
    ),
    "evaluated_candidates": (
        "candidate_id",
        "canonical_smiles",
        "scaffold_smiles",
        "generation_iteration",
        "parent_chembl_id",
        "parent_smiles",
        "parent_exp_pic50",
        "parent_exp_ic50_nm",
        "parent_bao_format",
        "parent_assay_type",
        "parent_source_dataset",
        "mpo_harrington_score",
        "passes_all_constraints",
    ),
}

TableTypeLiteral = Literal[
    "raw_dataset",
    "curated_dataset",
    "mmp_pairs",
    "evaluated_candidates",
]


# ============================================================================
# КАТЕГОРИЯ 1: ЭКСПЕРИМЕНТАЛЬНЫЙ ФАКТ (Фиолетовая зона UI)
# ============================================================================
class ExperimentalParentFact(BaseModel):
    """Категория 1: Неизменяемый экспериментальный факт о родительском соединении из ChEMBL или CSV."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_chembl_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор соединения-родителя (ChEMBL ID или mol_id)",
    )
    parent_smiles: str = Field(
        ...,
        min_length=1,
        description="Канонический SMILES экспериментального соединения-родителя",
    )
    parent_exp_pic50: float = Field(
        ..., description="Экспериментально измеренный pIC50 родителя (-log10 M)"
    )
    parent_exp_ic50_nm: float = Field(
        ..., gt=0.0, description="Экспериментально измеренный IC50/Ki родителя в нМ"
    )
    parent_bao_format: str = Field(
        ...,
        min_length=1,
        description="Код формата протокола испытания родителя (например, 'BAO_0000357')",
    )
    parent_assay_type: str = Field(
        ...,
        min_length=1,
        description="Тип экспериментального теста ('B', 'F' или 'CUSTOM_WETLAB')",
    )
    parent_source_dataset: str = Field(
        ...,
        min_length=1,
        description="Источник факта ('ChEMBL_Parquet', 'ChEMBL_API', 'Custom_WetLab_CSV')",
    )


# ============================================================================
# КАТЕГОРИЯ 2: ДЕТЕРМИНИРОВАННЫЙ РАСЧЕТ RDKIT (Зеленая зона UI)
# ============================================================================
class CalcRDKitMetrics(BaseModel):
    """Категория 2: Детерминированные физико-химические расчеты и структурные флаги RDKit.

    Вычисляются со 100% воспроизводимостью без участия ML или LLM.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mw: float = Field(
        ..., gt=0.0, description="Молекулярная масса (ExactMolWt), г/моль"
    )
    clogp: float = Field(
        ..., description="Расчетный коэффициент липофильности Wildman-Crippen (MolLogP)"
    )
    tpsa: float = Field(
        ...,
        ge=0.0,
        description="Топологическая полярная площадь поверхности (TPSA), Å²",
    )
    hbd: int = Field(
        ..., ge=0, description="Число доноров водородной связи (NumHDonors)"
    )
    hba: int = Field(
        ..., ge=0, description="Число акцепторов водородной связи (NumHAcceptors)"
    )
    rotatable_bonds: int = Field(
        ..., ge=0, description="Число вращающихся связей (NumRotatableBonds)"
    )
    fsp3: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Доля sp3-гибридизованных атомов углерода (FractionCSP3)",
    )
    qed: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Взвешенный индекс лекарственного сходства Bickerton (QED)",
    )
    sa_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Индекс синтетической доступности Ertl (1 — легко, 10 — сложно)",
    )
    lipinski_violations: int = Field(
        ..., ge=0, le=4, description="Число нарушений правила пяти Липински"
    )
    pains_alerts_count: int = Field(
        ..., ge=0, description="Число срабатываний фильтра PAINS (A/B/C)"
    )
    pains_matched_names: list[str] = Field(
        default_factory=list, description="Названия сработавших SMARTS-алертов PAINS"
    )
    brenk_alerts_count: int = Field(
        ...,
        ge=0,
        description="Число токсикофорных/реакционноспособных групп по каталогу Brenk",
    )
    brenk_matched_names: list[str] = Field(
        default_factory=list, description="Названия сработавших SMARTS-алертов Brenk"
    )
    glaxo_alerts_count: int = Field(
        default=0, ge=0, description="Число структурных алертов каталога Glaxo"
    )
    extra_alerts_count: int = Field(
        default=0,
        ge=0,
        description="Число структурных алертов доп. каталогов (Glaxo/SureChEMBL/NIH/ZINC) и custom_ban_smarts",
    )
    extra_alerts_matched_names: list[str] = Field(
        default_factory=list,
        description="Названия сработавших доп. SMARTS-алертов и пользовательских паттернов",
    )
    extra_descriptors: dict[str, float] = Field(
        default_factory=dict,
        description="Дополнительные вычисленные дескрипторы RDKit (ToolConfig.extra_rdkit_descriptors)",
    )
    brics_fragments: list[str] = Field(
        default_factory=list,
        description="Список канонических SMILES ретросинтетических строительных блоков (алгоритм BRICS)",
    )
    brics_bonds_broken: int = Field(
        default=0, ge=0, description="Число разорванных ретросинтетических связей BRICS"
    )
    delta_clogp_vs_parent: float = Field(
        ...,
        description="Точный сдвиг липофильности относительно родителя (cLogP_cand - cLogP_parent)",
    )
    delta_tpsa_vs_parent: float = Field(
        ...,
        description="Точный сдвиг полярной поверхности относительно родителя (TPSA_cand - TPSA_parent)",
    )


# ============================================================================
# КАТЕГОРИЯ 3: МОДЕЛЬНЫЙ ПРОГНОЗ ADMET / QSAR (Желтая зона UI)
# ============================================================================
class PredADMETMetrics(BaseModel):
    """Категория 3: Вероятностные предсказания ML/QSAR-моделей с явным указанием источника модели."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_source: Literal[
        "admet_ai_dmpnn", "catboost_local_qsar", "rdkit_analytical_qsar"
    ] = Field(..., description="Фактически использованный эшелон каскада моделей ADMET")
    log_s_aq: float = Field(
        ..., description="Прогноз логарифма молярной водной растворимости LogS (моль/л)"
    )
    solubility_class: Literal["high", "moderate", "poor", "insoluble"] = Field(
        ..., description="Категория растворимости по шкале LogS"
    )
    bbb_permeability_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Вероятность проникновения через гематоэнцефалический барьер (BBB+)",
    )
    herg_liability_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Вероятность кардиотоксичности (ингибирование калиевых каналов hERG)",
    )
    herg_risk_level: Literal["low", "moderate", "high"] = Field(
        ..., description="Категория риска hERG (<0.3 low, 0.3-0.6 moderate, >0.6 high)"
    )
    cyp3a4_inhibition_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Вероятность ингибирования изофермента цитохрома CYP3A4",
    )
    predicted_pic50: float = Field(
        ...,
        description="Ожидаемый pIC50 кандидата (parent_exp_pic50 + mean_delta_pic50 правила или локальный QSAR)",
    )
    prediction_confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Оценка надежности прогноза (зависит от числа подтверждающих пар MMP и AD)",
    )


# ============================================================================
# КАТЕГОРИЯ 4: НАУЧНАЯ ГИПОТЕЗА SAR (Синяя зона UI)
# ============================================================================
class SARHypothesisRationale(BaseModel):
    """Категория 4: Интерпретируемая научная гипотеза, связывающая модификацию со статистикой MMP."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_origin: Literal[
        "mmp_smirks_rule", "crem_fragment_replacement", "bioisostere_catalog"
    ] = Field(..., description="Механизм порождения модификации")
    rule_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор примененного правила SAR (например, 'R2' или 'CREM_ISO_04')",
    )
    applied_smirks: str = Field(
        ...,
        min_length=1,
        description="Строка трансформации SMIRKS вида '[*:1]R_old >> [*:1]R_new'",
    )
    replaced_group_smiles: str = Field(
        ..., description="Замененный периферийный фрагмент родительской молекулы"
    )
    introduced_group_smiles: str = Field(..., description="Введенный новый заместитель")
    historical_mean_delta_pic50: float = Field(
        ...,
        description="Средний прирост активности ΔpIC50 данной трансформации на исторических парах",
    )
    historical_std_delta_pic50: float = Field(
        default=0.0,
        ge=0.0,
        description="Стандартное отклонение ΔpIC50 на исторических парах",
    )
    support_pairs_count: int = Field(
        ...,
        ge=0,
        description="Число экспериментальных пар молекул, подтверждающих правило",
    )
    assay_homogeneous: bool = Field(
        ...,
        description="Подтверждение, что правило извлечено на однородных протоколах испытаний (BAO)",
    )
    text_summary: str = Field(
        ...,
        min_length=1,
        description="Текстовая формулировка гипотезы на русском языке с обоснованием",
    )


# ============================================================================
# ЕДИНЫЙ КОНТРАКТ КАНДИДАТА (CandidateMolecule)
# ============================================================================
class CandidateMolecule(BaseModel):
    """Контракт молекулы-кандидата с принудительным разделением 4 категорий знаний."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Идентификация кандидата
    candidate_id: str = Field(
        ...,
        min_length=1,
        description="Уникальный идентификатор кандидата (напр., 'CAND_IT2_01')",
    )
    canonical_smiles: str = Field(
        ..., min_length=1, description="Канонический SMILES сгенерированного кандидата"
    )
    scaffold_smiles: str = Field(
        ...,
        min_length=1,
        description="Канонический SMILES сохраненного скаффолда Bemis-Murcko",
    )
    generation_iteration: int = Field(
        ..., ge=1, description="Номер итерации дизайна, на которой получен кандидат"
    )

    # 1. ЭКСПЕРИМЕНТАЛЬНЫЙ ФАКТ (Фиолетовая зона UI) — строго данные из БД ChEMBL или лабораторного CSV
    parent_chembl_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор экспериментального соединения-родителя (ChEMBL ID или mol_id)",
    )
    parent_smiles: str = Field(
        ...,
        min_length=1,
        description="Канонический SMILES экспериментального соединения-родителя",
    )
    parent_exp_pic50: float = Field(
        ..., description="Экспериментально измеренный pIC50 родителя (-log10 M)"
    )
    parent_exp_ic50_nm: float = Field(
        ..., gt=0.0, description="Экспериментально измеренный IC50/Ki родителя в нМ"
    )
    parent_bao_format: str = Field(
        ...,
        min_length=1,
        description="Код формата протокола испытания родителя (например, 'BAO_0000357')",
    )
    parent_assay_type: str = Field(
        ...,
        min_length=1,
        description="Тип экспериментального теста ('B', 'F' или 'CUSTOM_WETLAB')",
    )
    parent_source_dataset: str = Field(
        ...,
        min_length=1,
        description="Источник факта ('ChEMBL_Parquet', 'ChEMBL_API', 'Custom_WetLab_CSV')",
    )

    # 2. ДЕТЕРМИНИРОВАННЫЙ РАСЧЕТ (Зеленая зона UI)
    calc_rdkit: CalcRDKitMetrics

    # 3. МОДЕЛЬНЫЙ ПРОГНОЗ (Желтая зона UI)
    pred_admet: PredADMETMetrics

    # 4. НАУЧНАЯ ГИПОТЕЗА (Синяя зона UI)
    hypothesis_rationale: SARHypothesisRationale

    # Интегральная многопараметрическая оценка и статус валидации Критиком
    mpo_harrington_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Обобщенный индекс желательности Харрингтона D in [0, 1] (0 при Veto)",
    )
    passes_all_constraints: bool = Field(
        ...,
        description="Флаг полного соответствия всем целевым ограничениям пользователя",
    )
    constraint_violations: list[str] = Field(
        default_factory=list,
        description="Список нарушенных порогов (заполняется для компромиссных кандидатов при выходе по лимиту итераций)",
    )
    highlighted_atom_indices: list[int] = Field(
        default_factory=list,
        description="Индексы атомов модифицированного заместителя для цветовой подсветки в rdMolDraw2D",
    )
    alert_penalty_mode: Literal["veto", "soft_penalty", "ignore"] = Field(
        default="veto",
        description="Режим штрафа за структурные алерты при расчете индекса Харрингтона",
    )

    @model_validator(mode="after")
    def _enforce_veto_invariant(self) -> CandidateMolecule:
        """Проверяет инвариант вето Харрингтона: при режиме 'veto' и наличии PAINS/Brenk индекс D обязан быть 0.0."""
        if (
            self.alert_penalty_mode == "veto"
            and (
                self.calc_rdkit.pains_alerts_count > 0
                or self.calc_rdkit.brenk_alerts_count > 0
                or self.calc_rdkit.extra_alerts_count > 0
            )
            and self.mpo_harrington_score > 0.0
        ):
            raise ValueError(
                "Harrington Veto Invariant violated: mpo_harrington_score must be 0.0 "
                "when alerts > 0 under alert_penalty_mode='veto'."
            )
        return self

    @property
    def experimental_fact(self) -> ExperimentalParentFact:
        """Возвращает изолированный объект Категории 1 (Экспериментальный факт родителя)."""
        return ExperimentalParentFact(
            parent_chembl_id=self.parent_chembl_id,
            parent_smiles=self.parent_smiles,
            parent_exp_pic50=self.parent_exp_pic50,
            parent_exp_ic50_nm=self.parent_exp_ic50_nm,
            parent_bao_format=self.parent_bao_format,
            parent_assay_type=self.parent_assay_type,
            parent_source_dataset=self.parent_source_dataset,
        )

    def to_four_category_dict(self) -> dict[str, dict[str, Any]]:
        """Возвращает представление кандидата, сгруппированное строго по 4 категориям знаний."""
        return {
            "category_1_experiment": self.experimental_fact.model_dump(mode="python"),
            "category_2_calculation": self.calc_rdkit.model_dump(mode="python"),
            "category_3_prediction": self.pred_admet.model_dump(mode="python"),
            "category_4_hypothesis": self.hypothesis_rationale.model_dump(
                mode="python"
            ),
        }

    def to_flat_parquet_record(self) -> dict[str, Any]:
        """Сериализует объект кандидата в плоскую строку для сохранения в evaluated_candidates.parquet."""
        return {
            "candidate_id": self.candidate_id,
            "canonical_smiles": self.canonical_smiles,
            "scaffold_smiles": self.scaffold_smiles,
            "generation_iteration": self.generation_iteration,
            "parent_chembl_id": self.parent_chembl_id,
            "parent_smiles": self.parent_smiles,
            "parent_exp_pic50": self.parent_exp_pic50,
            "parent_exp_ic50_nm": self.parent_exp_ic50_nm,
            "parent_bao_format": self.parent_bao_format,
            "parent_assay_type": self.parent_assay_type,
            "parent_source_dataset": self.parent_source_dataset,
            # Поля верхнего уровня для быстрой фильтрации в DuckDB
            "mw": self.calc_rdkit.mw,
            "clogp": self.calc_rdkit.clogp,
            "tpsa": self.calc_rdkit.tpsa,
            "qed": self.calc_rdkit.qed,
            "sa_score": self.calc_rdkit.sa_score,
            "pains_alerts_count": self.calc_rdkit.pains_alerts_count,
            "brenk_alerts_count": self.calc_rdkit.brenk_alerts_count,
            "model_source": self.pred_admet.model_source,
            "predicted_pic50": self.pred_admet.predicted_pic50,
            "herg_liability_prob": self.pred_admet.herg_liability_prob,
            "rule_id": self.hypothesis_rationale.rule_id,
            "applied_smirks": self.hypothesis_rationale.applied_smirks,
            "mpo_harrington_score": self.mpo_harrington_score,
            "passes_all_constraints": self.passes_all_constraints,
            "alert_penalty_mode": self.alert_penalty_mode,
            # Полные JSON-слепки вложенных контрактов для 100% обратимой десериализации
            "calc_rdkit_json": self.calc_rdkit.model_dump_json(),
            "pred_admet_json": self.pred_admet.model_dump_json(),
            "hypothesis_rationale_json": self.hypothesis_rationale.model_dump_json(),
            "constraint_violations_json": json.dumps(
                self.constraint_violations, ensure_ascii=False
            ),
            "highlighted_atom_indices_json": json.dumps(self.highlighted_atom_indices),
        }

    @classmethod
    def from_flat_parquet_record(cls, row: dict[str, Any]) -> CandidateMolecule:
        """Восстанавливает типизированный CandidateMolecule из строки evaluated_candidates.parquet."""
        return cls(
            candidate_id=str(row["candidate_id"]),
            canonical_smiles=str(row["canonical_smiles"]),
            scaffold_smiles=str(row["scaffold_smiles"]),
            generation_iteration=int(row["generation_iteration"]),
            parent_chembl_id=str(row["parent_chembl_id"]),
            parent_smiles=str(row["parent_smiles"]),
            parent_exp_pic50=float(row["parent_exp_pic50"]),
            parent_exp_ic50_nm=float(row["parent_exp_ic50_nm"]),
            parent_bao_format=str(row["parent_bao_format"]),
            parent_assay_type=str(row["parent_assay_type"]),
            parent_source_dataset=str(row["parent_source_dataset"]),
            calc_rdkit=CalcRDKitMetrics.model_validate_json(row["calc_rdkit_json"]),
            pred_admet=PredADMETMetrics.model_validate_json(row["pred_admet_json"]),
            hypothesis_rationale=SARHypothesisRationale.model_validate_json(
                row["hypothesis_rationale_json"]
            ),
            mpo_harrington_score=float(row["mpo_harrington_score"]),
            passes_all_constraints=bool(row["passes_all_constraints"]),
            alert_penalty_mode=row.get("alert_penalty_mode") or "veto",
            constraint_violations=json.loads(
                row.get("constraint_violations_json") or "[]"
            ),
            highlighted_atom_indices=json.loads(
                row.get("highlighted_atom_indices_json") or "[]"
            ),
        )


# ============================================================================
# ТОЛСТЫЕ УКАЗАТЕЛИ (FAT POINTERS) И ВСПОМОГАТЕЛЬНЫЕ КОНТРАКТЫ СОСТОЯНИЯ
# ============================================================================
class ParquetPointer(BaseModel):
    """Толстый указатель (Fat Pointer) на табличную выборку в сессионном хранилище Parquet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(
        ...,
        min_length=1,
        description="Абсолютный или относительный путь к Parquet-файлу в workspace/<session_id>/",
    )
    table_type: TableTypeLiteral
    row_count: int = Field(..., ge=0, description="Количество записей в таблице")
    columns: list[str] = Field(..., description="Список колонок таблицы")
    sha256_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Хэш контрольной суммы файла (SHA-256) для воспроизводимости",
    )

    @model_validator(mode="after")
    def _validate_required_columns(self) -> ParquetPointer:
        """Гарантирует, что Parquet-таблица содержит все обязательные колонки своего типа."""
        required = set(PARQUET_TABLE_REQUIRED_COLUMNS.get(self.table_type, ()))
        actual = set(self.columns)
        missing = sorted(required - actual)
        if missing:
            raise ValueError(
                f"ParquetPointer schema mismatch for table_type='{self.table_type}': "
                f"missing required columns {missing}"
            )
        return self


class QueryConstraints(BaseModel):
    """Структурированные ограничения и цели оптимизации, извлеченные из запроса пользователя."""

    model_config = ConfigDict(extra="forbid")

    scenario_type: Literal[
        "target_based_sar", "lead_optimization", "custom_csv_analysis"
    ]
    target_chembl_id: str | None = Field(
        default=None,
        description="Целевой идентификатор мишени ChEMBL (напр., 'CHEMBL203')",
    )
    lead_smiles: str | None = Field(
        default=None, description="SMILES исходного лидера для Сценария 2"
    )
    freeze_scaffold: bool = Field(
        default=True, description="Требование сохранения каркаса Bemis-Murcko"
    )
    min_logp: float | None = Field(
        default=None, description="Нижняя граница cLogP (если задана)"
    )
    max_logp: float = Field(default=4.5, description="Верхний допустимый порог cLogP")
    max_sa_score: float = Field(
        default=3.2,
        description="Верхний допустимый порог синтетической доступности SAscore",
    )
    min_tpsa: float | None = Field(
        default=None,
        description="Минимальная полярная площадь поверхности TPSA (устанавливается также критиком)",
    )
    max_mw: float = Field(
        default=550.0, description="Максимальная молекулярная масса, г/моль"
    )
    max_herg_prob: float = Field(
        default=0.5, description="Максимально допустимая вероятность блокады hERG"
    )
    require_no_pains: bool = Field(
        default=True, description="Жесткий запрет структурных алертов PAINS/Brenk"
    )
    top_k_candidates: int = Field(
        default=5, ge=1, le=20, description="Требуемое число финальных кандидатов"
    )
    preferred_substituent_types: list[
        Literal["polar", "less_lipophilic", "exclude_halogens", "any"]
    ] = Field(
        default_factory=lambda: ["any"],
        description="Директивные ограничения на тип радикалов, модифицируемые Critic Agent во 2-й петле",
    )


class SARRule(BaseModel):
    """Извлеченное правило трансформации 'структура–активность' из сети парных соответствий rdMMPA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(
        ..., min_length=1, description="Уникальный код правила (напр., 'R1', 'R2')"
    )
    smirks: str = Field(
        ...,
        min_length=1,
        description="Трансформация в формате SMIRKS: '[*:1]R_from >> [*:1]R_to'",
    )
    from_fragment: str = Field(..., description="Исходный заместитель")
    to_fragment: str = Field(..., description="Новый заместитель")
    mean_delta_pic50: float = Field(
        ..., description="Средний сдвиг активности ΔpIC50 по всем подтверждающим парам"
    )
    std_delta_pic50: float = Field(
        default=0.0, ge=0.0, description="Стандартное отклонение ΔpIC50"
    )
    support_pairs_count: int = Field(
        ..., ge=1, description="Число экспериментальных пар (Support)"
    )
    mean_delta_clogp: float = Field(
        ..., description="Средний сдвиг липофильности ΔcLogP при данной замене"
    )
    mean_delta_tpsa: float = Field(
        ..., description="Средний сдвиг полярной площади ΔTPSA при данной замене"
    )
    assay_homogeneous: bool = Field(
        ...,
        description="True, если 100% пар внутри правила измерены в одном формате BAO и одном типе теста",
    )
    participating_bao_formats: list[str] = Field(
        ...,
        description="Список уникальных кодов BAO_format среди пар, образующих правило",
    )
    example_pair_ids: list[str] = Field(
        default_factory=list, description="Ссылки на pair_id в mmp_pairs.parquet"
    )


class ActivityCliff(BaseModel):
    """Запись о критическом перепаде активности (Activity Cliff) по индексу SALI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cliff_id: str = Field(..., min_length=1)
    mol_id_active: str = Field(..., min_length=1)
    mol_id_inactive: str = Field(..., min_length=1)
    smiles_active: str = Field(..., min_length=1)
    smiles_inactive: str = Field(..., min_length=1)
    pic50_active: float
    pic50_inactive: float
    abs_delta_pic50: float = Field(
        ..., ge=0.0, description="Абсолютный перепад активности |ΔpIC50|"
    )
    tanimoto_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Сходство по выбранной метрике фингерпринтов"
    )
    sali_index: float = Field(..., gt=0.0, description="Значение индекса SALI")
    smirks_difference: str = Field(
        ..., description="Минимальная структурная замена, вызвавшая скачок активности"
    )
    assay_homogeneous: bool


class DatasetCurationSummary(BaseModel):
    """Агрегированная сводка курации выборки для передачи в состояние графа без перегрузки LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: Literal["chembl_parquet", "chembl_api", "custom_csv"]
    initial_rows_count: int = Field(..., ge=0)
    invalid_smiles_dropped: int = Field(..., ge=0)
    salts_stripped_count: int = Field(..., ge=0)
    duplicates_merged_count: int = Field(..., ge=0)
    censored_values_count: int = Field(..., ge=0)
    curated_molecules_count: int = Field(..., ge=0)
    dominant_bao_format: str = Field(
        ..., description="Доминирующий протокол BAO (напр., 'BAO_0000357')"
    )
    dominant_standard_type: str = Field(
        ..., description="Доминирующий тип измерения (напр., 'IC50')"
    )
    bao_distribution: dict[str, int] = Field(
        ..., description="Распределение числа записей по кодам BAO_format"
    )
    is_dataset_assay_homogeneous: bool = Field(
        ...,
        description="True, если в выборке присутствует только один BAO_format и standard_type",
    )
    top_scaffolds: list[dict[str, str | int | float]] = Field(
        default_factory=list,
        description="Топ-5 скаффолдов Bemis-Murcko с числом представителей и средним pIC50",
    )
    validation_errors_log: list[str] = Field(
        default_factory=list, description="Журнал отбракованных строк CSV/ChEMBL"
    )


class TraceEvent(BaseModel):
    """Запись журнала Explainable Trace для отображения в UI и экспорта в trace_log.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(..., ge=0)
    timestamp_iso: str = Field(..., min_length=1)
    agent_name: Literal[
        "ResearchAgent", "SARAgent", "DesignAgent", "EvaluationAgent", "CriticAgent"
    ]
    phase: str = Field(..., min_length=1)
    input_summary: str
    action_taken: str
    output_summary: str
    verdict_or_routing: str | None = None
    fallback_triggered: str | None = None
    duration_ms: float = Field(..., ge=0.0)


class ParentSeed(BaseModel):
    """Экспериментальная родительская молекула, подаваемая на вход модуля генерации."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mol_id: str = Field(..., min_length=1)
    canonical_smiles: str = Field(..., min_length=1)
    scaffold_smiles: str = Field(..., min_length=1)
    exp_pic50: float
    exp_ic50_nm: float = Field(..., gt=0.0)
    bao_format: str = Field(..., min_length=1)
    assay_type: str = Field(..., min_length=1)
    source_dataset: str = Field(..., min_length=1)


class GeneratedRawCandidate(BaseModel):
    """Сырой сгенерированный кандидат до расчета дескрипторов и ADMET (выход модуля generation.py)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    smiles: str = Field(..., min_length=1)
    scaffold_smiles: str = Field(..., min_length=1)
    generation_iteration: int = Field(..., ge=1)
    parent_seed: ParentSeed
    rule_id: str = Field(..., min_length=1)
    applied_smirks: str = Field(..., min_length=1)
    replaced_group_smiles: str
    introduced_group_smiles: str
    generation_origin: Literal[
        "mmp_smirks_rule", "crem_fragment_replacement", "bioisostere_catalog"
    ]
    historical_mean_delta_pic50: float
    historical_std_delta_pic50: float = Field(default=0.0, ge=0.0)
    support_pairs_count: int = Field(..., ge=0)
    assay_homogeneous: bool
    highlighted_atom_indices: list[int] = Field(default_factory=list)


# ============================================================================
# ЕДИНАЯ ПАРАМЕТРИЗАЦИЯ ИНСТРУМЕНТОВ И ОТСЛЕЖИВАНИЕ ИСТОЧНИКОВ (ToolConfig)
# ============================================================================
ParamSourceLiteral = Literal["default", "nl_query", "ui_manual", "critic_loop"]

FilterCatalogLiteral = Literal[
    "PAINS_A", "PAINS_B", "PAINS_C", "Brenk", "Glaxo", "SureChEMBL", "NIH", "ZINC"
]

ADMETBackendLiteral = Literal[
    "auto",
    "auto_catboost",
    "admet_ai_dmpnn",
    "catboost_local_qsar",
    "rdkit_analytical_qsar",
    "empirical_only",
]


def _default_mpo_weights() -> dict[str, float]:
    return {
        "pic50": 1.5,
        "logp": 1.0,
        "sascore": 1.0,
        "qed": 1.0,
        "tpsa": 0.8,
        "logs": 0.8,
        "herg": 1.2,
    }


class ToolConfig(BaseModel):
    """Единая конфигурация всех вычислительных подсистем, фильтров, ADMET/MPO, 3D и оркестратора."""

    model_config = ConfigDict(extra="forbid")

    # 1. Курация (curation.py)
    allowed_bao_formats: list[str] = Field(
        default_factory=lambda: ["BAO_0000357", "BAO_0000219"]
    )
    standard_types: list[str] = Field(default_factory=lambda: ["IC50", "Ki"])
    relation_mode: Literal["exact_only", "include_censored"] = "include_censored"
    pchembl_min_cutoff: float = Field(default=4.0, ge=0.0, le=15.0)
    duplicate_agg_method: Literal["median", "mean", "max"] = "median"
    max_duplicate_std: float = Field(default=0.5, ge=0.0)
    strip_salts: bool = True
    neutralize_charges: bool = True
    canonicalize_tautomers: bool = True
    keep_chirality: bool = True
    scaffold_mode: Literal["bemis_murcko", "generic_framework"] = "bemis_murcko"
    chembl_api_limit: int = Field(default=350, ge=10, le=5000)
    min_neighborhood_rows_fallback: int = Field(default=5, ge=1)
    top_scaffolds_limit: int = Field(default=5, ge=1, le=50)

    # 2. Парный анализ и SALI (sar_analysis.py)
    mmpa_max_cuts: Literal[1, 2, 3] = 1
    mmpa_max_cut_bonds: int = Field(default=20, ge=1)
    min_support_pairs: int = Field(default=2, ge=1)
    min_mean_delta_pic50: float = 0.3
    max_rgroup_heavy_atoms: int = Field(default=12, ge=1)
    fp_type: Literal["ecfp", "fcfp", "rdkit", "maccs"] = "ecfp"
    fp_radius: Literal[1, 2, 3] = 2
    fp_nbits: Literal[1024, 2048, 4096] = 2048
    similarity_metric: Literal["tanimoto", "dice", "cosine"] = "tanimoto"
    cliff_similarity_cutoff: float = Field(default=0.85, ge=0.0, le=1.0)
    cliff_delta_pic50_cutoff: float = Field(default=1.5, ge=0.0)
    sali_epsilon: float = Field(default=0.05, gt=0.0)
    mmpa_min_core_heavy_atoms: int = Field(default=6, ge=1)
    min_rules_before_enrichment: int = Field(default=3, ge=0)
    background_target_id: str = "CHEMBL203"
    prioritize_positive_rules: bool = True

    # 3. Генерация и биоизостеры (generation.py, bioisosteres.py, design_node.py)
    generation_engine: Literal["mmp_and_crem", "mmp_only", "crem_only"] = (
        "mmp_and_crem"
    )
    crem_mode: Literal["mutate", "grow"] = "mutate"
    crem_context_radius: Literal[1, 2, 3] = 3
    crem_min_rel_atoms: int = -3
    crem_max_rel_atoms: int = 6
    crem_max_replacements: int = Field(default=25, ge=1)
    freeze_murcko_core: bool = True
    protected_smarts: str | None = None
    modifiable_atom_indices: list[int] | None = None
    neighborhood_tanimoto: float = Field(default=0.60, ge=0.0, le=1.0)
    max_candidates_pool: int = Field(default=25, ge=1)
    top_k_output: int = Field(default=5, ge=1, le=50)
    max_parent_seeds: int = Field(default=5, ge=1, le=50)
    use_bioisostere_catalog: bool = True
    bioisostere_fallback_threshold: int = Field(default=5, ge=0)
    crem_max_fragment_heavy_atoms: int = Field(default=4, ge=1)

    # 4. Оценка, дескрипторы, фильтры, BRICS, ADMET, MPO и 3D
    active_filter_catalogs: list[FilterCatalogLiteral] = Field(
        default_factory=lambda: ["PAINS_A", "PAINS_B", "PAINS_C", "Brenk"]
    )
    custom_ban_smarts: list[str] = Field(default_factory=list)
    brics_min_fragment_size: int = Field(default=3, ge=1)
    brics_only_leaf_nodes: bool = True
    admet_backend: ADMETBackendLiteral = "auto_catboost"
    cns_profile_mode: Literal["peripheral_low_bbb", "cns_high_bbb", "neutral"] = (
        "peripheral_low_bbb"
    )
    alert_penalty_mode: Literal["veto", "soft_penalty", "ignore"] = "veto"
    alert_soft_multiplier: float = Field(default=0.5, ge=0.0, le=1.0)
    mpo_weights: dict[str, float] = Field(default_factory=_default_mpo_weights)
    etkdg_variant: Literal["ETKDGv3", "ETKDGv2"] = "ETKDGv3"
    force_field: Literal["MMFF94s", "MMFF94", "UFF", "none"] = "MMFF94s"
    num_conformers: int = Field(default=5, ge=1, le=50)
    extra_rdkit_descriptors: list[str] = Field(default_factory=list)
    brenk_ignore_long_chain: bool = True
    lipinski_mw_cutoff: float = Field(default=500.0, gt=0.0)
    lipinski_logp_cutoff: float = 5.0
    lipinski_hbd_cutoff: int = Field(default=5, ge=0)
    lipinski_hba_cutoff: int = Field(default=10, ge=0)
    harrington_activity_low_delta: float = -0.20
    harrington_activity_high_delta: float = 0.55
    harrington_qed_min: float = Field(default=0.35, ge=0.0, le=1.0)
    harrington_qed_target: float = Field(default=0.65, ge=0.0, le=1.0)
    harrington_sa_min: float = Field(default=2.0, ge=1.0, le=10.0)
    herg_low_cutoff: float = Field(default=0.30, ge=0.0, le=1.0)
    herg_high_cutoff: float = Field(default=0.60, ge=0.0, le=1.0)
    logs_soluble_cutoff: float = -2.0
    logs_moderate_cutoff: float = -4.0
    logs_poor_cutoff: float = -6.0
    conformer_opt_max_iters: int = Field(default=200, ge=0, le=5000)
    conformer_random_seed: int = 42
    pca_random_seed: int = 42

    # 5. Оркестрация (critic_node.py, workflow.py)
    max_iterations: int = Field(default=2, ge=0, le=10)
    strict_assay_homogeneity: bool = True
    strict_property_thresholds: bool = True
    max_refilter_iterations: int = Field(default=1, ge=0, le=5)
    critic_fallback_min_tpsa: float = Field(default=85.0, ge=0.0)
    critic_logp_step_reduction: float = Field(default=0.7, ge=0.0)

    def apply_overrides(
        self,
        overrides: dict[str, Any] | None,
        *,
        allow_none_keys: set[str] | frozenset[str] | None = None,
    ) -> tuple[ToolConfig, list[str]]:
        """Валидирует и применяет частичные переопределения, возвращая новый ToolConfig и список измененных ключей."""
        if not overrides:
            return self, []

        current = self.model_dump(mode="python")
        candidate = dict(current)
        nullable = allow_none_keys or frozenset()
        for key, raw_val in overrides.items():
            if key not in ToolConfig.model_fields:
                continue
            if raw_val is None and key not in nullable:
                continue
            if key == "mpo_weights" and isinstance(raw_val, dict):
                merged_weights = dict(current["mpo_weights"])
                for wk, wv in raw_val.items():
                    if wv is not None:
                        merged_weights[str(wk)] = float(wv)
                candidate["mpo_weights"] = merged_weights
            else:
                candidate[key] = raw_val

        updated = ToolConfig.model_validate(candidate)
        updated_dump = updated.model_dump(mode="python")
        changed_keys = [
            k
            for k in ToolConfig.model_fields
            if updated_dump.get(k) != current.get(k)
        ]
        return updated, changed_keys
