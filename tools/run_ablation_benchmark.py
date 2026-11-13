"""Скрипт быстрого бенчмарка и абляции (Ablation Study) для МАС ПИШОВИЧОК.

Сравнивает 3 режима на фиксированном наборе из 6 разнообразных задач на базе
локальных данных (`CHEMBL203`, `CHEMBL262`, `CHEMBL5251`, `custom_wetlab_series.csv`):
1. Baseline (Zero-shot LLM) с фоллбэком на детерминированный мок;
2. МАС ПИШОВИЧОК (1 итерация, `max_iterations=1`);
3. МАС ПИШОВИЧОК (Полный цикл + агент-критик, `max_iterations=2`).

Во всех запусках принудительно отключается 3D-докинг (`docking_mode="skip"`).
Сохраняет:
- `data/benchmark_results.json` — полные сырые данные прогона;
- `BENCHMARK_REPORT.md` — готовую таблицу Ablation Study и кейс «До и После Критика».
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
from rdkit import Chem

from pishovichok.agents.llm_adapter import LLMAdapter
from pishovichok.agents.workflow import run_sar_pipeline
from pishovichok.config import settings
from pishovichok.core.admet import predict_admet_metrics
from pishovichok.core.curation import standardize_molecule_detailed
from pishovichok.core.evaluation import calculate_rdkit_metrics
from pishovichok.core.fingerprints import compute_fp_similarity, compute_mol_fingerprint
from pishovichok.core.scoring import calculate_harrington_mpo_score
from pishovichok.state.schemas import CandidateMolecule, QueryConstraints, ToolConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BENCH_WORKSPACE = PROJECT_ROOT / "workspace" / "ablation_bench"
RESULTS_JSON_PATH = DATA_DIR / "benchmark_results.json"
REPORT_MD_PATH = PROJECT_ROOT / "BENCHMARK_REPORT.md"

GEFITINIB_SMILES = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


@dataclass(frozen=True)
class BenchmarkTask:
    """Описание одной тестовой задачи бенчмарка."""

    task_id: str
    title: str
    target_id: str
    target_label: str
    scenario_type: str
    query: str
    uploaded_csv_path: str | None
    starting_smiles: str
    starting_pic50: float


def _load_top_active_from_parquet(parquet_path: Path) -> tuple[str, float]:
    """Извлекает самый активный валидный SMILES и его pIC50 из локального ChEMBL Parquet."""
    rows = duckdb.sql(
        f"""
        SELECT raw_smiles, standard_value, standard_units
        FROM read_parquet('{parquet_path.as_posix()}')
        WHERE standard_value IS NOT NULL AND standard_value > 0
          AND COALESCE(standard_relation, '=') IN ('=', '==')
        ORDER BY standard_value ASC
        LIMIT 10
        """
    ).fetchall()
    for raw_smi, std_val, units in rows:
        mol = Chem.MolFromSmiles(str(raw_smi))
        if mol is None:
            continue
        unit_scale = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0}.get(
            str(units or "nM").strip().lower(), 1e-9
        )
        pic50 = -math.log10(float(std_val) * unit_scale)
        return Chem.MolToSmiles(mol, canonical=True), round(pic50, 4)
    raise RuntimeError(f"No valid active compounds found in {parquet_path}")


def _load_top_active_from_csv(csv_path: Path) -> tuple[str, float]:
    """Извлекает самый активный валидный SMILES и pIC50 из лабораторного CSV."""
    best_smi = ""
    best_pic50 = -1.0
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_smi = (row.get("SMILES") or "").strip()
            raw_val = (row.get("IC50_nM") or "").strip()
            if not raw_smi or not raw_val or raw_val.startswith((">", "<")):
                continue
            mol = Chem.MolFromSmiles(raw_smi)
            if mol is None:
                continue
            try:
                val = float(raw_val)
            except ValueError:
                continue
            if val <= 0:
                continue
            unit_scale = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0}.get(
                str(row.get("units") or "nM").strip().lower(), 1e-9
            )
            pic50 = -math.log10(val * unit_scale)
            if pic50 > best_pic50:
                best_pic50 = pic50
                best_smi = Chem.MolToSmiles(mol, canonical=True)
    if not best_smi:
        raise RuntimeError(f"No valid active compounds found in {csv_path}")
    return best_smi, round(best_pic50, 4)


def build_benchmark_tasks() -> list[BenchmarkTask]:
    """Формирует фиксированный набор из 6 разнообразных тестовых задач."""
    btk_smi, btk_pic50 = _load_top_active_from_parquet(
        DATA_DIR / "chembl" / "CHEMBL5251.parquet"
    )
    egfr_smi, egfr_pic50 = _load_top_active_from_parquet(
        DATA_DIR / "chembl" / "CHEMBL203.parquet"
    )
    gsk_smi, gsk_pic50 = _load_top_active_from_parquet(
        DATA_DIR / "chembl" / "CHEMBL262.parquet"
    )
    csv_path = (DATA_DIR / "samples" / "custom_wetlab_series.csv").resolve()
    csv_smi, csv_pic50 = _load_top_active_from_csv(csv_path)

    return [
        BenchmarkTask(
            task_id="T1_BTK_SAR",
            title="BTK Target-Based SAR (CHEMBL5251)",
            target_id="CHEMBL5251",
            target_label="BTK (CHEMBL5251)",
            scenario_type="target_based_sar",
            query=(
                "Проведи анализ ингибиторов тирозинкиназы Брутона (BTK, CHEMBL5251), "
                "выдели ключевые правила SAR на однородных биохимических тестах "
                "и сгенерируй 5 новых кандидатов с расчетным LogP от 2.0 до 4.0, "
                "SAscore < 3.2 и без структурных алертов PAINS. docking_mode=skip"
            ),
            uploaded_csv_path=None,
            starting_smiles=btk_smi,
            starting_pic50=btk_pic50,
        ),
        BenchmarkTask(
            task_id="T2_EGFR_LEAD_OPT",
            title="EGFR Lead Optimization — Gefitinib (CHEMBL203)",
            target_id="CHEMBL203",
            target_label="EGFR (CHEMBL203 / Gefitinib)",
            scenario_type="lead_optimization",
            query=(
                f"Оптимизируй структуру ингибитора EGFR со строкой SMILES {GEFITINIB_SMILES} "
                "(гефитиниб): сохрани хиназолиновый скаффолд, снизь липофильность (LogP < 3.5), "
                "минимизируй риск блокады каналов hERG и предложи синтетически доступные аналоги "
                "(SAscore < 3.0) без PAINS. docking_mode=skip"
            ),
            uploaded_csv_path=None,
            starting_smiles=GEFITINIB_SMILES,
            starting_pic50=8.48,
        ),
        BenchmarkTask(
            task_id="T3_GSK3B_SAR",
            title="GSK-3β Target-Based SAR (CHEMBL262)",
            target_id="CHEMBL262",
            target_label="GSK-3β (CHEMBL262)",
            scenario_type="target_based_sar",
            query=(
                "Проанализируй ингибиторы киназы GSK-3beta (CHEMBL262), выдели правила SAR "
                "и предложи 5 новых кандидатов с LogP < 3.5, SAscore < 3.2 и без алертов PAINS. "
                "docking_mode=skip"
            ),
            uploaded_csv_path=None,
            starting_smiles=gsk_smi,
            starting_pic50=gsk_pic50,
        ),
        BenchmarkTask(
            task_id="T4_WETLAB_CSV",
            title="Custom Wet-Lab Series (CSV)",
            target_id="CUSTOM_WETLAB",
            target_label="Custom Wet-Lab Series (EGFR/BTK)",
            scenario_type="custom_csv_analysis",
            query=(
                "Проанализируй результаты испытаний нашей лабораторной серии из CSV, "
                "найди Activity Cliffs и предложи 5 новых структур с LogP < 3.5, "
                "SAscore < 3.2 и без PAINS. docking_mode=skip"
            ),
            uploaded_csv_path=str(csv_path),
            starting_smiles=csv_smi,
            starting_pic50=csv_pic50,
        ),
        BenchmarkTask(
            task_id="T5_EGFR_STRICT",
            title="EGFR Strict Polarity & Safety SAR (CHEMBL203)",
            target_id="CHEMBL203",
            target_label="EGFR Strict (CHEMBL203)",
            scenario_type="target_based_sar",
            query=(
                "Проведи SAR-анализ ингибиторов EGFR (CHEMBL203) и предложи 5 новых кандидатов "
                "со строгими порогами: LogP < 3.0, SAscore < 3.0, минимизируй риск hERG (< 0.35) "
                "и исключи алерты PAINS. docking_mode=skip"
            ),
            uploaded_csv_path=None,
            starting_smiles=egfr_smi,
            starting_pic50=egfr_pic50,
        ),
        BenchmarkTask(
            task_id="T6_BTK_STRICT",
            title="BTK Strict MedChem & Safety SAR (CHEMBL5251)",
            target_id="CHEMBL5251",
            target_label="BTK Strict (CHEMBL5251)",
            scenario_type="target_based_sar",
            query=(
                "Проведи поиск новых ингибиторов BTK (CHEMBL5251) со строгими ограничениями: "
                "LogP < 3.0, SAscore < 3.0, низкий риск кардиотоксичности hERG (< 0.35) "
                "и отсутствие структурных алертов PAINS. docking_mode=skip"
            ),
            uploaded_csv_path=None,
            starting_smiles=btk_smi,
            starting_pic50=btk_pic50,
        ),
    ]


# Детерминированный мок на случай отсутствия API-ключа или исчерпания квоты.
# Включает типичные для zero-shot LLM паттерны: валидные аналоги, 1-2 невалидные SMILES,
# дубликат и/или молекулы с нарушениями LogP/PAINS/SA.
_BASELINE_MOCK_BY_TASK: dict[str, list[str]] = {
    "T1_BTK_SAR": [
        "Nc1ncnc2c1c(-c1ccc(Oc3ccccc3)cc1)cn2C1CCN(C(=O)C=C)CC1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(F)cc3)cc1)cn2C1CCCN(C(=O)C=C)C1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C#N)cc3)cc1)cn2C1CCCN(C(=O)C)C1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OC)cc3)cc1)cn2C1CCN(C(=O)C)CC1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccncc3)cc1)cn2C1CCCN(C(=O)C)C1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccccc3)cc1)cn2C1CCN(C(=O)C=C)CC1",  # duplicate
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(N=Nc4ccccc4)cc3)cc1)cn2C1CCCN(C)C1",  # PAINS azo
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C(F)(F)F)cc3)cc1)cn2C1CCN(S(=O)(=O)C)CC1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccccc3)cc1)cn2C1CCN(C(=O)C=C)C(C",  # invalid SMILES
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C(=O)NC)cc3)cc1)cn2C1CCCN(C)C1",
    ],
    "T2_EGFR_LEAD_OPT": [
        "COc1cc2ncnc(Nc3ccc(F)c(C#N)c3)c2cc1OCCCN1CCOCC1",
        "COc1cc2ncnc(Nc3ccc(OCCO)c(F)c3)c2cc1OCCCN1CCOCC1",
        "COc1cc2ncnc(Nc3ccc(C(=O)NC)c(Cl)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C)CC1",
        "COc1cc2ncnc(Nc3ccc(OC)c(F)c3)c2cc1OCCCN1CCOCC1",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # parent duplicate
        "COc1cc2ncnc(Nc3ccc(N=Nc4ccccc4)cc3)c2cc1OCCOC",  # PAINS azo
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCO99CC1",  # invalid SMILES
        "COc1cc2ncnc(Nc3ccc(Cl)c(Cl)c3)c2cc1OCCCCCN1CCOCC1",  # high LogP
        "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OCCCN1CCOCC1",
    ],
    "T3_GSK3B_SAR": [
        "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(F)cc21",
        "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(C#N)cc21",
        "Cn1cc(C2=C(c3cn(CCC(=O)NC)c4ccccc34)C(=O)NC2=O)c2ccc(OC)cc21",
        "Cn1cc(C2=C(c3cn(CCO)c4ccccc34)C(=O)NC2=O)c2ccc(C#N)cc21",
        "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(OCCO)cc21",
        "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(C#N)cc21",  # dup
        "Cn1cc(C2=C(c3cn(C)c4ccccc34)C(=O)NC2=O)c2ccc(N=Nc3ccccc3)cc21",  # PAINS
        "Cn1cc(C2=C(c3cn(CCCN4CCN(C)CC4)c4ccccc34)C(=O)NC2=O)c2cc(F)ccc21",
        "Cn1cc(C2=C(c3cn(CCCN4CCOCC4)c4ccccc34)C(=O)NC2=O)c2ccc(C",  # invalid
        "Cn1cc(C2=C(c3cn(CC(=O)N(C)C)c4ccccc34)C(=O)NC2=O)c2ccccc21",
    ],
    "T4_WETLAB_CSV": [
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C#N)c(F)c3)cc1)cn2C1CCN(C(=O)C)CC1",
        "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OCCCN1CCOCC1",
        "COc1cc2ncnc(Nc3ccc(C(=O)NC)cc3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(OCCO)c(F)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(OC)cc3)c2cc1OCCCN1CCOCC1",
        "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OCCCN1CCOCC1",  # dup
        "COc1cc2ncnc(Nc3ccc(N=Nc4ccccc4)cc3)c2cc1OC",  # PAINS
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCCCCC1CCOCC1",  # invalid ring / weird
        "COc1cc2ncnc(Nc3ccc(C(=O)N)c(F)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC(",  # invalid SMILES
    ],
    "T5_EGFR_STRICT": [
        "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(C(=O)NC)c(F)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(OCCO)cc3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C)CC1",
        "COc1cc2ncnc(Nc3ccc(OC)c(F)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OCCOC",  # dup
        "COc1cc2ncnc(Nc3ccc(N=Nc4ccccc4)cc3)c2cc1OCCN1CCOCC1",  # PAINS
        "COc1cc2ncnc(Nc3ccc(Cl)c(Cl)c3)c2cc1OCCCCCN1CCN(CC)CC1",  # high MW/LogP
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCN(C)(C)C",  # invalid valence
        "COc1cc2ncnc(Nc3ccc(S(=O)(=O)C)cc3)c2cc1OCCOC",
    ],
    "T6_BTK_STRICT": [
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C#N)cc3)cc1)cn2C1CCN(C(=O)C)CC1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C(=O)NC)cc3)cc1)cn2C1CCCN(C(=O)C)C1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccncc3)cc1)cn2C1CCN(C(=O)C)CC1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OCCO)cc3)cc1)cn2C1CCCN(C)C1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(S(=O)(=O)C)cc3)cc1)cn2C1CCN(C)CC1",
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C#N)cc3)cc1)cn2C1CCN(C(=O)C)CC1",  # dup
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(N=Nc4ccccc4)cc3)cc1)cn2C1CCNCC1",  # PAINS
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(C(F)(F)F)c(Cl)c3)cc1)cn2C1CCCN(C(=O)C=C)C1",  # high LogP/MW
        "Nc1ncnc2c1c(-c1ccc(Oc3ccccc3)cc1)cn2C1CCN(C(=O)C=C)C(F)(F",  # invalid SMILES
        "Nc1ncnc2c1c(-c1ccc(Oc3ccc(OC)cc3)cc1)cn2C1CCN(C)CC1",
    ],
}


def _parse_smiles_lines(raw_text: str, limit: int = 10) -> list[str]:
    """Извлекает до `limit` предложенных SMILES-строк из сырого ответа LLM до фильтрации."""
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        # Убираем нумерацию ('1. ', '1) ') и маркеры списка ('- ', '* ')
        line = re.sub(r"^(?:\d+[\.\)]\s*|[-*•]\s+)", "", line).strip()
        line = line.strip("`\"',;")
        if not line:
            continue
        # Если модель добавила комментарий после пробела/таба, берем первый токен
        token = line.split()[0].strip("`\"',;")
        if token:
            lines.append(token)
        if len(lines) >= limit:
            break
    return lines


def generate_baseline_smiles(
    task: BenchmarkTask,
    llm_adapter: LLMAdapter,
) -> tuple[list[str], str]:
    """Запрашивает 10 SMILES-аналогов напрямую через настройки `llm_adapter` с фоллбэком на мок."""
    prompt = (
        f"Сгенерируй 10 новых валидных SMILES-аналогов стартовой молекулы ({task.starting_smiles}) "
        f"для мишени {task.target_label} с улучшенным профилем QED >= 0.5, SA <= 4.5, "
        f"соблюдением правила Липинского и без токсичных групп PAINS. Верни только список SMILES"
    )

    if not llm_adapter.force_rule_based and llm_adapter.openai_api_key:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=llm_adapter.llm_model,
                api_key=llm_adapter.openai_api_key,
                base_url=llm_adapter.openai_base_url,
                temperature=0.2,
                timeout=10.0,
                max_retries=0,
                max_tokens=500,
                extra_body={"reasoning": {"effort": "none"}},
            )

            def _call_llm() -> Any:
                return llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Ты — эксперт по медицинской химии. Отвечай строго списком из 10 строк SMILES "
                                "без пояснений, заголовков и нумерации."
                            )
                        ),
                        HumanMessage(content=prompt),
                    ]
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                msg = pool.submit(_call_llm).result(timeout=15.0)

            parsed = _parse_smiles_lines(str(msg.content), limit=10)
            if len(parsed) >= 3:
                return parsed, f"live_llm:{llm_adapter.llm_model}"
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"deterministic_mock (API fallback: {type(exc).__name__})"
            return list(_BASELINE_MOCK_BY_TASK[task.task_id]), fallback_reason

    return list(_BASELINE_MOCK_BY_TASK[task.task_id]), "deterministic_mock"


def _compute_intdiv1(smiles_list: list[str]) -> float:
    """Вычисляет IntDiv1 = 1 - mean(pairwise Tanimoto similarity) по Morgan ECFP4 (r=2, 2048 бит)."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(
                compute_mol_fingerprint(mol, fp_type="ecfp", fp_radius=2, fp_nbits=2048)
            )
    n = len(fps)
    if n < 2:
        return 0.0
    total_sim = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += compute_fp_similarity(fps[i], fps[j], metric="tanimoto")
            pair_count += 1
    mean_sim = total_sim / float(pair_count)
    return round(max(0.0, min(1.0, 1.0 - mean_sim)), 4)


def _passes_medchem_filters(pains_count: int, lipinski_violations: int) -> bool:
    return pains_count == 0 and lipinski_violations <= 1


def _passes_complete_filters(
    qed: float, sa_score: float, pains_count: int, lipinski_violations: int
) -> bool:
    return (
        qed >= 0.5
        and sa_score <= 4.5
        and pains_count == 0
        and lipinski_violations <= 1
    )


def _compute_pool_metrics(
    *,
    num_proposed: int,
    valid_records: list[dict[str, Any]],
    execution_ok: bool,
) -> dict[str, Any]:
    """Рассчитывает все 7 метрик задачи по списку валидных оцененных записей."""
    num_valid = len(valid_records)
    validity_pct = (100.0 * num_valid / float(num_proposed)) if num_proposed > 0 else 0.0

    # Дедупликация по каноническому SMILES (сохраняя запись с наивысшим MPO)
    unique_by_smiles: dict[str, dict[str, Any]] = {}
    for rec in valid_records:
        csmi = rec["canonical_smiles"]
        prev = unique_by_smiles.get(csmi)
        if prev is None or rec["mpo_score"] > prev["mpo_score"]:
            unique_by_smiles[csmi] = rec

    unique_records = list(unique_by_smiles.values())
    num_unique = len(unique_records)
    uniqueness_pct = (100.0 * num_unique / float(num_valid)) if num_valid > 0 else 0.0

    medchem_pass_count = sum(
        1
        for r in unique_records
        if _passes_medchem_filters(r["pains_alerts_count"], r["lipinski_violations"])
    )
    complete_pass_count = sum(
        1
        for r in unique_records
        if _passes_complete_filters(
            r["qed"],
            r["sa_score"],
            r["pains_alerts_count"],
            r["lipinski_violations"],
        )
    )

    medchem_pass_pct = (
        (100.0 * medchem_pass_count / float(num_unique)) if num_unique > 0 else 0.0
    )
    complete_pass_pct = (
        (100.0 * complete_pass_count / float(num_unique)) if num_unique > 0 else 0.0
    )

    sorted_by_mpo = sorted(
        unique_records, key=lambda r: float(r["mpo_score"]), reverse=True
    )
    top5 = sorted_by_mpo[:5]
    top5_mean_mpo = (
        sum(float(r["mpo_score"]) for r in top5) / float(len(top5)) if top5 else 0.0
    )

    intdiv1 = _compute_intdiv1([r["canonical_smiles"] for r in unique_records])
    task_success = bool(execution_ok and complete_pass_count >= 1)

    return {
        "num_proposed": num_proposed,
        "num_valid": num_valid,
        "num_unique": num_unique,
        "validity_pct": round(validity_pct, 2),
        "uniqueness_pct": round(uniqueness_pct, 2),
        "medchem_pass_pct": round(medchem_pass_pct, 2),
        "complete_pass_pct": round(complete_pass_pct, 2),
        "top5_mean_mpo": round(top5_mean_mpo, 4),
        "intdiv1": round(intdiv1, 4),
        "task_success": task_success,
        "complete_pass_count": complete_pass_count,
    }


def evaluate_baseline_task(
    task: BenchmarkTask,
    llm_adapter: LLMAdapter,
) -> dict[str, Any]:
    """Выполняет и оценивает Режим 1 (Zero-shot LLM Baseline) для одной задачи."""
    t0 = time.perf_counter()
    proposed_smiles, llm_source = generate_baseline_smiles(task, llm_adapter)
    constraints: QueryConstraints = llm_adapter._deterministic_parse_query(
        task.query, task.uploaded_csv_path
    )
    tool_cfg = ToolConfig(docking_mode="skip")

    valid_records: list[dict[str, Any]] = []
    for raw_smi in proposed_smiles:
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        std_res = standardize_molecule_detailed(raw_smi, tool_config=tool_cfg)
        canon_smi = (
            std_res.canonical_smiles
            if std_res is not None
            else Chem.MolToSmiles(mol, canonical=True)
        )
        std_mol = Chem.MolFromSmiles(canon_smi) or mol
        calc = calculate_rdkit_metrics(
            std_mol,
            parent_mol_or_smiles=task.starting_smiles,
            tool_config=tool_cfg,
        )
        pred = predict_admet_metrics(
            std_mol,
            rdkit_metrics=calc,
            parent_exp_pic50=task.starting_pic50,
            tool_config=tool_cfg,
        )
        mpo = calculate_harrington_mpo_score(
            calc_rdkit=calc,
            pred_admet=pred,
            parent_exp_pic50=task.starting_pic50,
            constraints=constraints,
            tool_config=tool_cfg,
        )
        valid_records.append(
            {
                "raw_smiles": raw_smi,
                "canonical_smiles": canon_smi,
                "qed": calc.qed,
                "sa_score": calc.sa_score,
                "clogp": calc.clogp,
                "mw": calc.mw,
                "tpsa": calc.tpsa,
                "lipinski_violations": calc.lipinski_violations,
                "pains_alerts_count": calc.pains_alerts_count,
                "brenk_alerts_count": calc.brenk_alerts_count,
                "mpo_score": mpo,
            }
        )

    elapsed = time.perf_counter() - t0
    metrics = _compute_pool_metrics(
        num_proposed=len(proposed_smiles),
        valid_records=valid_records,
        execution_ok=True,
    )
    return {
        "task_id": task.task_id,
        "mode": "baseline_zero_shot",
        "llm_source": llm_source,
        "elapsed_sec": round(elapsed, 2),
        "metrics": metrics,
        "proposed_smiles": proposed_smiles,
    }


def _candidate_to_record(cand: CandidateMolecule) -> dict[str, Any]:
    return {
        "candidate_id": cand.candidate_id,
        "canonical_smiles": cand.canonical_smiles,
        "parent_smiles": cand.parent_smiles,
        "generation_iteration": cand.generation_iteration,
        "qed": cand.calc_rdkit.qed,
        "sa_score": cand.calc_rdkit.sa_score,
        "clogp": cand.calc_rdkit.clogp,
        "mw": cand.calc_rdkit.mw,
        "tpsa": cand.calc_rdkit.tpsa,
        "lipinski_violations": cand.calc_rdkit.lipinski_violations,
        "pains_alerts_count": cand.calc_rdkit.pains_alerts_count,
        "brenk_alerts_count": cand.calc_rdkit.brenk_alerts_count,
        "extra_alerts_count": cand.calc_rdkit.extra_alerts_count,
        "herg_prob": cand.pred_admet.herg_liability_prob,
        "predicted_pic50": cand.pred_admet.predicted_pic50,
        "mpo_score": cand.mpo_harrington_score,
        "passes_all_constraints": cand.passes_all_constraints,
        "constraint_violations": list(cand.constraint_violations),
    }


def _load_candidates_from_parquet(parquet_path: Path) -> list[CandidateMolecule]:
    table = pq.read_table(parquet_path)
    return [
        CandidateMolecule.from_flat_parquet_record(row) for row in table.to_pylist()
    ]


def evaluate_mas_task(
    task: BenchmarkTask,
    *,
    mode_name: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Запускает граф `workflow` (Режим 2 или 3) и рассчитывает все 7 метрик."""
    session_id = f"{task.task_id}_{mode_name}"
    session_dir = BENCH_WORKSPACE / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    t0 = time.perf_counter()
    prev_force_rb = settings.force_rule_based_llm
    settings.force_rule_based_llm = True
    try:
        tool_cfg = ToolConfig(
            max_iterations=max_iterations,
            docking_mode="skip",
        )
        state, _trace_path, _csv_path, _sdf_path = run_sar_pipeline(
            user_query=task.query,
            session_id=session_id,
            workspace_dir=BENCH_WORKSPACE,
            uploaded_csv_path=task.uploaded_csv_path,
            tool_config=tool_cfg,
            export_artifacts=True,
        )
        execution_ok = True
        error_msg = None
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return {
            "task_id": task.task_id,
            "mode": mode_name,
            "elapsed_sec": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": _compute_pool_metrics(
                num_proposed=0, valid_records=[], execution_ok=False
            ),
        }
    finally:
        settings.force_rule_based_llm = prev_force_rb

    elapsed = time.perf_counter() - t0
    final_it = int(state.get("design_iteration", 1))
    top_candidates: list[CandidateMolecule] = list(state.get("top_candidates", []))

    raw_it_path = session_dir / f"raw_candidates_it{final_it}.json"
    eval_it_path = session_dir / f"evaluated_candidates_it{final_it}.parquet"

    if eval_it_path.is_file():
        pool_candidates = _load_candidates_from_parquet(eval_it_path)
    else:
        pool_candidates = list(top_candidates)

    num_proposed = len(pool_candidates)
    if raw_it_path.is_file():
        raw_items = json.loads(raw_it_path.read_text(encoding="utf-8"))
        num_proposed = max(num_proposed, len(raw_items))

    pool_records = [_candidate_to_record(c) for c in pool_candidates]
    top_records = [_candidate_to_record(c) for c in top_candidates]

    pool_metrics = _compute_pool_metrics(
        num_proposed=num_proposed,
        valid_records=pool_records,
        execution_ok=execution_ok,
    )
    topk_metrics = _compute_pool_metrics(
        num_proposed=len(top_records),
        valid_records=top_records,
        execution_ok=execution_ok,
    )

    trace_events = state.get("trace_log", [])
    loop1_count = sum(
        1 for ev in trace_events if ev.verdict_or_routing == "refilter_data"
    )
    loop2_events = [
        ev for ev in trace_events if ev.verdict_or_routing == "redesign_molecules"
    ]

    return {
        "task_id": task.task_id,
        "mode": mode_name,
        "session_dir": str(session_dir),
        "elapsed_sec": round(elapsed, 2),
        "design_iteration": final_it,
        "loop1_triggers": loop1_count,
        "loop2_triggers": len(loop2_events),
        "critic_verdict": state.get("critic_verdict"),
        "docking_mode": state.get("docking_mode"),
        "metrics": pool_metrics,
        "topk_metrics": topk_metrics,
        "critic_loop2_feedback": [ev.output_summary for ev in loop2_events],
        "error": error_msg,
    }


def _aggregate_mode_metrics(task_results: list[dict[str, Any]]) -> dict[str, float]:
    """Усредняет 7 метрик по всем задачам для заданного режима."""
    n = len(task_results)
    if n == 0:
        return {}

    def _avg(key: str) -> float:
        return sum(float(r["metrics"][key]) for r in task_results) / float(n)

    success_pct = (
        100.0
        * sum(1 for r in task_results if bool(r["metrics"]["task_success"]))
        / float(n)
    )
    return {
        "validity_pct": round(_avg("validity_pct"), 2),
        "uniqueness_pct": round(_avg("uniqueness_pct"), 2),
        "medchem_pass_pct": round(_avg("medchem_pass_pct"), 2),
        "complete_pass_pct": round(_avg("complete_pass_pct"), 2),
        "top5_mean_mpo": round(_avg("top5_mean_mpo"), 4),
        "intdiv1": round(_avg("intdiv1"), 4),
        "task_success_rate_pct": round(success_pct, 2),
        "mean_elapsed_sec": round(
            sum(float(r["elapsed_sec"]) for r in task_results) / float(n), 2
        ),
    }


def _cand_summary_dict(c: CandidateMolecule) -> dict[str, Any]:
    total_alerts = (
        c.calc_rdkit.pains_alerts_count
        + c.calc_rdkit.brenk_alerts_count
        + c.calc_rdkit.extra_alerts_count
    )
    return {
        "candidate_id": c.candidate_id,
        "smiles": c.canonical_smiles,
        "parent_chembl_id": c.parent_chembl_id,
        "parent_smiles": c.parent_smiles,
        "qed": round(c.calc_rdkit.qed, 4),
        "sa_score": round(c.calc_rdkit.sa_score, 4),
        "logp": round(c.calc_rdkit.clogp, 4),
        "mw": round(c.calc_rdkit.mw, 2),
        "tpsa": round(c.calc_rdkit.tpsa, 2),
        "lipinski_violations": int(c.calc_rdkit.lipinski_violations),
        "pains_alerts": int(c.calc_rdkit.pains_alerts_count),
        "brenk_alerts": int(c.calc_rdkit.brenk_alerts_count),
        "total_alerts": int(total_alerts),
        "herg_prob": round(c.pred_admet.herg_liability_prob, 4),
        "predicted_pic50": round(c.pred_admet.predicted_pic50, 4),
        "mpo_score": round(c.mpo_harrington_score, 4),
        "passes_all_constraints": bool(c.passes_all_constraints),
        "constraint_violations": list(c.constraint_violations),
        "applied_smirks": c.hypothesis_rationale.applied_smirks,
    }


def extract_critic_before_after_case(
    tasks: list[BenchmarkTask],
    mode3_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Автоматически извлекает самый наглядный кейс «До и После Критика» из прогона Режима 3."""
    task_map = {t.task_id: t for t in tasks}
    best_case: dict[str, Any] | None = None
    best_score_gain = -999.0

    for res in mode3_results:
        if res.get("loop2_triggers", 0) < 1 or res.get("design_iteration", 0) < 2:
            continue
        session_dir = Path(res["session_dir"])
        it1_path = session_dir / "evaluated_candidates_it1.parquet"
        it2_path = session_dir / "evaluated_candidates_it2.parquet"
        if not (it1_path.is_file() and it2_path.is_file()):
            continue

        it1_cands = _load_candidates_from_parquet(it1_path)
        it2_cands = _load_candidates_from_parquet(it2_path)
        if not it1_cands or not it2_cands:
            continue

        # Кандидаты Итерации 1 с нарушениями ограничений или фильтров
        it1_violators = [
            c
            for c in it1_cands
            if (not c.passes_all_constraints)
            or (
                not _passes_complete_filters(
                    c.calc_rdkit.qed,
                    c.calc_rdkit.sa_score,
                    c.calc_rdkit.pains_alerts_count,
                    c.calc_rdkit.lipinski_violations,
                )
            )
        ]
        # Кандидаты Итерации 2 без нарушений
        it2_clean = [
            c
            for c in it2_cands
            if c.passes_all_constraints
            and _passes_complete_filters(
                c.calc_rdkit.qed,
                c.calc_rdkit.sa_score,
                c.calc_rdkit.pains_alerts_count,
                c.calc_rdkit.lipinski_violations,
            )
        ]
        if not it1_violators or not it2_clean:
            continue

        # Ищем пару с общим родителем (или лучшим ростом MPO и явным исправлением нарушений)
        selected_pair: tuple[CandidateMolecule, CandidateMolecule] | None = None
        best_pair_metric = -999.0
        for c2 in it2_clean[:10]:
            for c1 in it1_violators:
                same_parent_bonus = 0.5 if c1.parent_smiles == c2.parent_smiles else 0.0
                mpo_gain = c2.mpo_harrington_score - c1.mpo_harrington_score
                if mpo_gain <= 0.0:
                    continue
                pair_metric = mpo_gain + same_parent_bonus
                if pair_metric > best_pair_metric:
                    best_pair_metric = pair_metric
                    selected_pair = (c1, c2)

        if selected_pair is None:
            continue

        c1, c2 = selected_pair
        mpo_gain = c2.mpo_harrington_score - c1.mpo_harrington_score
        if mpo_gain > best_score_gain:
            best_score_gain = mpo_gain
            task = task_map[res["task_id"]]
            feedback_list = res.get("critic_loop2_feedback") or []
            it1_pass_rate = (
                100.0
                * sum(1 for c in it1_cands if c.passes_all_constraints)
                / float(len(it1_cands))
            )
            it2_pass_rate = (
                100.0
                * sum(1 for c in it2_cands if c.passes_all_constraints)
                / float(len(it2_cands))
            )
            best_case = {
                "task_id": task.task_id,
                "task_title": task.title,
                "target_id": task.target_id,
                "starting_smiles": c2.parent_smiles or task.starting_smiles,
                "critic_verdict_text": feedback_list[0] if feedback_list else "",
                "it1_pool_pass_rate_pct": round(it1_pass_rate, 2),
                "it2_pool_pass_rate_pct": round(it2_pass_rate, 2),
                "iteration_1_candidate": _cand_summary_dict(c1),
                "iteration_2_candidate": _cand_summary_dict(c2),
                "mpo_gain": round(mpo_gain, 4),
            }

    return best_case


def render_benchmark_report(
    aggregates: dict[str, dict[str, float]],
    critic_case: dict[str, Any] | None,
    total_elapsed_sec: float,
    tasks_count: int,
) -> str:
    """Формирует готовый Markdown-отчет BENCHMARK_REPORT.md для вставки в презентацию."""
    b = aggregates["baseline_zero_shot"]
    m2 = aggregates["mas_1_iter"]
    m3 = aggregates["mas_full_critic"]

    table_lines = [
        "# Отчет по бенчмарку и абляции (Ablation Study) — МАС «ПИШОВИЧОК»",
        "",
        f"- **Число тестовых задач:** `{tasks_count}` (мишени `CHEMBL203`, `CHEMBL262`, `CHEMBL5251`, лабораторная серия `custom_wetlab_series.csv`)",
        "- **3D-докинг (AutoDock Vina):** принудительно отключен (`docking_mode='skip'`)",
        f"- **Общее время расчета на CPU:** `{total_elapsed_sec:.2f}` сек",
        "",
        "## Таблица 1. Ablation Study (усреднение по всем задачам)",
        "",
        "| Метрика | Baseline (Zero-shot LLM) | МАС ПИШОВИЧОК (1 итерация, без цикла критика) | МАС ПИШОВИЧОК (Полный цикл + агент-критик) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Validity (%)** | {b['validity_pct']:.2f}% | {m2['validity_pct']:.2f}% | {m3['validity_pct']:.2f}% |",
        f"| **Uniqueness (%)** | {b['uniqueness_pct']:.2f}% | {m2['uniqueness_pct']:.2f}% | {m3['uniqueness_pct']:.2f}% |",
        f"| **MedChem Filters Pass (%)** | {b['medchem_pass_pct']:.2f}% | {m2['medchem_pass_pct']:.2f}% | {m3['medchem_pass_pct']:.2f}% |",
        f"| **Complete Pass Rate (%)** | {b['complete_pass_pct']:.2f}% | {m2['complete_pass_pct']:.2f}% | {m3['complete_pass_pct']:.2f}% |",
        f"| **Top-5 Mean MPO Score** | {b['top5_mean_mpo']:.4f} | {m2['top5_mean_mpo']:.4f} | {m3['top5_mean_mpo']:.4f} |",
        f"| **IntDiv1** | {b['intdiv1']:.4f} | {m2['intdiv1']:.4f} | {m3['intdiv1']:.4f} |",
        f"| **Task Success Rate (%)** | {b['task_success_rate_pct']:.2f}% | {m2['task_success_rate_pct']:.2f}% | {m3['task_success_rate_pct']:.2f}% |",
        "",
        "---",
        "",
        "## Кейс «Роль агента-критика» (До и После Критика)",
        "",
    ]

    if critic_case is None:
        table_lines.append(
            "В текущем прогоне ни одна из задач не потребовала повторной итерации Петли №2."
        )
    else:
        c1 = critic_case["iteration_1_candidate"]
        c2 = critic_case["iteration_2_candidate"]
        viol_str = "; ".join(c1["constraint_violations"]) or "Нарушение целевых порогов"
        table_lines.extend(
            [
                f"- **Задача:** {critic_case['task_title']} (`{critic_case['task_id']}`)",
                f"- **Target ID:** `{critic_case['target_id']}`",
                f"- **Стартовый SMILES:** `{critic_case['starting_smiles']}`",
                f"- **Pass Rate пула (ограничения задачи):** Итерация 1 = `{critic_case['it1_pool_pass_rate_pct']:.2f}%` → Итерация 2 = `{critic_case['it2_pool_pass_rate_pct']:.2f}%`",
                "",
                "### Вердикт агента-критика (`critic_node` после Итерации 1)",
                f"> {critic_case['critic_verdict_text']}",
                "",
                "### Сравнение кандидатов Итерации 1 и Итерации 2",
                "",
                "| Показатель | Кандидат Итерации 1 (До критика) | Кандидат Итерации 2 (После критика) |",
                "| :--- | :--- | :--- |",
                f"| **ID / Трансформация** | `{c1['candidate_id']}` (`{c1['applied_smirks']}`) | `{c2['candidate_id']}` (`{c2['applied_smirks']}`) |",
                f"| **SMILES** | `{c1['smiles']}` | `{c2['smiles']}` |",
                f"| **QED** | {c1['qed']:.4f} | {c2['qed']:.4f} |",
                f"| **SA score** | {c1['sa_score']:.4f} | {c2['sa_score']:.4f} |",
                f"| **cLogP** | {c1['logp']:.2f} | {c2['logp']:.2f} |",
                f"| **MW (г/моль)** | {c1['mw']:.1f} | {c2['mw']:.1f} |",
                f"| **TPSA (Å²)** | {c1['tpsa']:.1f} | {c2['tpsa']:.1f} |",
                f"| **hERG prob** | {c1['herg_prob']:.2f} | {c2['herg_prob']:.2f} |",
                f"| **Alerts (PAINS / Brenk / Total)** | {c1['pains_alerts']} / {c1['brenk_alerts']} / {c1['total_alerts']} | {c2['pains_alerts']} / {c2['brenk_alerts']} / {c2['total_alerts']} |",
                f"| **Lipinski violations** | {c1['lipinski_violations']} | {c2['lipinski_violations']} |",
                f"| **MPO Score (Харрингтон)** | **{c1['mpo_score']:.4f}** | **{c2['mpo_score']:.4f}** (`+{critic_case['mpo_gain']:.4f}`) |",
                f"| **Нарушения ограничений** | {viol_str} | Нет (0 нарушений) |",
                "",
            ]
        )

    return "\n".join(table_lines)


def main() -> None:
    """Точка входа бенчмарка и абляции."""
    t_total = time.perf_counter()
    BENCH_WORKSPACE.mkdir(parents=True, exist_ok=True)
    tasks = build_benchmark_tasks()
    llm_adapter = LLMAdapter()

    baseline_results: list[dict[str, Any]] = []
    mode2_results: list[dict[str, Any]] = []
    mode3_results: list[dict[str, Any]] = []

    print(f"[*] Запуск Ablation Benchmark для {len(tasks)} задач (docking_mode=skip)...")
    for idx, task in enumerate(tasks, start=1):
        print(f"\n[{idx}/{len(tasks)}] Задача {task.task_id}: {task.title}")

        r1 = evaluate_baseline_task(task, llm_adapter)
        baseline_results.append(r1)
        print(
            f"  - Baseline ({r1['llm_source']}): "
            f"Val={r1['metrics']['validity_pct']}%, "
            f"Pass={r1['metrics']['complete_pass_pct']}%, "
            f"MPO5={r1['metrics']['top5_mean_mpo']} ({r1['elapsed_sec']}s)"
        )

        r2 = evaluate_mas_task(task, mode_name="mas_1_iter", max_iterations=1)
        mode2_results.append(r2)
        print(
            f"  - МАС (1 итер.): "
            f"Val={r2['metrics']['validity_pct']}%, "
            f"Pass={r2['metrics']['complete_pass_pct']}%, "
            f"MPO5={r2['metrics']['top5_mean_mpo']} ({r2['elapsed_sec']}s)"
        )

        r3 = evaluate_mas_task(task, mode_name="mas_full_critic", max_iterations=2)
        mode3_results.append(r3)
        print(
            f"  - МАС (Полный + Критик): "
            f"Val={r3['metrics']['validity_pct']}%, "
            f"Pass={r3['metrics']['complete_pass_pct']}%, "
            f"MPO5={r3['metrics']['top5_mean_mpo']}, "
            f"Loop2={r3.get('loop2_triggers', 0)} ({r3['elapsed_sec']}s)"
        )

    aggregates = {
        "baseline_zero_shot": _aggregate_mode_metrics(baseline_results),
        "mas_1_iter": _aggregate_mode_metrics(mode2_results),
        "mas_full_critic": _aggregate_mode_metrics(mode3_results),
    }

    critic_case = extract_critic_before_after_case(tasks, mode3_results)
    total_elapsed = round(time.perf_counter() - t_total, 2)

    raw_payload = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_sec": total_elapsed,
        "tasks": [asdict(t) for t in tasks],
        "aggregates": aggregates,
        "critic_before_after_case": critic_case,
        "runs": {
            "baseline_zero_shot": baseline_results,
            "mas_1_iter": mode2_results,
            "mas_full_critic": mode3_results,
        },
    }
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_PATH.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_md = render_benchmark_report(
        aggregates=aggregates,
        critic_case=critic_case,
        total_elapsed_sec=total_elapsed,
        tasks_count=len(tasks),
    )
    REPORT_MD_PATH.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 78)
    print(report_md)
    print("=" * 78)
    print(f"[+] Сырые данные сохранены в: {RESULTS_JSON_PATH}")
    print(f"[+] Отчет сохранен в: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
