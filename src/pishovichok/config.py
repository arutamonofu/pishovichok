"""Конфигурация окружения, путей и пороговых констант системы (Слой 5).

Использует Pydantic Settings v2 для чтения переменных окружения и предоставления
детерминированных дефолтных значений, гарантирующих автономную работу без внешних сервисов.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    """Определяет корневую директорию репозитория относительно пакета src/pishovichok/."""
    return Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    """Глобальные настройки приложения pishovichok."""

    model_config = SettingsConfigDict(
        env_prefix="PISHOVICHOK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Пути к директориям данных и сессионного хранилища Fat Pointers
    project_root: Path = Field(default_factory=_default_project_root)
    data_dir: Path | None = Field(default=None)
    workspace_dir: Path | None = Field(default=None)

    # Параметры каскада LLM (Уровень 1: Облачный API -> Уровень 2: Ollama -> Уровень 3: Rule-Based)
    openai_api_key: str | None = Field(
        default=None, description="Ключ OpenAI-совместимого API"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Базовый URL OpenAI / DeepSeek / OpenRouter / vLLM",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="Имя языковой модели")
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Локальный эндпоинт Ollama для второго уровня фолбэка",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b", description="Имя локальной модели Ollama"
    )
    llm_timeout_seconds: float = Field(default=4.0, ge=0.5, le=60.0)
    force_rule_based_llm: bool = Field(
        default=False,
        description="Принудительное использование детерминированного Rule-Based движка (Уровень 3)",
    )

    # Лимиты итераций петель самокоррекции в графе LangGraph
    max_data_refilter_iterations: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Макс. число возвратов Петли №1 (Critic -> Research)",
    )
    max_design_iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Макс. число итераций дизайна Петли №2 (Critic -> Design)",
    )

    # Хемоинформатические константы и пороги (Слой 1)
    dominant_bao_default: str = Field(
        default="BAO_0000357",
        description="Стандартный формат биохимического теста на белке",
    )
    sali_tanimoto_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Порог сходства Танимото для детекции Activity Cliffs",
    )
    sali_delta_pic50_threshold: float = Field(
        default=1.5,
        ge=0.5,
        description="Минимальный перепад |ΔpIC50| для Activity Cliff",
    )
    sali_epsilon: float = Field(
        default=0.001,
        gt=0.0,
        description="Сглаживающая константа в знаменателе индекса SALI",
    )
    lead_neighborhood_tanimoto: float = Field(
        default=0.60,
        ge=0.1,
        le=1.0,
        description="Порог окрестности Танимото вокруг лидера (Сценарий 2)",
    )
    min_sar_rules_before_enrichment: int = Field(
        default=3,
        ge=1,
        description="Мин. число правил в CSV до обогащения фоном ChEMBL (Сценарий 3)",
    )
    default_top_k_candidates: int = Field(default=5, ge=1, le=20)
    max_candidates_pool: int = Field(default=30, ge=5, le=200)

    # Приоритет каскада ADMET
    preferred_admet_backend: Literal[
        "auto", "admet_ai_dmpnn", "catboost_local_qsar", "rdkit_analytical_qsar"
    ] = Field(default="auto")

    @property
    def resolved_data_dir(self) -> Path:
        return (
            self.data_dir if self.data_dir is not None else (self.project_root / "data")
        )

    @property
    def resolved_workspace_dir(self) -> Path:
        return (
            self.workspace_dir
            if self.workspace_dir is not None
            else (self.project_root / "workspace")
        )

    @property
    def chembl_cache_dir(self) -> Path:
        return self.resolved_data_dir / "chembl"

    @property
    def crem_db_path(self) -> Path:
        return self.resolved_data_dir / "crem" / "replacements_sa2.db"

    @property
    def models_dir(self) -> Path:
        return self.resolved_data_dir / "models"

    @property
    def samples_dir(self) -> Path:
        return self.resolved_data_dir / "samples"


settings = AppSettings()
