"""Общие фикстуры pytest для детерминированного и автономного выполнения тестов."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

from pishovichok.config import settings


@pytest.fixture(autouse=True)
def _force_offline_rule_based_llm_in_tests() -> Generator[None, None, None]:
    """Гарантирует воспроизводимое и быстрое оффлайн-выполнение LLM-узлов в pytest."""
    if os.environ.get("PISHOVICHOK_TEST_LIVE_LLM") == "1":
        yield
        return

    prev = settings.force_rule_based_llm
    settings.force_rule_based_llm = True
    try:
        yield
    finally:
        settings.force_rule_based_llm = prev
