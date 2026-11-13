"""Слой 4: графический интерфейс «ПИШОВИЧОК · SAR-дизайн» на базе NiceGUI 4.0."""

from __future__ import annotations

from pishovichok.ui.app import register_ui_assets_and_routes, run_app, run_ui
from pishovichok.ui.bridge import AgentUIBridge, agent_bridge
from pishovichok.ui.components.agent_rail import DEMO_SCENARIOS
from pishovichok.ui.session import (
    HITLDecision,
    HITLMode,
    SessionRegistry,
    SessionRuntime,
    session_registry,
)
from pishovichok.ui.state_vm import (
    CandidateRowVM,
    MPOWeightsAndThresholds,
    ReferencePointVM,
    StateViewModel,
)

__all__ = [
    "DEMO_SCENARIOS",
    "AgentUIBridge",
    "CandidateRowVM",
    "HITLDecision",
    "HITLMode",
    "MPOWeightsAndThresholds",
    "ReferencePointVM",
    "SessionRegistry",
    "SessionRuntime",
    "StateViewModel",
    "agent_bridge",
    "register_ui_assets_and_routes",
    "run_app",
    "run_ui",
    "session_registry",
]
