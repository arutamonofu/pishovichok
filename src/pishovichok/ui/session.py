"""Серверный реестр вычислительных сессий и шлюзов Human-in-the-Loop (Слой 4).

Жизненный цикл графа (`asyncio.Task`), состояние `SARGraphState`, View-Model
и шлюз паузы HITL (`asyncio.Event`) живут в `SessionRegistry` на сервере
независимо от контекста `@ui.page` и переживают перезагрузку вкладки (`F5`).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nicegui import run
from pydantic import BaseModel, Field

from pishovichok.config import settings
from pishovichok.state.graph_state import SARGraphState, create_initial_graph_state
from pishovichok.state.schemas import (
    ActivityCliff,
    CandidateMolecule,
    QueryConstraints,
    SARRule,
    TraceEvent,
)
from pishovichok.state.storage import create_session_workspace
from pishovichok.ui.state_vm import ReferencePointVM, StateViewModel

HITLMode = Literal["auto", "pause_after_scaffolds", "pause_before_critic"]
SessionStatus = Literal[
    "idle", "running", "paused_hitl", "completed", "cancelled", "error"
]


class HITLDecision(BaseModel):
    """Решение исследователя в точке остановки Human-in-the-Loop."""

    action: Literal["approve", "restart_branch"] = "approve"
    target_stage: Literal["research", "design"] = "design"
    modified_lead_smiles: str | None = None
    modified_smarts: str | None = None
    constraints_patch: dict[str, Any] = Field(default_factory=dict)


@dataclass
class SessionRuntime:
    """Серверное состояние одной вычислительной сессии, независимое от вкладок браузера."""

    session_id: str
    workspace_dir: Path
    state: SARGraphState
    view_model: StateViewModel
    status: SessionStatus = "idle"
    current_node: str | None = None
    hitl_mode: HITLMode = "auto"
    hitl_pause_reason: str | None = None
    hitl_gate: asyncio.Event = field(default_factory=asyncio.Event)
    pending_decision: HITLDecision | None = None
    active_task: asyncio.Task[Any] | None = None
    subscribers: dict[str, Callable[[dict[str, Any]], Any]] = field(
        default_factory=dict
    )
    svg_cache: dict[tuple[str, str, int, int], str] = field(default_factory=dict)
    depth_mode: Literal["lab", "expert"] = "lab"
    active_scenario_key: str | None = None
    error_message: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def resolve_molecule_for_svg(
        self, mol_id: str, variant: str = "diff"
    ) -> tuple[str, str | None, str | None, list[int], bool, str] | None:
        """Возвращает параметры `(smiles, parent_smiles, scaffold_smiles, highlights, is_positive, legend)` для SVG."""
        if mol_id.startswith("SCAF_"):
            idx_str = mol_id.removeprefix("SCAF_")
            summary = self.state.get("curation_summary")
            if summary and idx_str.isdigit():
                idx = int(idx_str)
                if 0 <= idx < len(summary.top_scaffolds):
                    scaf_smi = str(summary.top_scaffolds[idx].get("scaffold_smiles", ""))
                    if scaf_smi:
                        return (scaf_smi, None, None, [], True, f"Scaffold #{idx + 1}")

        cand: CandidateMolecule | None = self.view_model.candidates_by_id.get(mol_id)
        if cand is None:
            for c in self.state.get("top_candidates", []):
                if c.candidate_id == mol_id:
                    cand = c
                    break
        if cand is None:
            lead_ref = self.state.get("lead_reference_metrics")
            if lead_ref is not None and lead_ref.candidate_id == mol_id:
                cand = lead_ref

        if cand is None:
            return None

        if variant == "parent":
            return (
                cand.parent_smiles,
                cand.canonical_smiles,
                cand.scaffold_smiles,
                [],
                True,
                "",
            )
        if variant == "scaffold":
            return (cand.scaffold_smiles, None, None, [], True, "Scaffold")
        if variant == "cand":
            return (
                cand.canonical_smiles,
                None,
                cand.scaffold_smiles,
                list(cand.highlighted_atom_indices),
                True,
                cand.candidate_id,
            )
        # По умолчанию 'diff': выровненная структура кандидата с подсветкой заменённого фрагмента
        is_positive = (
            cand.pred_admet.predicted_pic50 - cand.parent_exp_pic50 >= 0.0
        )
        return (
            cand.canonical_smiles,
            cand.parent_smiles,
            cand.scaffold_smiles,
            list(cand.highlighted_atom_indices),
            is_positive,
            "",
        )

    def resolve_brics_fragment(self, mol_id: str, frag_idx: int) -> str | None:
        """Возвращает SMILES фрагмента BRICS по индексу для ленивого SVG-рендера."""
        cand: CandidateMolecule | None = self.view_model.candidates_by_id.get(mol_id)
        if cand is None:
            return None
        frags = cand.calc_rdkit.brics_fragments
        if 0 <= frag_idx < len(frags):
            return frags[frag_idx]
        return None

    def resolve_rule_for_svg(self, rule_id: str) -> tuple[str, str, float] | None:
        """Возвращает `(from_fragment, to_fragment, mean_delta_pic50)` для 2D-превью правила SAR."""
        for rule in self.state.get("extracted_sar_rules", []):
            if getattr(rule, "rule_id", None) == rule_id:
                return (
                    str(rule.from_fragment),
                    str(rule.to_fragment),
                    float(rule.mean_delta_pic50),
                )
        return None


def _read_saved_session_sync(
    session_id: str,
    base_workspace_dir: Path,
) -> dict[str, Any] | None:
    """Синхронное чтение сохранённой сессии из `workspace/<session_id>/` (для `run.io_bound`)."""
    sess_dir = (base_workspace_dir / session_id).resolve()
    if not sess_dir.is_dir():
        return None

    import duckdb

    trace_file = sess_dir / "trace_log.json"
    candidates_file = sess_dir / "evaluated_candidates.parquet"
    curated_file = sess_dir / "curated_dataset.parquet"
    mmp_file = sess_dir / "mmp_pairs.parquet"
    report_file = sess_dir / "report.md"

    trace_events: list[TraceEvent] = []
    if trace_file.is_file():
        raw_trace = json.loads(trace_file.read_text(encoding="utf-8"))
        trace_events = [TraceEvent.model_validate(item) for item in raw_trace]

    candidates: list[CandidateMolecule] = []
    if candidates_file.is_file():
        cand_rows = (
            duckdb.connect()
            .sql(f"SELECT * FROM '{candidates_file.as_posix()}'")
            .to_arrow_table()
            .to_pylist()
        )
        for row in cand_rows:
            candidates.append(CandidateMolecule.from_flat_parquet_record(row))

    ref_points: list[ReferencePointVM] = []
    if curated_file.is_file():
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            cur_rows = (
                duckdb.connect()
                .sql(f"SELECT * FROM '{curated_file.as_posix()}' LIMIT 250")
                .to_arrow_table()
                .to_pylist()
            )
            for row in cur_rows:
                smi = str(row.get("canonical_smiles") or "")
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None:
                    continue
                assay_t = str(row.get("assay_type") or "")
                src_grp: Literal["chembl", "wetlab", "lead"] = (
                    "wetlab" if assay_t == "CUSTOM_WETLAB" else "chembl"
                )
                ref_points.append(
                    ReferencePointVM(
                        mol_id=str(row.get("mol_id") or ""),
                        canonical_smiles=smi,
                        pchembl_value=float(row.get("pchembl_value") or 0.0),
                        clogp=round(float(Descriptors.MolLogP(mol)), 4),
                        tpsa=round(float(Descriptors.TPSA(mol)), 4),
                        source_group=src_grp,
                    )
                )
        except Exception:  # noqa: BLE001
            ref_points = []

    sar_rules: list[SARRule] = []
    activity_cliffs: list[ActivityCliff] = []
    mmp_pairs_preview: list[dict[str, Any]] = []
    if mmp_file.is_file():
        try:
            conn = duckdb.connect()
            mmp_posix = mmp_file.as_posix()
            agg_sql = f"""
                SELECT
                    smirks_transformation,
                    from_fragment,
                    to_fragment,
                    COUNT(*) AS support_pairs_count,
                    AVG(delta_pic50) AS mean_delta_pic50,
                    COALESCE(STDDEV_SAMP(delta_pic50), 0.0) AS std_delta_pic50,
                    AVG(delta_clogp) AS mean_delta_clogp,
                    AVG(delta_tpsa) AS mean_delta_tpsa,
                    BOOL_AND(pair_assay_homogeneous) AS all_pairs_homog,
                    LIST_SORT(LIST_DISTINCT(LIST_CONCAT(LIST(bao_lhs), LIST(bao_rhs)))) AS bao_formats,
                    LIST(pair_id) AS pair_ids
                FROM read_parquet('{mmp_posix}')
                GROUP BY smirks_transformation, from_fragment, to_fragment
                ORDER BY support_pairs_count DESC, mean_delta_pic50 DESC
                LIMIT 25
            """
            for idx_r, rec in enumerate(conn.sql(agg_sql).to_arrow_table().to_pylist(), start=1):
                bao_list = [str(b) for b in (rec.get("bao_formats") or [])]
                sar_rules.append(
                    SARRule(
                        rule_id=f"R{idx_r}",
                        smirks=str(rec["smirks_transformation"]),
                        from_fragment=str(rec["from_fragment"]),
                        to_fragment=str(rec["to_fragment"]),
                        mean_delta_pic50=round(float(rec["mean_delta_pic50"]), 4),
                        std_delta_pic50=round(max(0.0, float(rec["std_delta_pic50"] or 0.0)), 4),
                        support_pairs_count=int(rec["support_pairs_count"]),
                        mean_delta_clogp=round(float(rec["mean_delta_clogp"] or 0.0), 4),
                        mean_delta_tpsa=round(float(rec["mean_delta_tpsa"] or 0.0), 4),
                        assay_homogeneous=bool(rec["all_pairs_homog"]) and len(bao_list) <= 1,
                        participating_bao_formats=bao_list or ["BAO_0000357"],
                        example_pair_ids=[str(p) for p in (rec.get("pair_ids") or [])][:5],
                    )
                )

            pair_rows = (
                conn.sql(f"SELECT * FROM read_parquet('{mmp_posix}') LIMIT 250")
                .to_arrow_table()
                .to_pylist()
            )
            cliff_idx = 0
            for prow in pair_rows:
                tanimoto = float(prow.get("tanimoto_similarity") or 0.0)
                abs_d = abs(float(prow.get("delta_pic50") or 0.0))
                sali = float(prow.get("sali_index") or 0.0)
                is_cliff = bool(prow.get("is_activity_cliff") or False)
                mmp_pairs_preview.append(
                    {
                        "pair_id": str(prow.get("pair_id") or ""),
                        "mol_id_lhs": str(prow.get("mol_id_lhs") or ""),
                        "mol_id_rhs": str(prow.get("mol_id_rhs") or ""),
                        "tanimoto": round(tanimoto, 4),
                        "abs_delta_pic50": round(abs_d, 4),
                        "sali_index": round(sali, 2),
                        "is_cliff": is_cliff,
                        "smirks": str(prow.get("smirks_transformation") or ""),
                    }
                )
                if is_cliff:
                    cliff_idx += 1
                    p_lhs = float(prow.get("pic50_lhs") or 0.0)
                    p_rhs = float(prow.get("pic50_rhs") or 0.0)
                    lhs_act = p_lhs >= p_rhs
                    activity_cliffs.append(
                        ActivityCliff(
                            cliff_id=f"CLIFF_{cliff_idx:02d}",
                            mol_id_active=str(prow["mol_id_lhs"] if lhs_act else prow["mol_id_rhs"]),
                            mol_id_inactive=str(prow["mol_id_rhs"] if lhs_act else prow["mol_id_lhs"]),
                            smiles_active=str(prow["smiles_lhs"] if lhs_act else prow["smiles_rhs"]),
                            smiles_inactive=str(prow["smiles_rhs"] if lhs_act else prow["smiles_lhs"]),
                            pic50_active=p_lhs if lhs_act else p_rhs,
                            pic50_inactive=p_rhs if lhs_act else p_lhs,
                            abs_delta_pic50=round(abs_d, 4),
                            tanimoto_similarity=round(tanimoto, 4),
                            sali_index=max(0.01, round(sali, 4)),
                            smirks_difference=str(prow.get("smirks_transformation") or ""),
                            assay_homogeneous=bool(prow.get("pair_assay_homogeneous", True)),
                        )
                    )
        except Exception:  # noqa: BLE001
            sar_rules = []
            activity_cliffs = []
            mmp_pairs_preview = []

    report_md = (
        report_file.read_text(encoding="utf-8") if report_file.is_file() else None
    )

    return {
        "session_dir": sess_dir,
        "trace_events": trace_events,
        "candidates": candidates,
        "ref_points": ref_points,
        "sar_rules": sar_rules,
        "activity_cliffs": activity_cliffs,
        "mmp_pairs_preview": mmp_pairs_preview,
        "report_md": report_md,
    }


class SessionRegistry:
    """Глобальный реестр активных сессий и UI-подписчиков."""

    def __init__(self, base_workspace_dir: Path | None = None) -> None:
        self._base_workspace_dir = (
            base_workspace_dir or settings.resolved_workspace_dir
        )
        self._sessions: dict[str, SessionRuntime] = {}
        self._registry_lock = asyncio.Lock()

    @property
    def base_workspace_dir(self) -> Path:
        return self._base_workspace_dir

    def get(self, session_id: str) -> SessionRuntime | None:
        """Возвращает существующую сессию из памяти (без блокировки)."""
        return self._sessions.get(session_id)

    def get_or_create(
        self,
        session_id: str | None = None,
        user_query: str = "",
        query_constraints: QueryConstraints | None = None,
    ) -> SessionRuntime:
        """Возвращает существующую или создаёт новую сессию в реестре."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        actual_id = session_id or f"ide_{uuid.uuid4().hex[:8]}"
        actual_id, session_dir = create_session_workspace(
            base_workspace_dir=self._base_workspace_dir,
            session_id=actual_id,
        )
        init_state = create_initial_graph_state(
            user_query=user_query,
            session_id=actual_id,
            workspace_dir=session_dir,
            query_constraints=query_constraints,
        )
        vm = StateViewModel(session_id=actual_id)
        if query_constraints is not None:
            vm.sync_constraints(query_constraints)

        runtime = SessionRuntime(
            session_id=actual_id,
            workspace_dir=session_dir,
            state=init_state,
            view_model=vm,
        )
        self._sessions[actual_id] = runtime
        return runtime

    async def restore_from_workspace(
        self, session_id: str
    ) -> SessionRuntime | None:
        """Асинхронно восстанавливает сессию из `workspace/<session_id>/` через `run.io_bound`."""
        payload = await run.io_bound(
            _read_saved_session_sync, session_id, self._base_workspace_dir
        )
        if payload is None:
            return None

        async with self._registry_lock:
            runtime = self.get_or_create(session_id=session_id)
            candidates: list[CandidateMolecule] = payload["candidates"]
            trace_events: list[TraceEvent] = payload["trace_events"]
            runtime.state["trace_log"] = trace_events
            runtime.state["top_candidates"] = candidates[:20]
            runtime.state["extracted_sar_rules"] = payload.get("sar_rules", [])
            runtime.state["detected_activity_cliffs"] = payload.get("activity_cliffs", [])
            runtime.state["final_report_markdown"] = payload["report_md"]
            runtime.state["critic_verdict"] = "approve" if candidates else None
            runtime.state["data_refilter_iteration"] = sum(
                1 for ev in trace_events if ev.verdict_or_routing == "refilter_data"
            )
            redesign_count = sum(
                1 for ev in trace_events if ev.verdict_or_routing == "redesign_molecules"
            )
            if redesign_count > 0:
                runtime.state["design_iteration"] = redesign_count + 1
            elif candidates:
                runtime.state["design_iteration"] = max(
                    (c.generation_iteration for c in candidates), default=1
                )
            runtime.status = "completed" if candidates else "idle"
            runtime.svg_cache.clear()
            runtime.view_model.clear_candidates()
            runtime.view_model.reference_points = payload["ref_points"]
            runtime.view_model.mmp_pairs_preview = payload.get("mmp_pairs_preview", [])
            runtime.view_model.upsert_candidates(candidates)
            return runtime

    def subscribe(
        self,
        session_id: str,
        client_id: str,
        callback: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Регистрирует UI-слушателя вкладки по `client_id`."""
        runtime = self.get(session_id)
        if runtime is not None:
            runtime.subscribers[client_id] = callback

    def unsubscribe(self, session_id: str, client_id: str) -> None:
        """Удаляет UI-слушателя при отключении вкладки (`client.on_disconnect`)."""
        runtime = self.get(session_id)
        if runtime is not None:
            runtime.subscribers.pop(client_id, None)

    async def broadcast(self, session_id: str, event: dict[str, Any]) -> None:
        """Рассылает событие всем активным подписчикам сессии."""
        runtime = self.get(session_id)
        if runtime is None:
            return
        items = list(runtime.subscribers.items())
        dead_clients: list[str] = []
        for cid, cb in items:
            try:
                res = cb(event)
                if inspect.isawaitable(res):
                    await res
            except Exception:  # noqa: BLE001
                dead_clients.append(cid)
        for cid in dead_clients:
            runtime.subscribers.pop(cid, None)

    def list_saved_sessions(self) -> list[str]:
        """Возвращает список сохранённых сессий в `workspace/`."""
        if not self._base_workspace_dir.is_dir():
            return []
        result: list[str] = []
        for entry in sorted(self._base_workspace_dir.iterdir()):
            if entry.is_dir() and (
                (entry / "evaluated_candidates.parquet").is_file()
                or (entry / "trace_log.json").is_file()
            ):
                result.append(entry.name)
        return result


session_registry = SessionRegistry()
