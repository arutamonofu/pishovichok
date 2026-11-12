"""Мост между асинхронным NiceGUI UI и графом агентов LangGraph (Слой 4).

Обеспечивает:
1. Запуск `agents/workflow.py` в фоне через `nicegui.background_tasks.create(...)`.
2. Пошаговую трансляцию событий узлов (`research -> sar -> design -> evaluation -> critic`)
   всем активным подписчикам сессии.
3. Механизм точек остановки Human-in-the-Loop (HITL) с ожиданием `await hitl_gate.wait()`
   и передачей модифицированных скаффолдов/SMARTS-ограничений обратно в граф.
4. Корректную отмену текущей ветви (`task.cancel()` + перехват `asyncio.CancelledError`)
   и перезапуск с обновлёнными параметрами.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from nicegui import background_tasks, run

from pishovichok.agents.workflow import (
    build_sar_graph,
    export_candidates_to_csv,
    export_candidates_to_sdf,
)
from pishovichok.core.rendering import compute_chemical_space_projection
from pishovichok.state.graph_state import SARGraphState, create_initial_graph_state
from pishovichok.state.schemas import QueryConstraints, SARRule
from pishovichok.state.storage import (
    create_session_workspace,
    export_trace_log,
    read_candidates_parquet,
)
from pishovichok.ui.session import (
    HITLDecision,
    HITLMode,
    SessionRegistry,
    SessionRuntime,
    session_registry,
)
from pishovichok.ui.state_vm import ReferencePointVM

_ALL_GRAPH_NODES = ["research", "sar", "design", "evaluation", "critic"]

_ALLOWED_CHECKPOINT_MODULES: list[tuple[str, str]] = [
    ("pishovichok.state.schemas", "ExperimentalParentFact"),
    ("pishovichok.state.schemas", "CalcRDKitMetrics"),
    ("pishovichok.state.schemas", "PredADMETMetrics"),
    ("pishovichok.state.schemas", "SARHypothesisRationale"),
    ("pishovichok.state.schemas", "CandidateMolecule"),
    ("pishovichok.state.schemas", "ParquetPointer"),
    ("pishovichok.state.schemas", "QueryConstraints"),
    ("pishovichok.state.schemas", "SARRule"),
    ("pishovichok.state.schemas", "ActivityCliff"),
    ("pishovichok.state.schemas", "DatasetCurationSummary"),
    ("pishovichok.state.schemas", "TraceEvent"),
    ("pishovichok.state.schemas", "ParentSeed"),
    ("pishovichok.state.schemas", "GeneratedRawCandidate"),
]


def _run_graph_chunk_sync(
    compiled_graph: Any,
    input_payload: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Выполняет один шаг графа LangGraph до следующей точки прерывания."""
    updates: list[dict[str, Any]] = []
    for chunk in compiled_graph.stream(input_payload, config, stream_mode="updates"):
        if isinstance(chunk, dict):
            updates.append(chunk)
    return updates


def _read_curated_reference_points_sync(
    curated_path: str,
) -> list[ReferencePointVM]:
    """Читает референсные соединения из `curated_dataset.parquet` для проекции химпространства."""
    import duckdb
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    path = Path(curated_path)
    if not path.is_file():
        return []
    df = (
        duckdb.connect()
        .execute(f"SELECT * FROM '{path.as_posix()}' LIMIT 250")
        .df()
    )
    points: list[ReferencePointVM] = []
    for _, row in df.iterrows():
        smi = str(row.get("canonical_smiles") or "")
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        assay_t = str(row.get("assay_type") or "")
        src_grp: Literal["chembl", "wetlab", "lead"] = (
            "wetlab" if assay_t == "CUSTOM_WETLAB" else "chembl"
        )
        points.append(
            ReferencePointVM(
                mol_id=str(row.get("mol_id") or ""),
                canonical_smiles=smi,
                pchembl_value=float(row.get("pchembl_value") or 0.0),
                clogp=round(float(Descriptors.MolLogP(mol)), 4),
                tpsa=round(float(Descriptors.TPSA(mol)), 4),
                source_group=src_grp,
            )
        )
    return points


def _export_session_artifacts_sync(
    state: SARGraphState,
    session_dir: Path,
) -> None:
    """Экспортирует артефакты сессии (`trace_log.json`, `candidates.csv`, `candidates.sdf`, `report.md`)."""
    export_trace_log(state.get("trace_log", []), session_dir)
    top_candidates = state.get("top_candidates", [])
    if top_candidates:
        export_candidates_to_csv(top_candidates, session_dir / "candidates.csv")
        export_candidates_to_sdf(top_candidates, session_dir / "candidates.sdf")
    report_md = state.get("final_report_markdown")
    if report_md:
        (session_dir / "report.md").write_text(report_md, encoding="utf-8")


def _apply_modifications_to_state(
    state: SARGraphState,
    modified_lead_smiles: str | None = None,
    modified_smarts: str | None = None,
    constraints_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Формирует словарь обновлений `SARGraphState` из правок исследователя (скаффолд/SMARTS/пороги)."""
    patch: dict[str, Any] = {}
    constraints = state["query_constraints"]
    c_updates: dict[str, Any] = dict(constraints_patch or {})

    if modified_lead_smiles and modified_lead_smiles.strip():
        c_updates["lead_smiles"] = modified_lead_smiles.strip()
        c_updates["freeze_scaffold"] = True
        if constraints.scenario_type == "target_based_sar":
            c_updates["scenario_type"] = "lead_optimization"

    if c_updates:
        constraints = constraints.model_copy(update=c_updates)
        patch["query_constraints"] = constraints

    if modified_smarts and ">>" in modified_smarts:
        lhs, rhs = modified_smarts.split(">>", 1)
        lhs_clean = lhs.strip()
        rhs_clean = rhs.strip()
        if lhs_clean and rhs_clean:
            user_rule = SARRule(
                rule_id="USER_SMARTS_01",
                smirks=f"{lhs_clean}>>{rhs_clean}",
                from_fragment=lhs_clean,
                to_fragment=rhs_clean,
                mean_delta_pic50=0.65,
                std_delta_pic50=0.0,
                support_pairs_count=1,
                mean_delta_clogp=-0.2,
                mean_delta_tpsa=10.0,
                assay_homogeneous=True,
                participating_bao_formats=["USER_CONSTRAINT"],
                example_pair_ids=[],
            )
            existing_rules = [
                r for r in state.get("extracted_sar_rules", [])
                if r.rule_id != "USER_SMARTS_01"
            ]
            patch["extracted_sar_rules"] = [user_rule, *existing_rules]

    return patch


class AgentUIBridge:
    """Асинхронный оркестратор запуска, HITL-пауз и отмены веток LangGraph для NiceGUI."""

    def __init__(self, registry: SessionRegistry | None = None) -> None:
        self.registry = registry or session_registry

    async def start_pipeline(
        self,
        session_id: str,
        user_query: str,
        *,
        uploaded_csv_path: str | None = None,
        hitl_mode: HITLMode = "auto",
        query_constraints: QueryConstraints | None = None,
    ) -> SessionRuntime:
        """Запускает граф агентов в фоне через `nicegui.background_tasks.create`."""
        runtime = self.registry.get_or_create(
            session_id=session_id,
            user_query=user_query,
            query_constraints=query_constraints,
        )
        await self._cancel_active_task(runtime)

        _, session_dir = create_session_workspace(
            base_workspace_dir=self.registry.base_workspace_dir,
            session_id=session_id,
        )
        runtime.workspace_dir = session_dir
        runtime.state = create_initial_graph_state(
            user_query=user_query,
            session_id=session_id,
            workspace_dir=session_dir,
            query_constraints=query_constraints,
            uploaded_csv_path=uploaded_csv_path,
        )
        runtime.hitl_mode = hitl_mode
        runtime.hitl_pause_reason = None
        runtime.pending_decision = None
        runtime.hitl_gate.clear()
        runtime.error_message = None
        runtime.svg_cache.clear()
        runtime.view_model.clear_candidates()
        runtime.view_model.reference_points = []
        if query_constraints is not None:
            runtime.view_model.sync_constraints(query_constraints)

        runtime.status = "running"
        runtime.current_node = "research"

        task = background_tasks.create(
            self._run_graph_loop(runtime, resume_from_stage=None),
            name=f"sar_graph_{session_id}",
        )
        runtime.active_task = task
        await self.registry.broadcast(
            session_id,
            {"type": "pipeline_started", "session_id": session_id, "node": "research"},
        )
        return runtime

    async def submit_hitl_decision(
        self,
        session_id: str,
        decision: HITLDecision,
    ) -> None:
        """Передаёт решение исследователя в ожидающий `hitl_gate`."""
        runtime = self.registry.get(session_id)
        if runtime is None:
            return
        runtime.pending_decision = decision
        runtime.hitl_gate.set()

    async def cancel_current_run(self, session_id: str) -> None:
        """Корректно отменяет текущую фоновую задачу графа."""
        runtime = self.registry.get(session_id)
        if runtime is None:
            return
        await self._cancel_active_task(runtime)
        runtime.status = "cancelled"
        runtime.current_node = None
        runtime.hitl_pause_reason = None
        await self.registry.broadcast(
            session_id,
            {"type": "pipeline_cancelled", "session_id": session_id},
        )

    async def restart_branch(
        self,
        session_id: str,
        *,
        target_stage: Literal["research", "design"] = "design",
        modified_lead_smiles: str | None = None,
        modified_smarts: str | None = None,
        constraints_patch: dict[str, Any] | None = None,
    ) -> None:
        """Отменяет текущую ветвь и перезапускает граф с модифицированными скаффолдами/SMARTS."""
        runtime = self.registry.get(session_id)
        if runtime is None:
            return

        await self._cancel_active_task(runtime)

        patch = _apply_modifications_to_state(
            runtime.state,
            modified_lead_smiles=modified_lead_smiles,
            modified_smarts=modified_smarts,
            constraints_patch=constraints_patch,
        )
        for k, v in patch.items():
            runtime.state[k] = v  # type: ignore[literal-required]

        if "query_constraints" in patch:
            runtime.view_model.sync_constraints(patch["query_constraints"])

        runtime.status = "running"
        runtime.hitl_pause_reason = None
        runtime.pending_decision = None
        runtime.hitl_gate.clear()
        runtime.error_message = None

        if target_stage == "design" and runtime.state.get("dataset_uri") is not None:
            runtime.state["design_iteration"] = 0
            runtime.state["critic_verdict"] = "proceed_to_design"
            runtime.view_model.clear_candidates()
            runtime.svg_cache.clear()
            resume_stage: Literal["research", "design"] | None = "design"
            runtime.current_node = "design"
        else:
            runtime.state["last_executed_stage"] = "init"
            runtime.state["data_refilter_iteration"] = 0
            runtime.state["design_iteration"] = 0
            runtime.state["assay_filter_strict"] = False
            runtime.state["active_bao_filter"] = None
            runtime.view_model.clear_candidates()
            runtime.svg_cache.clear()
            resume_stage = "research"
            runtime.current_node = "research"

        task = background_tasks.create(
            self._run_graph_loop(runtime, resume_from_stage=resume_stage),
            name=f"sar_graph_restart_{session_id}",
        )
        runtime.active_task = task
        await self.registry.broadcast(
            session_id,
            {
                "type": "branch_restarted",
                "session_id": session_id,
                "target_stage": target_stage,
            },
        )

    async def _cancel_active_task(self, runtime: SessionRuntime) -> None:
        task = runtime.active_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        runtime.active_task = None
        runtime.hitl_gate.set()

    async def _run_graph_loop(
        self,
        runtime: SessionRuntime,
        resume_from_stage: Literal["research", "design"] | None = None,
    ) -> None:
        """Основной цикл пошагового выполнения графа с поддержкой HITL и отмены."""
        session_id = runtime.session_id
        try:
            checkpointer = MemorySaver(
                serde=JsonPlusSerializer(
                    allowed_msgpack_modules=_ALLOWED_CHECKPOINT_MODULES,
                )
            )
            compiled_graph = build_sar_graph(
                checkpointer=checkpointer,
                interrupt_after=_ALL_GRAPH_NODES,
            )
            config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

            # Если перезапускаем ветвь со стадии `design`, предварительно записываем
            # актуальное состояние в checkpointer на входе в `design` (как после `critic`).
            if resume_from_stage == "design":
                await run.io_bound(
                    compiled_graph.update_state,
                    config,
                    dict(runtime.state),
                    "critic",
                )
                next_input: dict[str, Any] | None = None
            else:
                next_input = dict(runtime.state)

            while True:
                updates = await run.io_bound(
                    _run_graph_chunk_sync, compiled_graph, next_input, config
                )
                next_input = None

                if not updates:
                    break

                executed_node: str | None = None
                for chunk in updates:
                    for node_name, node_delta in chunk.items():
                        if node_name.startswith("__"):
                            continue
                        executed_node = node_name
                        if isinstance(node_delta, dict):
                            self._merge_state_delta(runtime.state, node_delta)

                if executed_node is None:
                    break

                await self._on_node_completed(runtime, executed_node)

                # Проверка точки остановки Human-in-the-Loop
                pause_reason = self._check_hitl_breakpoint(runtime, executed_node)
                if pause_reason is not None:
                    runtime.status = "paused_hitl"
                    runtime.hitl_pause_reason = pause_reason
                    runtime.pending_decision = None
                    runtime.hitl_gate.clear()
                    await self.registry.broadcast(
                        session_id,
                        {
                            "type": "hitl_paused",
                            "session_id": session_id,
                            "node": executed_node,
                            "reason": pause_reason,
                        },
                    )
                    await runtime.hitl_gate.wait()

                    decision = runtime.pending_decision or HITLDecision(action="approve")
                    runtime.hitl_pause_reason = None

                    if decision.action == "restart_branch":
                        # Перезапуск текущей ветви в том же цикле с правками пользователя
                        patch = _apply_modifications_to_state(
                            runtime.state,
                            modified_lead_smiles=decision.modified_lead_smiles,
                            modified_smarts=decision.modified_smarts,
                            constraints_patch=decision.constraints_patch,
                        )
                        for k, v in patch.items():
                            runtime.state[k] = v  # type: ignore[literal-required]
                        if "query_constraints" in patch:
                            runtime.view_model.sync_constraints(
                                patch["query_constraints"]
                            )
                        runtime.view_model.clear_candidates()
                        runtime.svg_cache.clear()
                        runtime.status = "running"

                        if (
                            decision.target_stage == "design"
                            and runtime.state.get("dataset_uri") is not None
                        ):
                            runtime.state["design_iteration"] = 0
                            runtime.state["critic_verdict"] = "proceed_to_design"
                            await run.io_bound(
                                compiled_graph.update_state,
                                config,
                                dict(runtime.state),
                                "critic",
                            )
                            next_input = None
                        else:
                            runtime.state["last_executed_stage"] = "init"
                            runtime.state["data_refilter_iteration"] = 0
                            runtime.state["design_iteration"] = 0
                            runtime.state["assay_filter_strict"] = False
                            runtime.state["active_bao_filter"] = None
                            config = {
                                "configurable": {
                                    "thread_id": f"{session_id}_re_{len(runtime.state.get('trace_log', []))}"
                                }
                            }
                            next_input = dict(runtime.state)
                        await self.registry.broadcast(
                            session_id,
                            {
                                "type": "branch_restarted",
                                "session_id": session_id,
                                "target_stage": decision.target_stage,
                            },
                        )
                        continue

                    # action == "approve": применяем правки ограничений/скаффолда и продолжаем граф
                    patch = _apply_modifications_to_state(
                        runtime.state,
                        modified_lead_smiles=decision.modified_lead_smiles,
                        modified_smarts=decision.modified_smarts,
                        constraints_patch=decision.constraints_patch,
                    )
                    if patch:
                        for k, v in patch.items():
                            runtime.state[k] = v  # type: ignore[literal-required]
                        if "query_constraints" in patch:
                            runtime.view_model.sync_constraints(
                                patch["query_constraints"]
                            )
                        await run.io_bound(
                            compiled_graph.update_state, config, patch
                        )

                    runtime.status = "running"
                    await self.registry.broadcast(
                        session_id,
                        {
                            "type": "hitl_resumed",
                            "session_id": session_id,
                            "node": executed_node,
                        },
                    )

                # Проверяем, завершён ли граф
                snap = await run.io_bound(compiled_graph.get_state, config)
                if not getattr(snap, "next", None):
                    break

            runtime.status = "completed"
            runtime.current_node = None
            await run.io_bound(
                _export_session_artifacts_sync, runtime.state, runtime.workspace_dir
            )
            await self.registry.broadcast(
                session_id,
                {"type": "pipeline_completed", "session_id": session_id},
            )
        except asyncio.CancelledError:
            runtime.status = "cancelled"
            runtime.current_node = None
            raise
        except Exception as exc:  # noqa: BLE001
            runtime.status = "error"
            runtime.error_message = f"{type(exc).__name__}: {exc}"
            runtime.current_node = None
            await self.registry.broadcast(
                session_id,
                {
                    "type": "pipeline_error",
                    "session_id": session_id,
                    "error": runtime.error_message,
                },
            )

    @staticmethod
    def _merge_state_delta(state: SARGraphState, delta: dict[str, Any]) -> None:
        """Объединяет частичный словарь обновлений узла LangGraph в `runtime.state`."""
        for key, val in delta.items():
            if key == "trace_log" and isinstance(val, list):
                existing = list(state.get("trace_log", []))
                existing.extend(val)
                state["trace_log"] = existing
            else:
                state[key] = val  # type: ignore[literal-required]

    @staticmethod
    def _check_hitl_breakpoint(
        runtime: SessionRuntime, executed_node: str
    ) -> str | None:
        """Определяет, нужно ли поставить граф на HITL-паузу после `executed_node`."""
        if runtime.hitl_mode == "auto":
            return None
        if (
            runtime.hitl_mode == "pause_after_scaffolds"
            and executed_node == "critic"
            and runtime.state.get("critic_verdict") == "proceed_to_design"
        ):
            return (
                "Пауза после извлечения скаффолдов и правил SAR (перед генерацией кандидатов). "
                "Вы можете скорректировать скаффолд в скетчере или утвердить переход к дизайну."
            )
        if (
            runtime.hitl_mode == "pause_before_critic"
            and executed_node == "evaluation"
        ):
            return (
                "Пауза после оценки кандидатов (перед проверкой Critic Agent). "
                "Изучите кандидатов в таблице и на Парето-фронте, затем утвердите или перезапустите ветвь."
            )
        return None

    async def _on_node_completed(
        self, runtime: SessionRuntime, executed_node: str
    ) -> None:
        """Обновляет View-Model, проекцию химпространства и рассылает событие узла."""
        session_id = runtime.session_id
        runtime.current_node = executed_node
        runtime.view_model.sync_constraints(runtime.state.get("query_constraints"))

        if executed_node == "research":
            dataset_ptr = runtime.state.get("dataset_uri")
            if dataset_ptr is not None:
                ref_pts = await run.io_bound(
                    _read_curated_reference_points_sync, dataset_ptr.uri
                )
                runtime.view_model.reference_points = ref_pts

        if executed_node == "evaluation":
            # Загружаем полный пул оценённых кандидатов (или top_candidates) в ViewModel
            cand_ptr = runtime.state.get("candidates_table_uri")
            if cand_ptr is not None:
                pool = await run.io_bound(read_candidates_parquet, cand_ptr)
            else:
                pool = list(runtime.state.get("top_candidates", []))

            lead_ref = runtime.state.get("lead_reference_metrics")
            new_ids = runtime.view_model.upsert_candidates(pool, lead_reference=lead_ref)

            # Расчёт 2D PCA проекции химпространства в cpu_bound (spawn)
            all_smiles: list[str] = []
            cand_ids: list[str] = []
            for cid in runtime.view_model.row_order:
                row_vm = runtime.view_model.rows_by_id[cid]
                cand_ids.append(cid)
                all_smiles.append(row_vm.canonical_smiles)
            ref_ids: list[str] = []
            for ref in runtime.view_model.reference_points:
                ref_ids.append(ref.mol_id)
                all_smiles.append(ref.canonical_smiles)
            has_lead_ref = lead_ref is not None
            if has_lead_ref and lead_ref is not None:
                all_smiles.append(lead_ref.canonical_smiles)

            if all_smiles:
                coords = await run.cpu_bound(
                    compute_chemical_space_projection, all_smiles
                )
                n_cands = len(cand_ids)
                cand_coord_map = {
                    cand_ids[i]: coords[i] for i in range(n_cands)
                }
                ref_coord_map = {
                    ref_ids[j]: coords[n_cands + j]
                    for j in range(len(ref_ids))
                }
                runtime.view_model.update_pca_coordinates(
                    cand_coord_map, ref_coord_map
                )
                if has_lead_ref and len(coords) > n_cands + len(ref_ids):
                    runtime.view_model.lead_pca_coords = coords[
                        n_cands + len(ref_ids)
                    ]

            await self.registry.broadcast(
                session_id,
                {
                    "type": "candidates_updated",
                    "session_id": session_id,
                    "new_ids": new_ids,
                },
            )

        await self.registry.broadcast(
            session_id,
            {
                "type": "node_completed",
                "session_id": session_id,
                "node": executed_node,
                "verdict": runtime.state.get("critic_verdict"),
            },
        )


agent_bridge = AgentUIBridge()
