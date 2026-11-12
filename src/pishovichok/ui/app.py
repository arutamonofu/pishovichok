"""Точка входа NiceGUI 4.0 для трёхпанельной рабочей среды исследователя («Chemical IDE»)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import Request
from nicegui import Client, app, ui

from pishovichok.config import settings
from pishovichok.ui.components.agent_rail import AgentRail, TopStatusBar
from pishovichok.ui.components.candidate_grid import CandidateGrid
from pishovichok.ui.components.inspector import InspectorPane
from pishovichok.ui.components.mpo_canvas import MPOCanvas
from pishovichok.ui.components.tool_settings import ToolSettingsDrawer
from pishovichok.ui.routes import router as api_router
from pishovichok.ui.session import SessionRuntime, session_registry

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ASSETS_REGISTERED = False


def register_ui_assets_and_routes() -> None:
    """Регистрирует локальную статику `/static` (без CDN) и FastAPI-маршруты SVG."""
    global _ASSETS_REGISTERED
    if _ASSETS_REGISTERED:
        return
    if _STATIC_DIR.is_dir():
        app.add_static_files("/static", str(_STATIC_DIR))
    app.include_router(api_router)
    _ASSETS_REGISTERED = True


@ui.page("/")
async def chemical_ide_page(
    client: Client,
    request: Request,
    session_id: str | None = None,
) -> None:
    """Главная страница Chemical IDE с сохранением состояния графа при перезагрузке (F5)."""
    register_ui_assets_and_routes()
    ui.dark_mode(True)
    ui.add_head_html(
        '<script src="/static/3Dmol-min.js"></script>'
        '<script src="/static/jsme/jsme.nocache.js"></script>'
    )

    await client.connected()

    resolved_sid = (
        session_id
        or request.query_params.get("session_id")
        or app.storage.tab.get("session_id")
    )

    runtime: SessionRuntime | None = None
    if resolved_sid:
        sid_str = str(resolved_sid)
        runtime = session_registry.get(sid_str)
        if runtime is None:
            runtime = await session_registry.restore_from_workspace(sid_str)
    if runtime is None:
        new_sid = str(resolved_sid or f"sess_{uuid.uuid4().hex[:8]}")
        runtime = session_registry.get_or_create(session_id=new_sid)

    app.storage.tab["session_id"] = runtime.session_id
    ui.navigate.history.replace(f"/?session_id={runtime.session_id}")

    top_bar: TopStatusBar | None = None
    agent_rail: AgentRail | None = None
    mpo_canvas: MPOCanvas | None = None
    candidate_grid: CandidateGrid | None = None
    inspector: InspectorPane | None = None
    tool_settings: ToolSettingsDrawer | None = None

    def _select_candidate(cid: str) -> None:
        runtime.view_model.selected_candidate_id = cid
        if inspector is not None:
            inspector.refresh()
        if mpo_canvas is not None:
            mpo_canvas.refresh(mark_pca_dirty=False)
        if candidate_grid is not None:
            candidate_grid.select_row(cid)

    def _on_mpo_recalculated() -> None:
        if candidate_grid is not None:
            candidate_grid.push_updated_rows()
        if inspector is not None:
            inspector.refresh()
        if top_bar is not None:
            top_bar.refresh()

    def _on_drawer_local_mpo_updated() -> None:
        if mpo_canvas is not None:
            mpo_canvas.refresh(mark_pca_dirty=False)
        _on_mpo_recalculated()

    def _on_drawer_3d_pca_updated() -> None:
        if inspector is not None:
            inspector.invalidate_3d_cache()
            inspector.refresh()
        if mpo_canvas is not None:
            mpo_canvas.refresh(mark_pca_dirty=True)

    def _on_drawer_tool_config_changed() -> None:
        if top_bar is not None:
            top_bar.refresh()
        if candidate_grid is not None:
            candidate_grid.refresh_telemetry()
        if inspector is not None:
            inspector.refresh()

    def _on_external_tool_config_changed() -> None:
        if top_bar is not None:
            top_bar.refresh()
        if tool_settings is not None:
            tool_settings.refresh()
        if candidate_grid is not None:
            candidate_grid.refresh_telemetry()

    def _on_row_updated(cid: str) -> None:
        if candidate_grid is not None:
            candidate_grid.push_updated_rows([cid])
        if mpo_canvas is not None:
            mpo_canvas.refresh(mark_pca_dirty=False)

    async def _on_session_switched(new_sid: str) -> None:
        app.storage.tab["session_id"] = new_sid
        ui.navigate.to(f"/?session_id={new_sid}")

    def _on_csv_uploaded(csv_path: Path) -> None:
        if agent_rail is not None:
            agent_rail.uploaded_csv_path = csv_path

    def _toggle_settings() -> None:
        if tool_settings is not None:
            tool_settings.toggle()

    top_bar = TopStatusBar(
        runtime,
        on_session_switched=_on_session_switched,
        on_csv_uploaded=_on_csv_uploaded,
        on_toggle_settings=_toggle_settings,
    )

    tool_settings = ToolSettingsDrawer(
        runtime,
        on_local_mpo_updated=_on_drawer_local_mpo_updated,
        on_3d_pca_params_updated=_on_drawer_3d_pca_updated,
        on_protected_smarts_changed=lambda sm: (
            agent_rail.set_protected_smarts(sm) if agent_rail is not None else None
        ),
        get_sketcher_smiles=lambda: (
            agent_rail.get_sketcher_smiles() if agent_rail is not None else ""
        ),
        set_sketcher_smiles=lambda smi: (
            agent_rail.set_sketcher_smiles(smi) if agent_rail is not None else None
        ),
        on_tool_config_changed=_on_drawer_tool_config_changed,
    )

    with (
        ui.element("div").classes("w-full h-[calc(100vh-52px)] bg-slate-950 overflow-hidden"),
        ui.splitter(value=24, limits=(16, 42)).classes("w-full h-full") as outer_split,
    ):
        with outer_split.before:
            agent_rail = AgentRail(
                runtime,
                on_tool_config_changed=_on_external_tool_config_changed,
            )
        with (
            outer_split.after,
            ui.splitter(value=62, limits=(38, 82)).classes("w-full h-full") as mid_split,
        ):
                    with mid_split.before, ui.splitter(
                        horizontal=True,
                        value=48,
                        limits=(24, 76),
                    ).classes("w-full h-full") as center_split:
                        with center_split.before:
                            mpo_canvas = MPOCanvas(
                                runtime,
                                on_candidate_selected=_select_candidate,
                                on_mpo_recalculated=_on_mpo_recalculated,
                                on_tool_config_changed=_on_external_tool_config_changed,
                            )
                        with center_split.after:
                            candidate_grid = CandidateGrid(
                                runtime,
                                on_candidate_selected=_select_candidate,
                            )
                    with mid_split.after:
                        inspector = InspectorPane(
                            runtime,
                            on_row_updated=_on_row_updated,
                        )

    async def _on_runtime_event(event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        with client:
            if top_bar is not None:
                top_bar.refresh()
            if agent_rail is not None:
                agent_rail.refresh()
            if tool_settings is not None:
                tool_settings.refresh()
            if candidate_grid is not None:
                candidate_grid.refresh_telemetry()
            if event_type in ("pipeline_started", "branch_restarted"):
                if candidate_grid is not None:
                    candidate_grid.reset_rows()
                    if event_type == "branch_restarted":
                        candidate_grid.push_new_rows()
                if mpo_canvas is not None:
                    mpo_canvas.refresh(mark_pca_dirty=True)
                if inspector is not None:
                    inspector.refresh()
            elif event_type in (
                "node_completed",
                "pipeline_completed",
                "hitl_resumed",
            ):
                if candidate_grid is not None:
                    candidate_grid.push_new_rows()
                if mpo_canvas is not None:
                    mpo_canvas.refresh(mark_pca_dirty=True)
                if inspector is not None:
                    inspector.refresh()
            elif event_type in ("mpo_recalculated", "tool_config_reset"):
                if candidate_grid is not None:
                    candidate_grid.push_updated_rows()
                if mpo_canvas is not None:
                    mpo_canvas.refresh(mark_pca_dirty=False)
                if inspector is not None:
                    inspector.refresh()
            elif event_type == "pipeline_error":
                ui.notify(f"Ошибка графа: {event.get('error', '')}", type="negative")

    client_key = str(client.id)
    session_registry.subscribe(runtime.session_id, client_key, _on_runtime_event)
    client.on_disconnect(
        lambda: session_registry.unsubscribe(runtime.session_id, client_key)
    )


def run_ui(
    *,
    host: str | None = None,
    port: int | None = None,
    native: bool = False,
    reload: bool = False,
) -> None:
    """Запускает NiceGUI-приложение строго в одном воркере (`workers=1`)."""
    register_ui_assets_and_routes()
    ui.run(
        host=host or settings.ui_host,
        port=port or settings.ui_port,
        title="Pishovichok · Chemical IDE",
        dark=True,
        reload=reload,
        native=native,
        workers=1,
        show=False,
        storage_secret="pishovichok-chemical-ide-secret",
    )


def run_app() -> None:
    """Entry point для консольной команды `pishovichok-ui`."""
    run_ui()


if __name__ in {"__main__", "__mp_main__"}:
    run_app()
