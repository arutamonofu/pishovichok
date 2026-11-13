"""FastAPI-маршруты для ленивой отдачи 2D SVG-структур (без передачи SVG по WebSocket)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from nicegui import run
from pydantic import BaseModel, ConfigDict

from pishovichok.core.rendering import (
    COLOR_ACTIVITY_GAIN,
    render_aligned_diff_svg,
    render_molecule_svg,
    render_parent_leaving_svg,
    render_rule_transformation_svg,
)
from pishovichok.ui.bridge import agent_bridge
from pishovichok.ui.session import session_registry

router = APIRouter(prefix="/api", tags=["molecules"])


@router.get("/sessions/{session_id}/mol/{mol_id}/svg")
async def get_molecule_svg(
    session_id: str,
    mol_id: str,
    variant: str = Query(default="diff"),
    width: int = Query(default=280, ge=60, le=1000),
    height: int = Query(default=190, ge=60, le=1000),
) -> Response:
    """Возвращает выровненный по скаффолду 2D SVG молекулы с кэшированием на уровне сессии.

    Рендеринг RDKit выполняется строго в пуле процессов `spawn` через `nicegui.run.cpu_bound`.
    """
    runtime = session_registry.get(session_id)
    if runtime is None:
        runtime = await session_registry.restore_from_workspace(session_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    cache_key = (mol_id, variant, width, height)
    cached = runtime.svg_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    spec = runtime.resolve_molecule_for_svg(mol_id, variant=variant)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Molecule '{mol_id}' not found in session '{session_id}'",
        )

    smiles, parent_smiles, scaffold_smiles, hl_atoms, is_positive, legend = spec

    if variant == "parent":
        svg_text = await run.cpu_bound(
            render_parent_leaving_svg,
            smiles,
            parent_smiles,
            scaffold_smiles,
            width,
            height,
            legend,
        )
    elif variant == "diff":
        svg_text = await run.cpu_bound(
            render_aligned_diff_svg,
            smiles,
            parent_smiles,
            scaffold_smiles,
            hl_atoms,
            is_positive,
            width,
            height,
            legend,
        )
    else:
        hl_colors = (
            {idx: COLOR_ACTIVITY_GAIN for idx in hl_atoms} if hl_atoms else None
        )
        svg_text = await run.cpu_bound(
            render_molecule_svg,
            smiles,
            hl_atoms,
            hl_colors,
            width,
            height,
            legend,
            True,
            scaffold_smiles,
        )

    runtime.svg_cache[cache_key] = svg_text
    return Response(
        content=svg_text,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/sessions/{session_id}/mol/{mol_id}/brics/{frag_idx}/svg")
async def get_brics_fragment_svg(
    session_id: str,
    mol_id: str,
    frag_idx: int,
    width: int = Query(default=180, ge=60, le=600),
    height: int = Query(default=130, ge=60, le=600),
) -> Response:
    """Возвращает 2D SVG ретросинтетического фрагмента BRICS на белом фоне."""
    runtime = session_registry.get(session_id)
    if runtime is None:
        runtime = await session_registry.restore_from_workspace(session_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    cache_key = (f"{mol_id}_brics_{frag_idx}", "brics", width, height)
    cached = runtime.svg_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    frag_smiles = runtime.resolve_brics_fragment(mol_id, frag_idx)
    if frag_smiles is None:
        raise HTTPException(
            status_code=404,
            detail=f"BRICS fragment {frag_idx} for '{mol_id}' not found",
        )

    svg_text = await run.cpu_bound(
        render_molecule_svg,
        frag_smiles,
        None,
        None,
        width,
        height,
        "",
        True,
        None,
    )
    runtime.svg_cache[cache_key] = svg_text
    return Response(
        content=svg_text,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/sessions/{session_id}/rule/{rule_id}/svg")
async def get_sar_rule_svg(
    session_id: str,
    rule_id: str,
    width: int = Query(default=320, ge=120, le=800),
    height: int = Query(default=110, ge=60, le=400),
) -> Response:
    """Возвращает 2D SVG превью трансформации правила SAR (`R_from -> R_to`)."""
    runtime = session_registry.get(session_id)
    if runtime is None:
        runtime = await session_registry.restore_from_workspace(session_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    cache_key = (f"rule_{rule_id}", "rule", width, height)
    cached = runtime.svg_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    spec = runtime.resolve_rule_for_svg(rule_id)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"SARRule '{rule_id}' not found in session '{session_id}'",
        )

    from_frag, to_frag, delta_pic50 = spec
    svg_text = await run.cpu_bound(
        render_rule_transformation_svg,
        from_frag,
        to_frag,
        delta_pic50,
        width,
        height,
    )
    runtime.svg_cache[cache_key] = svg_text
    return Response(
        content=svg_text,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


class StandaloneDockingRequest(BaseModel):
    """Запрос на ручной запуск физического 3D-докинга для сессии."""

    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[str] | None = None
    custom_pdb_id: str | None = None
    custom_box_center: list[float] | None = None
    exhaustiveness: int | None = None
    top_k: int | None = None


@router.post("/sessions/{session_id}/docking")
async def trigger_standalone_docking(
    session_id: str,
    payload: StandaloneDockingRequest | None = None,
) -> dict[str, object]:
    """Запускает физический 3D-докинг вручную без перезапуска ChEMBL/SAR/CReM."""
    req = payload or StandaloneDockingRequest()
    try:
        summary = await agent_bridge.run_standalone_docking(
            session_id=session_id,
            candidate_ids=req.candidate_ids,
            custom_pdb_id=req.custom_pdb_id,
            custom_box_center=req.custom_box_center,
            exhaustiveness=req.exhaustiveness,
            top_k=req.top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return summary
