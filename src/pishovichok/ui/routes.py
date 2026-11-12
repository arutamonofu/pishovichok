"""FastAPI-маршруты для ленивой отдачи 2D SVG-структур (без передачи SVG по WebSocket)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from nicegui import run

from pishovichok.core.rendering import (
    COLOR_ACTIVITY_GAIN,
    render_aligned_diff_svg,
    render_molecule_svg,
)
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

    if variant == "diff":
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
