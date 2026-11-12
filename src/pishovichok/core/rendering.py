"""Модуль 2D-рендеринга, выравнивания по скаффолду, 3D-конформеров и проекции химпространства (Слой 1).

Полностью изолирован от веб-фреймворков (`nicegui`, `fastapi`, `starlette`).
Все публичные функции объявлены на верхнем уровне модуля без замыканий и кэш-обёрток,
принимают и возвращают исключительно `pickle`-сериализуемые типы для безопасного исполнения
в пуле процессов `spawn` (`nicegui.run.cpu_bound`).
"""

from __future__ import annotations

import base64
import contextlib
import html

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdDepictor, rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA

# Палитра стандартных цветов для подсветки (RGB строго в [0.0, 1.0])
COLOR_ACTIVITY_GAIN: tuple[float, float, float] = (0.1, 0.8, 0.2)  # Зелёный
COLOR_ACTIVITY_LOSS: tuple[float, float, float] = (0.9, 0.2, 0.2)  # Красный
COLOR_CORE_REFERENCE: tuple[float, float, float] = (0.2, 0.4, 0.9)  # Синий
COLOR_NEUTRAL_ACCENT: tuple[float, float, float] = (0.95, 0.65, 0.15)  # Оранжевый

_PCA_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def _render_placeholder_svg(message: str, width: int, height: int) -> str:
    """Генерирует заглушку SVG с сообщением об ошибке."""
    safe_msg = html.escape(message)
    return (
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg' "
        f"style='border:1px dashed #475569; border-radius:6px; background:#0f172a;'>"
        f"<rect width='100%' height='100%' fill='#0f172a'/>"
        f"<text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' "
        f"fill='#94a3b8' font-family='sans-serif' font-size='13'>{safe_msg}</text>"
        f"</svg>"
    )


def _compute_aligned_2d_coords(
    mol: Chem.Mol,
    align_to_scaffold_smiles: str | None = None,
) -> None:
    """Вычисляет 2D-координаты молекулы, выравнивая её по скаффолду при наличии."""
    if align_to_scaffold_smiles:
        scaf_mol = Chem.MolFromSmiles(align_to_scaffold_smiles.strip())
        if scaf_mol is not None and scaf_mol.GetNumAtoms() > 0:
            try:
                AllChem.Compute2DCoords(scaf_mol)
                if mol.HasSubstructMatch(scaf_mol):
                    rdDepictor.GenerateDepictionMatching2DStructure(mol, scaf_mol)
                    return
            except (RuntimeError, ValueError):
                pass
    with contextlib.suppress(RuntimeError, ValueError):
        AllChem.Compute2DCoords(mol)


def render_molecule_svg(
    smiles: str,
    highlight_atoms: tuple[int, ...] | list[int] | None = None,
    highlight_colors: dict[int, tuple[float, float, float]] | None = None,
    width: int = 340,
    height: int = 240,
    legend: str = "",
    clear_background: bool = True,
    align_to_scaffold_smiles: str | None = None,
) -> str:
    """Генерирует векторный SVG код для молекулы с выравниванием по скаффолду и подсветкой атомов."""
    if not smiles or not isinstance(smiles, str):
        return _render_placeholder_svg("Пустой SMILES", width, height)

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return _render_placeholder_svg("Невалидная структура SMILES", width, height)

    _compute_aligned_2d_coords(mol, align_to_scaffold_smiles=align_to_scaffold_smiles)

    num_atoms = mol.GetNumAtoms()
    atoms_to_highlight = (
        [int(idx) for idx in highlight_atoms if 0 <= int(idx) < num_atoms]
        if highlight_atoms
        else []
    )
    colors_dict: dict[int, tuple[float, float, float]] = (
        {int(k): v for k, v in highlight_colors.items() if 0 <= int(k) < num_atoms}
        if highlight_colors
        else {}
    )

    if atoms_to_highlight and not colors_dict:
        colors_dict = {idx: COLOR_ACTIVITY_GAIN for idx in atoms_to_highlight}

    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = clear_background
        opts.bondLineWidth = 2

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            legend=legend,
            highlightAtoms=atoms_to_highlight if atoms_to_highlight else None,
            highlightAtomColors=colors_dict if colors_dict else None,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except (RuntimeError, ValueError, TypeError):
        try:
            fallback_drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            fallback_drawer.drawOptions().clearBackground = clear_background
            rdMolDraw2D.PrepareAndDrawMolecule(fallback_drawer, mol, legend=legend)
            fallback_drawer.FinishDrawing()
            return fallback_drawer.GetDrawingText()
        except (RuntimeError, ValueError, TypeError) as exc:
            return _render_placeholder_svg(f"Ошибка рендеринга: {exc}", width, height)


def render_aligned_diff_svg(
    smiles: str,
    parent_smiles: str | None = None,
    scaffold_smiles: str | None = None,
    highlight_atoms: list[int] | tuple[int, ...] | None = None,
    is_positive_delta: bool = True,
    width: int = 280,
    height: int = 190,
    legend: str = "",
) -> str:
    """Рендерит 2D SVG молекулы, выровненной по скаффолду, с подсветкой изменённых фрагментов."""
    if not smiles or not isinstance(smiles, str):
        return _render_placeholder_svg("Пустой SMILES", width, height)

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return _render_placeholder_svg("Невалидный SMILES", width, height)

    hl_indices = list(highlight_atoms) if highlight_atoms else []
    if not hl_indices and parent_smiles:
        parent_mol = Chem.MolFromSmiles(parent_smiles.strip())
        if parent_mol is not None:
            with contextlib.suppress(RuntimeError, ValueError, TypeError):
                mcs_res = rdFMCS.FindMCS([parent_mol, mol], timeout=1)
                if mcs_res and mcs_res.smartsString:
                    patt = Chem.MolFromSmarts(mcs_res.smartsString)
                    if patt is not None:
                        matched = set(mol.GetSubstructMatch(patt))
                        if matched:
                            hl_indices = [
                                idx
                                for idx in range(mol.GetNumAtoms())
                                if idx not in matched
                            ]

    color = COLOR_ACTIVITY_GAIN if is_positive_delta else COLOR_ACTIVITY_LOSS
    colors_map = {idx: color for idx in hl_indices}

    return render_molecule_svg(
        smiles=smiles,
        highlight_atoms=hl_indices,
        highlight_colors=colors_map,
        width=width,
        height=height,
        legend=legend,
        clear_background=True,
        align_to_scaffold_smiles=scaffold_smiles or parent_smiles,
    )


def render_mmp_pair_svgs(
    smiles_lhs: str,
    smiles_rhs: str,
    scaffold_smiles: str | None = None,
    delta_pic50: float | None = None,
    width: int = 300,
    height: int = 210,
) -> tuple[str, str]:
    """Возвращает пару выровненных 2D SVG `(svg_lhs, svg_rhs)` с подсветкой различий."""
    color = (
        COLOR_ACTIVITY_GAIN
        if (delta_pic50 is None or delta_pic50 >= 0)
        else COLOR_ACTIVITY_LOSS
    )

    mol_l = Chem.MolFromSmiles(smiles_lhs) if smiles_lhs else None
    mol_r = Chem.MolFromSmiles(smiles_rhs) if smiles_rhs else None

    hl_atoms_l: list[int] = []
    hl_atoms_r: list[int] = []
    align_smarts_or_smiles = scaffold_smiles

    if mol_l and mol_r:
        with contextlib.suppress(RuntimeError, ValueError, TypeError):
            mcs_res = rdFMCS.FindMCS([mol_l, mol_r], timeout=1)
            if mcs_res and mcs_res.smartsString:
                patt = Chem.MolFromSmarts(mcs_res.smartsString)
                if patt:
                    match_l = set(mol_l.GetSubstructMatch(patt))
                    match_r = set(mol_r.GetSubstructMatch(patt))
                    hl_atoms_l = [
                        i for i in range(mol_l.GetNumAtoms()) if i not in match_l
                    ]
                    hl_atoms_r = [
                        i for i in range(mol_r.GetNumAtoms()) if i not in match_r
                    ]

    svg_l = render_molecule_svg(
        smiles=smiles_lhs,
        highlight_atoms=hl_atoms_l,
        highlight_colors={idx: color for idx in hl_atoms_l},
        width=width,
        height=height,
        legend="",
        align_to_scaffold_smiles=align_smarts_or_smiles,
    )
    svg_r = render_molecule_svg(
        smiles=smiles_rhs,
        highlight_atoms=hl_atoms_r,
        highlight_colors={idx: color for idx in hl_atoms_r},
        width=width,
        height=height,
        legend="",
        align_to_scaffold_smiles=align_smarts_or_smiles or smiles_lhs,
    )
    return (svg_l, svg_r)


def render_mmp_pair_svg(
    smiles_lhs: str,
    smiles_rhs: str,
    title_lhs: str = "Исходное соединение",
    title_rhs: str = "Модификация",
    delta_pic50: float | None = None,
    sali_index: float | None = None,
    width: int = 680,
    height: int = 260,
    scaffold_smiles: str | None = None,
) -> str:
    """Генерирует объединённый HTML/SVG-блок сравнения пары молекул (MMP / Activity Cliff)."""
    sub_width = max(int(width / 2) - 15, 200)
    svg_l, svg_r = render_mmp_pair_svgs(
        smiles_lhs=smiles_lhs,
        smiles_rhs=smiles_rhs,
        scaffold_smiles=scaffold_smiles,
        delta_pic50=delta_pic50,
        width=sub_width,
        height=height - 50,
    )

    badge_html = ""
    if delta_pic50 is not None:
        sign = "+" if delta_pic50 > 0 else ""
        badge_color = "#28a745" if delta_pic50 >= 0 else "#dc3545"
        badge_html += (
            f"<span style='background-color:{badge_color}; color:white; "
            f"padding:3px 8px; border-radius:12px; font-size:12px; font-weight:bold;'>"
            f"ΔpIC50 = {sign}{delta_pic50:.2f}</span> "
        )
    if sali_index is not None:
        badge_html += (
            f"<span style='background-color:#6f42c1; color:white; "
            f"padding:3px 8px; border-radius:12px; font-size:12px; font-weight:bold;'>"
            f"SALI = {sali_index:.1f}</span>"
        )

    title_l_html = html.escape(title_lhs) if title_lhs else ""
    title_r_html = html.escape(title_rhs) if title_rhs else ""

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; border:1px solid #e1e4e8; border-radius:8px; padding:10px; background:#ffffff; margin-bottom:12px;">
      <div style="margin-bottom:8px;">{badge_html}</div>
      <div style="display:flex; justify-content:space-around; width:100%; align-items:center;">
        <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
          <div style="font-size:13px; font-weight:bold; color:#24292e; margin-bottom:4px;">{title_l_html}</div>
          <div>{svg_l}</div>
        </div>
        <div style="font-size:24px; color:#586069; padding:0 10px;">➔</div>
        <div style="flex:1; display:flex; flex-direction:column; align-items:center;">
          <div style="font-size:13px; font-weight:bold; color:#24292e; margin-bottom:4px;">{title_r_html}</div>
          <div>{svg_r}</div>
        </div>
      </div>
    </div>
    """


def molecule_svg_to_data_uri(svg_content: str) -> str:
    """Кодирует SVG строку в Base64 Data URI."""
    encoded = base64.b64encode(svg_content.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def generate_3d_conformer_molblock(
    smiles: str,
    highlight_atoms: list[int] | tuple[int, ...] | None = None,
    random_seed: int = 42,
    optimize_uff: bool = True,
) -> tuple[str, list[int]]:
    """Генерирует 3D-конформер через ETKDGv3 + UFF и возвращает `(molblock, valid_highlight_indices)`.

    Тяжёлые атомы сохраняют исходную индексацию `[0 .. N-1]` после `Chem.AddHs(mol)`,
    поэтому индексы подсветки `highlight_atoms` корректно отображаются в 3Dmol.js.
    """
    if not smiles or not isinstance(smiles, str):
        raise ValueError("Пустой SMILES для генерации 3D-конформера.")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Невалидный SMILES для 3D-конформера: '{smiles}'")

    heavy_count = mol.GetNumAtoms()
    valid_highlights = [
        int(idx) for idx in (highlight_atoms or []) if 0 <= int(idx) < heavy_count
    ]

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)

    embed_status = AllChem.EmbedMolecule(mol_h, params)
    if embed_status == -1:
        params.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(mol_h, params)
    if embed_status == -1:
        AllChem.Compute2DCoords(mol_h)

    if optimize_uff:
        with contextlib.suppress(RuntimeError, ValueError):
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)

    molblock = Chem.MolToMolBlock(mol_h)
    return (molblock, valid_highlights)


def compute_chemical_space_projection(
    smiles_list: list[str],
) -> list[tuple[float, float]]:
    """Вычисляет 2D-проекцию (PCA по отпечаткам Моргана) для списка SMILES.

    При числе валидных молекул < 3 или нулевой дисперсии возвращает `(cLogP, TPSA)`.
    """
    if not smiles_list:
        return []

    fps: list[np.ndarray] = []
    valid_indices: list[int] = []
    fallback_coords: list[tuple[float, float]] = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi).strip()) if smi else None
        if mol is None:
            fallback_coords.append((0.0, 0.0))
            continue
        clogp = round(float(Descriptors.MolLogP(mol)), 4)
        tpsa = round(float(Descriptors.TPSA(mol)), 4)
        fallback_coords.append((clogp, tpsa))
        fp_arr = _PCA_FP_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.float32)
        fps.append(fp_arr)
        valid_indices.append(idx)

    if len(fps) < 3:
        return fallback_coords

    matrix = np.vstack(fps)
    try:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(matrix)
        result = list(fallback_coords)
        for local_i, orig_idx in enumerate(valid_indices):
            result[orig_idx] = (
                round(float(coords_2d[local_i, 0]), 4),
                round(float(coords_2d[local_i, 1]), 4),
            )
        return result
    except Exception:  # noqa: BLE001
        return fallback_coords
