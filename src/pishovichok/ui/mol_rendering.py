"""Модуль 2D-визуализации молекулярных структур и пар трансформаций (Слой 4).

Использует C++ ядро RDKit `rdMolDraw2D` для генерации векторных SVG-изображений
с цветовой подсветкой модифицируемых функциональных групп и атомов.

Особенности:
- Цвета нормализованы строго в диапазон [0.0, 1.0]:
  - Зеленый: (0.1, 0.8, 0.2) — прирост активности (positive delta / lead candidate).
  - Красный: (0.9, 0.2, 0.2) — падение активности / критический скачок (Activity Cliff).
  - Синий: (0.2, 0.4, 0.9) — скаффолд или референсное ядро.
- Гарантированный локальный откат (Zero-Failure): при ошибке подсветки
  структура отрисовывается без выделения атомов.
- Поддержка кэширования через `@st.cache_data`.
"""

from __future__ import annotations

import base64
import contextlib
import html

import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# Палитра стандартных цветов для подсветки (RGB в [0.0, 1.0])
COLOR_ACTIVITY_GAIN: tuple[float, float, float] = (0.1, 0.8, 0.2)  # Зеленый
COLOR_ACTIVITY_LOSS: tuple[float, float, float] = (0.9, 0.2, 0.2)  # Красный
COLOR_CORE_REFERENCE: tuple[float, float, float] = (0.2, 0.4, 0.9)  # Синий
COLOR_NEUTRAL_ACCENT: tuple[float, float, float] = (0.95, 0.65, 0.15)  # Оранжевый


@st.cache_data(show_spinner=False)
def render_molecule_svg(
    smiles: str,
    highlight_atoms: tuple[int, ...] | list[int] | None = None,
    highlight_colors: dict[int, tuple[float, float, float]] | None = None,
    width: int = 340,
    height: int = 240,
    legend: str = "",
    clear_background: bool = True,
) -> str:
    """Генерирует векторный SVG код для молекулы с подсветкой заданных атомов.

    Args:
        smiles: Канонический SMILES молекулы.
        highlight_atoms: Список индексов подсвечиваемых атомов.
        highlight_colors: Отображение {atom_idx: (r, g, b)} с нормализацией в [0.0, 1.0].
        width: Ширина SVG полотна.
        height: Высота SVG полотна.
        legend: Текстовая подпись под структурой.
        clear_background: Прозрачный фон для бесшовной интеграции в тему Streamlit.

    Returns:
        Валидный XML/SVG строковый фрагмент.
    """
    if not smiles or not isinstance(smiles, str):
        return _render_placeholder_svg("Пустой SMILES", width, height)

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return _render_placeholder_svg("Невалидная структура SMILES", width, height)

    # Генерация 2D-координат, если они отсутствуют
    with contextlib.suppress(RuntimeError, ValueError):
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)

    atoms_to_highlight = list(highlight_atoms) if highlight_atoms else []
    colors_dict = highlight_colors or {}

    # Если заданы атомы для подсветки, но не заданы цвета, назначаем цвет по умолчанию (зеленый)
    if atoms_to_highlight and not colors_dict:
        colors_dict = {idx: COLOR_ACTIVITY_GAIN for idx in atoms_to_highlight}

    # Попытка отрисовки с подсветкой
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
        # Детерминированный фолбэк: отрисовка базовой структуры без подсветки
        try:
            fallback_drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            fallback_drawer.drawOptions().clearBackground = clear_background
            rdMolDraw2D.PrepareAndDrawMolecule(fallback_drawer, mol, legend=legend)
            fallback_drawer.FinishDrawing()
            return fallback_drawer.GetDrawingText()
        except (RuntimeError, ValueError, TypeError) as exc:
            return _render_placeholder_svg(f"Ошибка рендеринга: {exc}", width, height)


@st.cache_data(show_spinner=False)
def render_mmp_pair_svg(
    smiles_lhs: str,
    smiles_rhs: str,
    title_lhs: str = "Исходное соединение",
    title_rhs: str = "Модификация",
    delta_pic50: float | None = None,
    sali_index: float | None = None,
    width: int = 680,
    height: int = 260,
) -> str:
    """Генерирует объединенный 2D-рендеринг пары молекул (например, Activity Cliff или MMP).

    Выделяет измененные функциональные группы:
    - зеленым при росте активности (delta_pic50 > 0),
    - красным при резком падении активности (Activity Cliff или delta_pic50 < 0).
    """
    sub_width = max(int(width / 2) - 15, 200)

    # Определение цвета подсветки по сдвигу активности
    color = (
        COLOR_ACTIVITY_GAIN
        if (delta_pic50 is not None and delta_pic50 >= 0)
        else COLOR_ACTIVITY_LOSS
    )

    # Поиск модифицированных фрагментов через максимальную общую подструктуру или подсветку
    mol_l = Chem.MolFromSmiles(smiles_lhs) if smiles_lhs else None
    mol_r = Chem.MolFromSmiles(smiles_rhs) if smiles_rhs else None

    hl_atoms_l: list[int] = []
    hl_atoms_r: list[int] = []

    if mol_l and mol_r:
        with contextlib.suppress(RuntimeError, ValueError, TypeError):
            from rdkit.Chem import rdFMCS

            mcs_res = rdFMCS.FindMCS([mol_l, mol_r], timeout=1)
            if mcs_res and mcs_res.smartsString:
                patt = Chem.MolFromSmarts(mcs_res.smartsString)
                if patt:
                    match_l = set(mol_l.GetSubstructMatch(patt))
                    match_r = set(mol_r.GetSubstructMatch(patt))
                    # Модифицированные атомы — это те, что НЕ входят в общую сердцевину
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
        width=sub_width,
        height=height - 50,
        legend="",
    )
    svg_r = render_molecule_svg(
        smiles=smiles_rhs,
        highlight_atoms=hl_atoms_r,
        highlight_colors={idx: color for idx in hl_atoms_r},
        width=sub_width,
        height=height - 50,
        legend="",
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

    combined_html = f"""
    <div style="display:flex; flex-direction:column; align-items:center; border:1px solid #e1e4e8; border-radius:8px; padding:10px; background:#ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.05); margin-bottom:12px;">
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
    return combined_html


def molecule_svg_to_data_uri(svg_content: str) -> str:
    """Кодирует SVG строку в Base64 Data URI."""
    encoded = base64.b64encode(svg_content.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _render_placeholder_svg(message: str, width: int, height: int) -> str:
    """Генерирует заглушку SVG с сообщением об ошибке."""
    safe_msg = html.escape(message)
    return (
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg' "
        f"style='border:1px dashed #d1d5db; border-radius:6px; background:#f9fafb;'>"
        f"<rect width='100%' height='100%' fill='#f9fafb'/>"
        f"<text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' "
        f"fill='#6b7280' font-family='sans-serif' font-size='13'>{safe_msg}</text>"
        f"</svg>"
    )
