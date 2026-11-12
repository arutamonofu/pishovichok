"""Слой 4: Интерактивный веб-интерфейс Streamlit и 2D-рендеринг молекул."""

from pishovichok.ui.app import main, run_app
from pishovichok.ui.components import (
    DEMO_SCENARIOS,
    render_candidates_four_zones_table,
    render_control_panel,
    render_explainable_trace,
    render_export_buttons,
    render_sar_and_cliffs_view,
)
from pishovichok.ui.mol_rendering import (
    molecule_svg_to_data_uri,
    render_mmp_pair_svg,
    render_molecule_svg,
)

__all__ = [
    "DEMO_SCENARIOS",
    "main",
    "molecule_svg_to_data_uri",
    "render_candidates_four_zones_table",
    "render_control_panel",
    "render_explainable_trace",
    "render_export_buttons",
    "render_mmp_pair_svg",
    "render_molecule_svg",
    "render_sar_and_cliffs_view",
    "run_app",
]
