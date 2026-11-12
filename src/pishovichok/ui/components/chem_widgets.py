"""Клиентские обёртки 3Dmol.js и JSME/Ketcher для NiceGUI 4.0 без внешних CDN."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_VIEWER3D_JS = str(_STATIC_DIR / "viewer3d.js")
_JSME_SKETCHER_JS = str(_STATIC_DIR / "jsme_sketcher.js")
_KETCHER_SKETCHER_JS = str(_STATIC_DIR / "ketcher_sketcher.js")


class MoleculeViewer3D(
    ui.element,
    component=_VIEWER3D_JS,
):
    """Vue/ESM-обёртка над локальным `3Dmol-min.js` с единым WebGL-контекстом."""

    def __init__(self, height: str = "290px") -> None:
        super().__init__()
        self._props["height"] = height
        self.classes("w-full rounded-lg border border-slate-700 bg-slate-950")

    def load_molblock(
        self,
        molblock: str,
        highlight_indices: list[int] | None = None,
    ) -> None:
        """Передаёт 3D MolBlock и индексы подсвечиваемых атомов в активный WebGL-вьюер."""
        self.run_method("loadMolBlock", molblock, list(highlight_indices or []))

    def clear_viewer(self) -> None:
        """Очищает текущую 3D-модель без пересоздания WebGL-контекста."""
        self.run_method("clearViewer")

    def clear(self) -> None:
        """Совместимый алиас очистки текущей 3D-модели."""
        self.clear_viewer()


class MoleculeSketcher(
    ui.element,
    component=_JSME_SKETCHER_JS,
):
    """Vue/ESM-обёртка над локальным редактором структур JSME."""

    def __init__(
        self,
        initial_smiles: str = "",
        height: str = "240px",
        on_smiles_change: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__()
        self._props["initialSmiles"] = initial_smiles
        self._props["height"] = height
        self.classes("w-full rounded-lg border border-slate-700 bg-white")
        if on_smiles_change is not None:
            self.on(
                "smiles-changed",
                lambda e: on_smiles_change(
                    str(e.args.get("smiles", "") if isinstance(e.args, dict) else e.args or "")
                ),
            )

    def set_smiles(self, smiles: str) -> None:
        """Программно загружает SMILES в скетчер."""
        self.run_method("setSmiles", smiles)


class KetcherSketcher(
    ui.element,
    component=_KETCHER_SKETCHER_JS,
):
    """Опциональная Vue/ESM-обёртка над предсобранным локальным `ketcher-standalone`."""

    def __init__(
        self,
        initial_smiles: str = "",
        height: str = "280px",
        on_smiles_change: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__()
        self._props["initialSmiles"] = initial_smiles
        self._props["height"] = height
        self.classes("w-full rounded-lg border border-slate-700 bg-white")
        if on_smiles_change is not None:
            self.on(
                "smiles-changed",
                lambda e: on_smiles_change(
                    str(e.args.get("smiles", "") if isinstance(e.args, dict) else e.args or "")
                ),
            )

    def set_smiles(self, smiles: str) -> None:
        """Программно загружает SMILES в Ketcher."""
        self.run_method("setSmiles", smiles)


def create_sketcher(
    *,
    backend: str = "jsme",
    initial_smiles: str = "",
    height: str = "240px",
    on_smiles_change: Callable[[str], Any] | None = None,
) -> MoleculeSketcher | KetcherSketcher:
    """Фабрика скетчера, переключаемая через `settings.sketcher_backend`."""
    if backend == "ketcher" and (_STATIC_DIR / "ketcher" / "index.html").is_file():
        return KetcherSketcher(
            initial_smiles=initial_smiles,
            height=height,
            on_smiles_change=on_smiles_change,
        )
    return MoleculeSketcher(
        initial_smiles=initial_smiles,
        height=height,
        on_smiles_change=on_smiles_change,
    )
