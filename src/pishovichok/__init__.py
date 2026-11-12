"""Мультиагентная система анализа зависимостей 'структура–активность' (SAR) и направленного дизайна малых молекул."""

from pishovichok.config import AppSettings, settings

__version__ = "0.1.0"
__all__ = ["AppSettings", "__version__", "main", "settings"]


def main() -> None:
    """Точка входа CLI пакета pishovichok."""
    print(
        f"pishovichok v{__version__} initialized (data_dir={settings.resolved_data_dir})"
    )
