"""Строгие монохромные векторные SVG-иконки (14px, stroke-width: 1.75, currentColor)."""

from __future__ import annotations

_SVG_PATHS: dict[str, str] = {
    "flask": (
        '<path d="M9 3h6"/><path d="M10 3v5.4a2 2 0 0 1-.4 1.2L5.2 15.8A2 2 0 0 0 6.8 19h10.4a2 2 0 0 0 1.6-3.2L14.4 9.6A2 2 0 0 1 14 8.4V3"/>'
    ),
    "play": '<polygon points="6 4 20 12 6 20 6 4"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "pin": '<path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/>',
    "exclude": '<circle cx="12" cy="12" r="9"/><path d="m4.9 4.9 14.2 14.2"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "clear": '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "chevron_left": '<polyline points="15 18 9 12 15 6"/>',
    "chevron_right": '<polyline points="9 18 15 12 9 6"/>',
    "chevron_up": '<polyline points="18 15 12 9 6 15"/>',
    "chevron_down": '<polyline points="6 9 12 15 18 9"/>',
    "paperclip": '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
    "edit": '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
    "arrow_right": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    "arrow_down": '<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>',
    "maximize": '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>',
    "close": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
    "bot": (
        '<rect x="3" y="11" width="18" height="10" rx="2"/>'
        '<circle cx="12" cy="5" r="2"/><path d="M12 7v4"/>'
        '<line x1="8" y1="16" x2="8.01" y2="16"/><line x1="16" y1="16" x2="16.01" y2="16"/>'
    ),
    "wrench": (
        '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77'
        'a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91'
        'a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'
    ),
    "sparkles": (
        '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912'
        'a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275'
        'L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>'
    ),
    "cube": (
        '<path d="m21 16-9 5-9-5V8l9-5 9 5v8z"/>'
        '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>'
    ),
    "send": '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
}


def svg_icon(name: str, size: int = 14, css_class: str = "inline-block shrink-0") -> str:
    """Возвращает инлайновую монохромную SVG-иконку (14px, stroke-width: 1.75, currentColor)."""
    body = _SVG_PATHS.get(name, _SVG_PATHS["file"])
    return (
        f'<svg class="{css_class}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.75" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f"{body}</svg>"
    )
