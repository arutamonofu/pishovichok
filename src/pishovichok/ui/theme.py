"""Дизайн-система академической светлой темы (Clean Wet-Lab Light Mode) для среды SAR-дизайна.

Определяет базовые токены поверхностей, типографики и 4 семантические категории
природы данных (Эксперимент, Расчёт RDKit, Прогноз ADMET, Гипотеза SAR)
для сквозного применения в таблице кандидатов, блоках верификации и карточках метрик.
"""

from __future__ import annotations

THEME_CSS: str = """
<style id="pishovichok-wetlab-theme">
:root {
  --q-primary: #8B495C !important;
  --q-secondary: #73394B !important;
  --q-accent: #A6687A !important;
  --q-positive: #059669 !important;
  --q-negative: #E11D48 !important;
  --q-info: #6D597A !important;
  --q-warning: #D97706 !important;

  --bg-canvas: #FAF8F9;
  --bg-surface: #FFFFFF;
  --bg-subtle: #F4EFF1;
  --border-subtle: #E2D9DD;
  --text-primary: #1F181B;
  --text-secondary: #594D52;
  --mono-bg: #F4EFF1;
  --mono-text: #261F22;
  --mono-border: #D5C8CD;

  /* Бренд-палитра маскота «Пишовичок» (сливово-ягодный + тёплый пепельно-меловой) */
  --brand-plum: #8B495C;
  --brand-plum-hover: #73394B;
  --brand-plum-deep: #5C2D3B;
  --brand-plum-tint: #FBF5F7;
  --brand-plum-border: #E5CFD6;

  --cat-exp-border: #7C3AED;
  --cat-exp-bg: #F5F3FF;
  --cat-exp-text: #4C1D95;

  --cat-calc-border: #059669;
  --cat-calc-bg: #ECFDF5;
  --cat-calc-text: #065F46;

  --cat-pred-border: #D97706;
  --cat-pred-bg: #FFFBEB;
  --cat-pred-text: #78350F;

  --cat-hyp-border: #2563EB;
  --cat-hyp-bg: #EFF6FF;
  --cat-hyp-text: #1E3A8A;
}

body {
  background-color: var(--bg-canvas) !important;
  color: var(--text-primary) !important;
  overflow: hidden;
}

.nicegui-content {
  padding: 0 !important;
  gap: 0 !important;
}

.mono-code {
  background-color: var(--mono-bg);
  color: var(--mono-text);
  border: 1px solid var(--mono-border);
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* 4 категории природы данных: верхний бордер 3px */
.cat-experiment {
  border-top: 3px solid var(--cat-exp-border) !important;
  background-color: var(--cat-exp-bg) !important;
  color: var(--cat-exp-text) !important;
}
.cat-calc {
  border-top: 3px solid var(--cat-calc-border) !important;
  background-color: var(--cat-calc-bg) !important;
  color: var(--cat-calc-text) !important;
}
.cat-pred {
  border-top: 3px solid var(--cat-pred-border) !important;
  background-color: var(--cat-pred-bg) !important;
  color: var(--cat-pred-text) !important;
}
.cat-hypothesis {
  border-top: 3px solid var(--cat-hyp-border) !important;
  background-color: var(--cat-hyp-bg) !important;
  color: var(--cat-hyp-text) !important;
}

/* 4 категории природы данных: левый бордер 3px (Provenance, карточки правил) */
.cat-experiment-left {
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--cat-exp-border) !important;
  background-color: var(--cat-exp-bg) !important;
  color: var(--cat-exp-text) !important;
}
.cat-calc-left {
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--cat-calc-border) !important;
  background-color: var(--cat-calc-bg) !important;
  color: var(--cat-calc-text) !important;
}
.cat-pred-left {
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--cat-pred-border) !important;
  background-color: var(--cat-pred-bg) !important;
  color: var(--cat-pred-text) !important;
}
.cat-hypothesis-left {
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--cat-hyp-border) !important;
  background-color: var(--cat-hyp-bg) !important;
  color: var(--cat-hyp-text) !important;
}

/* AG Grid Quartz Light: 4 цветовые группы шапки без обрезки заголовков */
.ag-theme-quartz {
  --ag-background-color: #FFFFFF;
  --ag-foreground-color: #0F172A;
  --ag-header-background-color: #F8FAFC;
  --ag-border-color: #E2E8F0;
  --ag-row-hover-color: #FBF7F8;
  --ag-selected-row-background-color: #F6EAEE;
}
.ag-theme-quartz .ag-header-group-cell-label,
.ag-theme-quartz .ag-header-cell-label {
  overflow: visible !important;
  text-overflow: clip !important;
}
.ag-theme-quartz .ag-header-group-cell.ag-cat-experiment {
  border-top: 3px solid #7C3AED !important;
  background-color: #F5F3FF !important;
  color: #4C1D95 !important;
  font-weight: 600 !important;
}
.ag-theme-quartz .ag-header-cell.ag-cat-experiment-col {
  background-color: #FAF8FF !important;
  color: #4C1D95 !important;
}
.ag-theme-quartz .ag-header-group-cell.ag-cat-calc {
  border-top: 3px solid #059669 !important;
  background-color: #ECFDF5 !important;
  color: #065F46 !important;
  font-weight: 600 !important;
}
.ag-theme-quartz .ag-header-cell.ag-cat-calc-col {
  background-color: #F4FDF9 !important;
  color: #065F46 !important;
}
.ag-theme-quartz .ag-header-group-cell.ag-cat-pred {
  border-top: 3px solid #D97706 !important;
  background-color: #FFFBEB !important;
  color: #78350F !important;
  font-weight: 600 !important;
}
.ag-theme-quartz .ag-header-cell.ag-cat-pred-col {
  background-color: #FFFDF5 !important;
  color: #78350F !important;
}
.ag-theme-quartz .ag-header-group-cell.ag-cat-hypothesis {
  border-top: 3px solid #2563EB !important;
  background-color: #EFF6FF !important;
  color: #1E3A8A !important;
  font-weight: 600 !important;
}
.ag-theme-quartz .ag-header-cell.ag-cat-hypothesis-col {
  background-color: #F5F9FF !important;
  color: #1E3A8A !important;
}

/* Трёхколоночная CSS Grid сетка рабочей среды */
.ide-shell-grid {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr) 430px;
  width: 100%;
  height: calc(100vh - 52px);
  overflow: hidden;
  background-color: var(--bg-canvas);
}
.ide-shell-grid--collapsed {
  grid-template-columns: 28px minmax(0, 1fr) 430px;
}
@media (max-width: 1320px) {
  .ide-shell-grid {
    grid-template-columns: 268px minmax(0, 1fr) 380px;
  }
  .ide-shell-grid--collapsed {
    grid-template-columns: 28px minmax(0, 1fr) 380px;
  }
}

/* Всегда заметные аккуратные скроллбары без прыжков макета */
.wetlab-scroll {
  scrollbar-gutter: stable;
  scrollbar-width: thin;
  scrollbar-color: #D2C7CB #F4F1F3;
}
.wetlab-scroll::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}
.wetlab-scroll::-webkit-scrollbar-track {
  background: #F4F1F3;
  border-radius: 4px;
}
.wetlab-scroll::-webkit-scrollbar-thumb {
  background: #D2C7CB;
  border-radius: 4px;
}
.wetlab-scroll::-webkit-scrollbar-thumb:hover {
  background: #A9959C;
}

/* Компактные вкладки без обрезки и скачков высоты (акцент под маскота) */
.compact-tabs .q-tabs__content {
  width: 100%;
}
.compact-tabs .q-tab {
  padding: 4px 6px !important;
  min-height: 30px !important;
  min-width: 0 !important;
  flex: 1 1 0% !important;
  font-size: 12px !important;
}
.compact-tabs .q-tab--active {
  color: var(--brand-plum-hover) !important;
  font-weight: 600 !important;
}
.compact-tabs .q-tabs__indicator {
  background-color: var(--brand-plum) !important;
}
.compact-tabs .q-tab__label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.compact-tabs .q-tabs__arrow {
  display: none !important;
}
.compact-toggle .q-btn {
  padding: 4px 10px !important;
  min-height: 26px !important;
  background-color: #F4EFF1 !important;
  color: #594D52 !important;
}
.compact-toggle .q-btn:hover {
  background-color: #ECE3E7 !important;
  color: #261F22 !important;
}
.compact-toggle .q-btn.bg-primary,
.compact-toggle .q-btn.bg-blue-7,
.compact-toggle .q-btn[aria-pressed="true"] {
  background: var(--brand-plum) !important;
  background-color: var(--brand-plum) !important;
  color: #FFFFFF !important;
  font-weight: 500 !important;
}

/* Акцент фокуса полей ввода и элементов Quasar в палитре «Пишовичок» */
.q-field--focused .q-field__control:after,
.q-field--outlined.q-field--focused .q-field__control:after {
  border-color: var(--brand-plum) !important;
}
.q-field--focused .q-field__label {
  color: var(--brand-plum-hover) !important;
}

/* Ячейки таблицы с 2D-структурами */
.ag-theme-quartz .ag-cell-structure {
  padding: 4px 6px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

/* Главная кнопка запуска расчёта в фирменном цвете «Пишовичок» */
.q-btn.btn-run-primary,
.btn-run-primary {
  background: var(--brand-plum) !important;
  background-color: var(--brand-plum) !important;
  color: #FFFFFF !important;
  font-weight: 500 !important;
}
.q-btn.btn-run-primary:hover,
.btn-run-primary:hover {
  background: var(--brand-plum-hover) !important;
  background-color: var(--brand-plum-hover) !important;
}

/* Отступ и выравнивание иконки с текстом внутри кнопок и плашек (.q-btn__content) */
.q-btn__content:not(:has(> .q-icon.on-left)):not(:has(> .q-icon.on-right)) {
  gap: 6px;
}
.q-btn.gap-1 > .q-btn__content:not(:has(> .q-icon.on-left)):not(:has(> .q-icon.on-right)) {
  gap: 5px;
}
.q-btn.gap-1\\.5 > .q-btn__content:not(:has(> .q-icon.on-left)):not(:has(> .q-icon.on-right)) {
  gap: 6px;
}
.q-btn.gap-2 > .q-btn__content:not(:has(> .q-icon.on-left)):not(:has(> .q-icon.on-right)) {
  gap: 8px;
}
.q-btn__content > div:has(> svg) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}
</style>
"""


def build_theme_head_css() -> str:
    """Возвращает HTML-блок стилей Clean Wet-Lab Light Mode для вставки в `<head>`."""
    return THEME_CSS
