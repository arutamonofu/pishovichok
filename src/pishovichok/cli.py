"""Интерфейс командной строки (CLI) для мультиагентного скрининга и запуска NiceGUI Chemical IDE."""

from __future__ import annotations

import argparse

from pishovichok import __version__
from pishovichok.agents.workflow import run_sar_pipeline
from pishovichok.config import settings


def main() -> None:
    """Точка входа CLI: консольный запуск пайплайна (-q) или запуск веб-интерфейса NiceGUI (--ui)."""
    parser = argparse.ArgumentParser(
        prog="pishovichok",
        description="Мультиагентный SAR-скрининг и направленный дизайн малых молекул (Chemical IDE)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Запустить интерактивную рабочую среду NiceGUI Chemical IDE",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.ui_host,
        help="Хост веб-сервера NiceGUI (по умолчанию из AppSettings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.ui_port,
        help="Порт веб-сервера NiceGUI (по умолчанию 8080)",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Запустить NiceGUI в нативном desktop-окне (pywebview)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Запрос на естественном языке для автономного запуска в консоли",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Путь к загруженному CSV-файлу с серией биоиспытаний (для Сценария 3)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Пользовательский идентификатор сессии (по умолчанию автогенерация)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="workspace",
        help="Директория для сохранения промежуточных Parquet и артефактов",
    )

    args = parser.parse_args()

    if args.ui or not args.query:
        from pishovichok.ui.app import run_ui

        run_ui(host=args.host, port=args.port, native=args.native)
        return

    print(f"[*] Запуск конвейера Pishovichok v{__version__}...")
    print(f"[*] Запрос: {args.query}")
    if args.csv:
        print(f"[*] Входной CSV: {args.csv}")

    final_state, trace_path, csv_path, sdf_path = run_sar_pipeline(
        user_query=args.query,
        session_id=args.session_id,
        workspace_dir=args.workspace_dir,
        uploaded_csv_path=args.csv,
        export_artifacts=True,
    )

    top_cands = final_state.get("top_candidates", [])
    print(f"[+] Сессия завершена. Получено кандидатов: {len(top_cands)}")
    for i, c in enumerate(top_cands, 1):
        print(
            f"  {i}. {c.candidate_id} | SMILES: {c.canonical_smiles} | "
            f"cLogP: {c.calc_rdkit.clogp:.2f} | SAscore: {c.calc_rdkit.sa_score:.2f} | "
            f"Harrington D: {c.mpo_harrington_score:.3f} | All constraints: {c.passes_all_constraints}"
        )

    if trace_path:
        print(f"[+] Журнал трассировки: {trace_path}")
    if csv_path:
        print(f"[+] Таблица кандидатов CSV: {csv_path}")
    if sdf_path:
        print(f"[+] Файл 2D-структур SDF: {sdf_path}")


if __name__ == "__main__":
    main()
