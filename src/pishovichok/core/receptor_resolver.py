"""Универсальный резолвер 3D-структур белковых мишеней (Слой 1).

Обеспечивает:
1. Мгновенный доступ к предкэшированным демо-мишеням (data/receptors/).
2. Автоматический поиск PDB ID для произвольного ChEMBL Target ID через ChEMBL REST API
   и RCSB Search API (с фильтрацией по наличию ко-кристаллизованного лиганда и разрешению).
3. Скачивание PDB-структуры, автоопределение координат активного центра (бокса докинга)
   и ключевых аминокислотных остатков в радиусе 5.0 Å (или апо-центроида CA-атомов).
4. Очистку белка, конвертацию в AutoDock PDBQT и локальное кэширование метаданных.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx

from pishovichok.config import (
    DEMO_RECEPTOR_PRESETS,
    RECEPTORS_DIR,
    settings,
)

CHEMBL_TARGET_API = "https://www.ebi.ac.uk/chembl/api/data/target/{target_id}.json"
RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Вода, ионы и стандартные кристаллографические буферы, исключаемые при поиске ко-лиганда
IGNORED_HETATM_RESNAMES: frozenset[str] = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "H2O",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
        "PEG",
        "PGE",
        "PG4",
        "1PE",
        "P6G",
        "CL",
        "NA",
        "K",
        "MG",
        "ZN",
        "CA",
        "FE",
        "FE2",
        "MN",
        "CO",
        "NI",
        "CU",
        "CD",
        "BR",
        "IOD",
        "DMS",
        "DMSO",
        "ACT",
        "ACY",
        "BME",
        "MES",
        "TRS",
        "EPE",
        "MPD",
        "IPA",
        "ACE",
        "NME",
        "FMT",
        "CIT",
        "TAR",
        "SUC",
        "NAG",
        "BMA",
        "MAN",
        "FUC",
    }
)

_AROMATIC_RESIDUE_ATOMS: dict[str, frozenset[str]] = {
    "PHE": frozenset({"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "TYR": frozenset({"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "TRP": frozenset({"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}),
    "HIS": frozenset({"CG", "ND1", "CD2", "CE1", "NE2"}),
}


def _resolve_receptors_root(receptors_dir: Path | str | None = None) -> Path:
    if receptors_dir is not None:
        return Path(receptors_dir)
    if settings.data_dir is not None:
        return settings.receptors_dir
    return RECEPTORS_DIR


def _infer_element(atom_name: str, explicit_element: str = "") -> str:
    elem = explicit_element.strip().upper()
    if elem and elem.isalpha():
        return elem[:2]
    stripped = atom_name.strip()
    if not stripped:
        return "C"
    # Берём ведущие буквы (учитывая сдвиг типа 1HD2)
    letters = "".join(ch for ch in stripped if ch.isalpha())
    if not letters:
        return "C"
    if letters[0] == "H":
        return "H"
    return letters[0].upper()


def _assign_autodock_atom_type(
    element: str,
    atom_name: str,
    res_name: str,
) -> tuple[str, float]:
    """Назначает тип атома AutoDock (C, A, N, NA, OA, S, SA, HD) и частичный заряд."""
    el = element.upper()
    aname = atom_name.strip().upper()
    rname = res_name.strip().upper()

    if el == "H":
        if aname in {"H", "HN", "HZ1", "HZ2", "HZ3", "HH", "HG", "HG1", "HD1", "HD2", "HE1", "HE2", "NE1"}:
            return "HD", 0.25
        return "HD", 0.20

    if el == "C":
        if aname in _AROMATIC_RESIDUE_ATOMS.get(rname, frozenset()):
            return "A", 0.02
        if aname in {"C", "CG", "CD"}:
            return "C", 0.15
        return "C", 0.04

    if el == "N":
        if rname == "HIS" and aname in {"ND1", "NE2"}:
            return "NA", -0.35
        return "N", -0.35

    if el == "O":
        return "OA", -0.45

    if el == "S":
        if rname in {"MET", "CYS"}:
            return "SA", -0.12
        return "S", -0.10

    if len(el) == 1:
        return el, 0.0
    return el[:2], 0.0


def _pdb_line_to_pdbqt_line(line: str) -> str | None:
    """Преобразует строку ATOM из PDB в корректный фиксированный формат AutoDock PDBQT."""
    if not line.startswith("ATOM  ") or len(line) < 54:
        return None
    atom_name = line[12:16]
    res_name = line[17:20]
    explicit_el = line[76:78] if len(line) >= 78 else ""
    element = _infer_element(atom_name, explicit_el)
    ad_type, charge = _assign_autodock_atom_type(element, atom_name, res_name)

    prefix = line[:54].ljust(54)
    occ = line[54:60] if len(line) >= 60 and line[54:60].strip() else "  1.00"
    bfac = line[60:66] if len(line) >= 66 and line[60:66].strip() else "  0.00"
    return f"{prefix}{occ[:6]:>6}{bfac[:6]:>6}    {charge:6.3f} {ad_type:<2}\n"


def _convert_pdb_lines_to_clean_pdb_and_pdbqt(
    raw_pdb_text: str,
) -> tuple[list[str], list[str]]:
    """Оставляет только белковые ATOM-строки и формирует соответствующие строки PDBQT."""
    clean_pdb: list[str] = []
    pdbqt_lines: list[str] = []
    for raw_line in raw_pdb_text.splitlines():
        if raw_line.startswith("ATOM  ") and len(raw_line) >= 54:
            alt_loc = raw_line[16] if len(raw_line) > 16 else " "
            if alt_loc not in {" ", "A"}:
                continue
            normalized_line = raw_line[:16] + " " + raw_line[17:]
            if len(normalized_line) < 78:
                el = _infer_element(normalized_line[12:16])
                normalized_line = normalized_line.ljust(76) + f"{el:>2}"
            clean_pdb.append(normalized_line.rstrip("\r\n") + "\n")
            q_line = _pdb_line_to_pdbqt_line(normalized_line)
            if q_line is not None:
                pdbqt_lines.append(q_line)
    if clean_pdb:
        clean_pdb.append("END\n")
    return clean_pdb, pdbqt_lines


def _extract_pocket_from_pdb(
    raw_pdb_text: str,
    preset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Извлекает координаты кармана связывания и ключевые остатки из PDB-текста."""
    protein_atoms: list[tuple[str, int, str, float, float, float]] = []
    ca_coords: list[tuple[float, float, float]] = []
    het_ligands: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}

    for line in raw_pdb_text.splitlines():
        if len(line) < 54:
            continue
        rec = line[:6]
        if rec not in {"ATOM  ", "HETATM"}:
            continue
        alt_loc = line[16] if len(line) > 16 else " "
        if alt_loc not in {" ", "A"}:
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip().upper()
        chain_id = line[21].strip() or "A"
        res_seq_raw = line[22:26].strip()
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        explicit_el = line[76:78] if len(line) >= 78 else ""
        element = _infer_element(atom_name, explicit_el)

        if rec == "ATOM  ":
            if element == "H":
                continue
            try:
                res_seq_int = int(res_seq_raw)
            except ValueError:
                continue
            protein_atoms.append((res_name, res_seq_int, atom_name, x, y, z))
            if atom_name == "CA":
                ca_coords.append((x, y, z))
        else:
            if element == "H" or res_name in IGNORED_HETATM_RESNAMES:
                continue
            key = (chain_id, res_seq_raw, res_name)
            het_ligands.setdefault(key, []).append((x, y, z))

    # Выбор самого крупного органического ко-кристаллизованного лиганда (>= 7 тяжёлых атомов)
    valid_ligands = [
        (key, coords) for key, coords in het_ligands.items() if len(coords) >= 7
    ]

    if valid_ligands:
        best_key, best_coords = max(valid_ligands, key=lambda item: len(item[1]))
        cx = sum(p[0] for p in best_coords) / len(best_coords)
        cy = sum(p[1] for p in best_coords) / len(best_coords)
        cz = sum(p[2] for p in best_coords) / len(best_coords)
        box_center = [round(cx, 3), round(cy, 3), round(cz, 3)]
        box_size = [20.0, 20.0, 20.0]
        ligand_resname = best_key[2]
        is_apo = False

        # Остатки белка в радиусе 5.0 Å от ко-лиганда
        residue_min_dist: dict[tuple[int, str], float] = {}
        for res_name, res_seq, _, px, py, pz in protein_atoms:
            min_d2 = min(
                (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2
                for lx, ly, lz in best_coords
            )
            if min_d2 <= 25.0:  # 5.0^2
                rlabel = f"{res_name}{res_seq}"
                prev = residue_min_dist.get((res_seq, rlabel))
                d = math.sqrt(min_d2)
                if prev is None or d < prev:
                    residue_min_dist[(res_seq, rlabel)] = d

        key_residues = [
            label
            for (_, label), _dist in sorted(
                residue_min_dist.items(), key=lambda kv: kv[1]
            )[:12]
        ]
    else:
        # Апо-форма: центроид CA-атомов
        coords_src = ca_coords if ca_coords else [(p[3], p[4], p[5]) for p in protein_atoms]
        if coords_src:
            cx = sum(p[0] for p in coords_src) / len(coords_src)
            cy = sum(p[1] for p in coords_src) / len(coords_src)
            cz = sum(p[2] for p in coords_src) / len(coords_src)
        else:
            cx, cy, cz = 0.0, 0.0, 0.0
        box_center = [round(cx, 3), round(cy, 3), round(cz, 3)]
        box_size = [26.0, 26.0, 26.0]
        ligand_resname = None
        is_apo = True

        residue_min_dist_apo: dict[tuple[int, str], float] = {}
        for res_name, res_seq, _, px, py, pz in protein_atoms:
            d2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
            if d2 <= 36.0:  # 6.0^2
                rlabel = f"{res_name}{res_seq}"
                d = math.sqrt(d2)
                prev = residue_min_dist_apo.get((res_seq, rlabel))
                if prev is None or d < prev:
                    residue_min_dist_apo[(res_seq, rlabel)] = d
        key_residues = [
            label
            for (_, label), _dist in sorted(
                residue_min_dist_apo.items(), key=lambda kv: kv[1]
            )[:12]
        ]

    if preset_info is not None:
        # Для демо-пресетов сохраняем эталонные координаты кармана, если они согласованы
        # с реальным ко-кристаллизованным лигандом (или если структура без лиганда)
        if preset_info.get("box_center"):
            preset_center = [float(v) for v in preset_info["box_center"]]
            if is_apo:
                box_center = preset_center
            else:
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(box_center, preset_center))
                )
                if dist <= 8.0:
                    box_center = preset_center
        if preset_info.get("box_size"):
            box_size = [float(v) for v in preset_info["box_size"]]
        if preset_info.get("key_residues"):
            merged = list(preset_info["key_residues"])
            for r in key_residues:
                if r not in merged:
                    merged.append(r)
            key_residues = merged[:12]

    return {
        "box_center": box_center,
        "box_size": box_size,
        "key_residues": key_residues,
        "ligand_resname": ligand_resname,
        "is_apo": is_apo,
    }


def _find_pdb_id_for_chembl_target(target_id: str, timeout: float = 8.0) -> str | None:
    """Ищет лучший PDB ID через ChEMBL REST API и RCSB Search API."""
    url = CHEMBL_TARGET_API.format(target_id=target_id)
    pdb_candidates: list[str] = []
    uniprot_ids: list[str] = []

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return None
            payload = resp.json()

            for comp in payload.get("target_components", []) or []:
                acc = comp.get("accession")
                if acc and str(acc).strip():
                    uniprot_ids.append(str(acc).strip())
                for xref in comp.get("target_component_xrefs", []) or []:
                    src_db = str(xref.get("xref_src_db", "")).upper()
                    xid = str(xref.get("xref_id", "")).strip()
                    if not xid:
                        continue
                    if src_db == "PDB":
                        pdb_candidates.append(xid.upper())
                    elif src_db == "UNIPROT" and xid not in uniprot_ids:
                        uniprot_ids.append(xid)

            for xref in payload.get("cross_references", []) or []:
                src_db = str(xref.get("xref_src", "")).upper()
                xid = str(xref.get("xref_id", "")).strip()
                if src_db == "PDB" and xid:
                    pdb_candidates.append(xid.upper())

            # При наличии UniProt ID сначала ищем в RCSB структуру с ко-кристаллизованным лигандом
            for uniprot_id in uniprot_ids[:2]:
                for require_ligand in (True, False):
                    nodes: list[dict[str, Any]] = [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                                "operator": "exact_match",
                                "value": uniprot_id,
                            },
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                                "operator": "exact_match",
                                "value": "UniProt",
                            },
                        },
                    ]
                    if require_ligand:
                        nodes.append(
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                                    "operator": "greater_or_equal",
                                    "value": 1,
                                },
                            }
                        )
                    query_body = {
                        "query": {
                            "type": "group",
                            "logical_operator": "and",
                            "nodes": nodes,
                        },
                        "return_type": "entry",
                        "request_options": {
                            "paginate": {"start": 0, "rows": 5},
                            "sort": [
                                {
                                    "sort_by": "rcsb_entry_info.resolution_combined",
                                    "direction": "asc",
                                }
                            ],
                        },
                    }
                    try:
                        rcsb_resp = client.post(RCSB_SEARCH_API, json=query_body)
                        if rcsb_resp.status_code == 200:
                            r_json = rcsb_resp.json()
                            results = r_json.get("result_set", []) or []
                            if results:
                                ident = str(results[0].get("identifier", "")).strip().upper()
                                if len(ident) == 4:
                                    return ident
                    except httpx.HTTPError:
                        continue
    except (httpx.HTTPError, ValueError):
        return None

    return pdb_candidates[0][:4] if pdb_candidates else None


def _parse_residue_label(label: str) -> tuple[str, int]:
    """Разбирает метку остатка вида MET769 на ('MET', 769)."""
    clean = label.strip().upper()
    alpha = "".join(ch for ch in clean if ch.isalpha()) or "ALA"
    digits = "".join(ch for ch in clean if ch.isdigit())
    res_seq = int(digits) if digits else 1
    return alpha[:3], res_seq


def generate_minimal_demo_receptor(
    target_id: str,
    receptors_dir: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Генерирует минимальные эталонные файлы .pdb, .pdbqt и .json для офлайн-демо."""
    target_clean = target_id.strip().upper()
    preset = DEMO_RECEPTOR_PRESETS.get(target_clean)
    if preset is None:
        raise ValueError(f"Мишень {target_id!r} не найдена в DEMO_RECEPTOR_PRESETS.")

    root = _resolve_receptors_root(receptors_dir)
    root.mkdir(parents=True, exist_ok=True)

    pdb_path = root / f"{target_clean}.pdb"
    pdbqt_path = root / f"{target_clean}.pdbqt"
    json_path = root / f"{target_clean}.json"

    if (
        not overwrite
        and pdb_path.exists()
        and pdbqt_path.exists()
        and json_path.exists()
        and pdb_path.stat().st_size > 0
        and pdbqt_path.stat().st_size > 0
    ):
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["status"] = "READY"
        meta["pdb_path"] = pdb_path.as_posix()
        meta["pdbqt_path"] = pdbqt_path.as_posix()
        return meta

    cx, cy, cz = [float(v) for v in preset["box_center"]]
    box_size = [float(v) for v in preset["box_size"]]
    key_residues = [str(r) for r in preset["key_residues"]]
    pdb_id = str(preset["pdb_id"])

    # Формируем реалистичную 3D-оболочку кармана вокруг box_center (~4.5-5.5 Å от центра)
    # с корректными координатами N, CA, C, O и ключевого бокового атома для каждого остатка.
    sidechain_atom_spec: dict[str, tuple[str, str]] = {
        "MET": ("SD", "S"),
        "LEU": ("CD1", "C"),
        "VAL": ("CG1", "C"),
        "ALA": ("CB", "C"),
        "LYS": ("NZ", "N"),
        "GLU": ("OE1", "O"),
        "ASP": ("OD1", "O"),
        "THR": ("OG1", "O"),
        "TYR": ("OH", "O"),
        "PHE": ("CZ", "C"),
        "TRP": ("NE1", "N"),
        "PRO": ("CG", "C"),
        "HIS": ("NE2", "N"),
    }

    pdb_lines: list[str] = [
        f"HEADER    MINIMAL OFFLINE POCKET RECEPTOR FOR {target_clean} ({pdb_id})\n"
    ]
    serial = 1
    n_res = max(len(key_residues), 1)
    for idx, rlabel in enumerate(key_residues):
        res_name, res_seq = _parse_residue_label(rlabel)
        theta = (2.0 * math.pi * idx) / n_res
        phi = ((idx % 3) - 1) * 0.45
        radius_bb = 5.4
        ca_x = cx + radius_bb * math.cos(theta) * math.cos(phi)
        ca_y = cy + radius_bb * math.sin(theta) * math.cos(phi)
        ca_z = cz + radius_bb * math.sin(phi)

        # Вектор внутрь кармана для боковой цепи
        ux = -math.cos(theta) * math.cos(phi)
        uy = -math.sin(theta) * math.cos(phi)
        uz = -math.sin(phi)

        sc_name, sc_elem = sidechain_atom_spec.get(res_name, ("CB", "C"))
        atoms_for_res = [
            ("N", "N", ca_x - 1.25, ca_y - 0.65, ca_z),
            ("CA", "C", ca_x, ca_y, ca_z),
            ("C", "C", ca_x + 1.30, ca_y + 0.55, ca_z),
            ("O", "O", ca_x + 1.85, ca_y + 1.55, ca_z + 0.45),
            ("CB", "C", ca_x + 1.05 * ux, ca_y + 1.05 * uy, ca_z + 1.05 * uz),
        ]
        if sc_name != "CB":
            atoms_for_res.append(
                (sc_name, sc_elem, ca_x + 2.15 * ux, ca_y + 2.15 * uy, ca_z + 2.15 * uz)
            )

        for aname, aelem, ax, ay, az in atoms_for_res:
            atom_col = f" {aname:<3}" if len(aname) <= 3 else f"{aname:<4}"
            pdb_lines.append(
                f"ATOM  {serial:5d} {atom_col} {res_name:>3} A{res_seq:4d}    "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00 20.00          {aelem:>2}\n"
            )
            serial += 1

    pdb_lines.append("TER\nEND\n")
    raw_pdb = "".join(pdb_lines)
    clean_pdb_lines, pdbqt_lines = _convert_pdb_lines_to_clean_pdb_and_pdbqt(raw_pdb)

    pdb_path.write_text("".join(clean_pdb_lines), encoding="utf-8")
    pdbqt_path.write_text("".join(pdbqt_lines), encoding="utf-8")

    meta: dict[str, Any] = {
        "status": "READY",
        "target_id": target_clean,
        "pdb_id": pdb_id,
        "source": "DEMO_CACHE",
        "pdb_path": pdb_path.as_posix(),
        "pdbqt_path": pdbqt_path.as_posix(),
        "box_center": [cx, cy, cz],
        "box_size": box_size,
        "key_residues": key_residues,
        "ligand_resname": None,
        "is_apo": False,
        "minimal_offline_reference": True,
    }
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def resolve_receptor(
    target_id: str,
    custom_pdb_id: str | None = None,
    custom_box_center: list[float] | None = None,
    receptors_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Находит, скачивает, очищает и кэширует 3D-структуру белка-мишени и координаты кармана.

    Шаг 1: Проверка локального кэша data/receptors/{cache_key}.pdbqt и .json.
    Шаг 2: Поиск PDB ID для произвольного target_id через ChEMBL REST / RCSB Search API.
    Шаг 3: Скачивание PDB, поиск ко-кристаллизованного лиганда (>= 7 тяжёлых атомов)
           и вычисление центроида + ключевых остатков в радиусе 5.0 Å (или апо-центроида CA).
    Шаг 4: Очистка ATOM-записей, конвертация в AutoDock PDBQT и сохранение метаданных.
    Шаг 5: Безопасный возврат {'status': 'NO_STRUCTURE', 'reason': ...} при отсутствии 3D-данных.
    """
    target_clean = (target_id or "").strip().upper()
    pdb_override = custom_pdb_id.strip().upper() if custom_pdb_id and custom_pdb_id.strip() else None

    if not target_clean and not pdb_override:
        return {
            "status": "NO_STRUCTURE",
            "reason": "Не указан ни ChEMBL Target ID, ни пользовательский PDB ID.",
        }

    root = _resolve_receptors_root(receptors_dir)
    root.mkdir(parents=True, exist_ok=True)

    preset = DEMO_RECEPTOR_PRESETS.get(target_clean) if target_clean else None
    cache_key = (
        f"{target_clean}_{pdb_override}"
        if (target_clean and pdb_override)
        else (pdb_override or target_clean)
    )

    pdb_path = root / f"{cache_key}.pdb"
    pdbqt_path = root / f"{cache_key}.pdbqt"
    json_path = root / f"{cache_key}.json"

    # Шаг 1: Проверка кэша
    if (
        pdbqt_path.exists()
        and json_path.exists()
        and pdb_path.exists()
        and pdbqt_path.stat().st_size > 0
        and pdb_path.stat().st_size > 0
    ):
        try:
            cached_meta = json.loads(json_path.read_text(encoding="utf-8"))
            cached_meta["status"] = "READY"
            cached_meta["pdb_path"] = pdb_path.as_posix()
            cached_meta["pdbqt_path"] = pdbqt_path.as_posix()
            if custom_box_center is not None and len(custom_box_center) == 3:
                cached_meta["box_center"] = [float(v) for v in custom_box_center]
                cached_meta["box_size"] = [20.0, 20.0, 20.0]
            return cached_meta
        except (json.JSONDecodeError, OSError):
            pass

    # Шаг 2: Определение PDB ID
    resolved_pdb_id = pdb_override
    if resolved_pdb_id is None and preset is not None:
        resolved_pdb_id = str(preset["pdb_id"]).upper()
    if resolved_pdb_id is None and target_clean:
        resolved_pdb_id = _find_pdb_id_for_chembl_target(target_clean)

    if not resolved_pdb_id:
        return {
            "status": "NO_STRUCTURE",
            "target_id": target_clean or None,
            "reason": (
                f"В ChEMBL и RCSB PDB не найдено экспериментальных 3D-структур для мишени {target_clean}."
            ),
        }

    # Шаг 3: Скачивание PDB-структуры
    raw_pdb_text: str | None = None
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            resp = client.get(RCSB_PDB_URL.format(pdb_id=resolved_pdb_id))
            if resp.status_code == 200 and "ATOM  " in resp.text:
                raw_pdb_text = resp.text
    except httpx.HTTPError:
        raw_pdb_text = None

    if raw_pdb_text is None:
        # Для 3 демо-пресетов при отсутствии интернета создаём локальный эталонный кэш
        if preset is not None and pdb_override is None:
            meta = generate_minimal_demo_receptor(target_clean, receptors_dir=root)
            if custom_box_center is not None and len(custom_box_center) == 3:
                meta["box_center"] = [float(v) for v in custom_box_center]
            return meta
        return {
            "status": "NO_STRUCTURE",
            "target_id": target_clean or None,
            "pdb_id": resolved_pdb_id,
            "reason": (
                f"Не удалось загрузить PDB-файл {resolved_pdb_id} из RCSB PDB (отсутствует сеть или структура недоступна)."
            ),
        }

    # Автоопределение кармана и ключевых остатков
    pocket = _extract_pocket_from_pdb(
        raw_pdb_text,
        preset_info=preset if pdb_override is None else None,
    )
    if custom_box_center is not None and len(custom_box_center) == 3:
        pocket["box_center"] = [float(v) for v in custom_box_center]
        pocket["box_size"] = [20.0, 20.0, 20.0]

    # Шаг 4: Очистка ATOM и конвертация в PDBQT
    clean_pdb_lines, pdbqt_lines = _convert_pdb_lines_to_clean_pdb_and_pdbqt(raw_pdb_text)
    if not clean_pdb_lines or not pdbqt_lines:
        return {
            "status": "NO_STRUCTURE",
            "target_id": target_clean or None,
            "pdb_id": resolved_pdb_id,
            "reason": f"PDB-файл {resolved_pdb_id} не содержит валидных белковых ATOM-записей.",
        }

    pdb_path.write_text("".join(clean_pdb_lines), encoding="utf-8")
    pdbqt_path.write_text("".join(pdbqt_lines), encoding="utf-8")

    source_label = "DEMO_CACHE" if (preset is not None and pdb_override is None) else "RCSB_LIVE"
    meta = {
        "status": "READY",
        "target_id": target_clean or None,
        "pdb_id": resolved_pdb_id,
        "source": source_label,
        "pdb_path": pdb_path.as_posix(),
        "pdbqt_path": pdbqt_path.as_posix(),
        "box_center": pocket["box_center"],
        "box_size": pocket["box_size"],
        "key_residues": pocket["key_residues"],
        "ligand_resname": pocket["ligand_resname"],
        "is_apo": pocket["is_apo"],
        "minimal_offline_reference": False,
    }
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


__all__ = [
    "IGNORED_HETATM_RESNAMES",
    "generate_minimal_demo_receptor",
    "resolve_receptor",
]
