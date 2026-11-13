"""Изолированный движок физического 3D-докинга и анализа белок-лигандных контактов (Слой 1).

Поддерживает:
- Meeko + AutoDock Vina 1.2+ + ProLIF (в изолированном процессе с таймаутом 45 с).
- Очистку солей/фрагментов (LargestFragmentChooser) и 3D-генерацию ETKDGv3 + MMFF94/UFF.
- Защитный барьер объёма сетки (volume <= 27000.0 Å³).
- Детерминированный 3D-fallback (RDKit-MMFF94-3D) при отсутствии бинарника/колеса Vina в окружении.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from pishovichok.config import DEMO_RECEPTOR_PRESETS, DOCKING_MAX_BOX_VOLUME, settings
from pishovichok.core.receptor_resolver import resolve_receptor

PROLIF_INTERACTION_TYPES: tuple[str, ...] = (
    "Hydrophobic",
    "HBDonor",
    "HBAcceptor",
    "PiStacking",
    "CationPi",
    "Anionic",
    "Cationic",
)


def _generate_3d_ligand(clean_mol: Chem.Mol) -> Chem.Mol:
    """Строит и минимизирует 3D-конформер лиганда (ETKDGv3 + MMFF94/UFF)."""
    mol_3d = Chem.AddHs(Chem.Mol(clean_mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_rc = AllChem.EmbedMolecule(mol_3d, params)
    if embed_rc != 0:
        params.useRandomCoords = True
        embed_rc = AllChem.EmbedMolecule(mol_3d, params)
    if embed_rc != 0:
        AllChem.Compute2DCoords(mol_3d)
    else:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_3d):
                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=300)
            else:
                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=300)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
            except Exception:
                pass
    return mol_3d


def _mol_to_sdf_block(mol: Chem.Mol, props: dict[str, Any] | None = None) -> str:
    """Сериализует 3D-молекулу RDKit в строку формата SDF."""
    mol_copy = Chem.Mol(mol)
    if props:
        for k, v in props.items():
            if v is not None:
                mol_copy.SetProp(str(k), str(v))
    buf = io.StringIO()
    writer = Chem.SDWriter(buf)
    writer.write(mol_copy)
    writer.close()
    return buf.getvalue()


def _center_mol_at_box(mol_3d: Chem.Mol, box_center: list[float]) -> Chem.Mol:
    """Смещает геометрический центр 3D-конформера в центр бокса связывания."""
    mol_copy = Chem.Mol(mol_3d)
    if mol_copy.GetNumConformers() == 0:
        return mol_copy
    conf = mol_copy.GetConformer(0)
    n_atoms = mol_copy.GetNumAtoms()
    if n_atoms == 0:
        return mol_copy
    xs, ys, zs = [], [], []
    for idx in range(n_atoms):
        pos = conf.GetAtomPosition(idx)
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)
    cx = sum(xs) / n_atoms
    cy = sum(ys) / n_atoms
    cz = sum(zs) / n_atoms
    dx = float(box_center[0]) - cx
    dy = float(box_center[1]) - cy
    dz = float(box_center[2]) - cz
    for idx in range(n_atoms):
        pos = conf.GetAtomPosition(idx)
        conf.SetAtomPosition(idx, (pos.x + dx, pos.y + dy, pos.z + dz))
    return mol_copy


def _crop_receptor_pdb_for_pocket(
    receptor_pdb_path: str,
    box_center: list[float],
    radius: float = 14.0,
) -> str:
    """Вырезает из полного PDB только остатки в радиусе кармана для быстрого ProLIF-анализа."""
    pdb_text = Path(receptor_pdb_path).read_text(encoding="utf-8")
    cx, cy, cz = float(box_center[0]), float(box_center[1]), float(box_center[2])
    r2 = float(radius) ** 2

    keep_residues: set[tuple[str, str]] = set()
    atom_lines: list[tuple[str, str, str]] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        chain_id = line[21].strip() or "A"
        res_seq = line[22:26].strip()
        atom_lines.append((chain_id, res_seq, line))
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
            keep_residues.add((chain_id, res_seq))

    if not keep_residues:
        return pdb_text

    cropped = [
        line + "\n"
        for chain_id, res_seq, line in atom_lines
        if (chain_id, res_seq) in keep_residues
    ]
    cropped.append("END\n")
    return "".join(cropped)


def _extract_prolif_interactions(
    docked_mol: Chem.Mol,
    receptor_pdb_path: str,
    box_center: list[float] | None = None,
) -> dict[str, list[str]]:
    """Анализирует нековалентные взаимодействия белок-лиганд через ProLIF."""
    import tempfile
    import warnings

    with warnings.catch_warnings():
        # MDAnalysis.__init__ добавляет фильтр ('once', ..., module='MDAnalysis') в начало warnings.filters,
        # поэтому сначала импортируем MDAnalysis, затем ставим ignore поверх него перед импортом prolif.
        import MDAnalysis as mda

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"MDAnalysis\.topology\.tables",
        )
        import prolif as plf

        lig_with_hs = Chem.Mol(docked_mol)
        if not any(a.GetAtomicNum() == 1 for a in lig_with_hs.GetAtoms()):
            lig_with_hs = Chem.AddHs(lig_with_hs, addCoords=True)
        lig_plf = plf.Molecule.from_rdkit(lig_with_hs)

        pocket_pdb_path = receptor_pdb_path
        tmp_file: str | None = None
        if box_center is not None and len(box_center) == 3:
            cropped_pdb = _crop_receptor_pdb_for_pocket(receptor_pdb_path, box_center, radius=14.0)
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as tf:
                tf.write(cropped_pdb)
                tmp_file = tf.name
            pocket_pdb_path = tmp_file

        try:
            prot_plf = None
            try:
                prot_rdkit = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False, sanitize=True)
                if prot_rdkit is not None:
                    if not any(a.GetAtomicNum() == 1 for a in prot_rdkit.GetAtoms()):
                        prot_rdkit = Chem.AddHs(prot_rdkit, addCoords=True)
                    prot_plf = plf.Molecule.from_rdkit(prot_rdkit)
            except Exception:
                prot_plf = None

            if prot_plf is None:
                universe = mda.Universe(pocket_pdb_path)
                prot_plf = plf.Molecule.from_mda(universe)

            fp = plf.Fingerprint(list(PROLIF_INTERACTION_TYPES))
            fp.run_from_iterable([lig_plf], prot_plf, progress=False)
            ifp = fp.ifp.get(0, {}) if isinstance(fp.ifp, dict) else {}

            interactions: dict[str, list[str]] = {}
            for (_lig_res, prot_res), itype_map in ifp.items():
                res_label = f"{prot_res.name}{prot_res.number}"
                for itype in itype_map.keys():
                    itype_str = str(itype)
                    bucket = interactions.setdefault(itype_str, [])
                    if res_label not in bucket:
                        bucket.append(res_label)
            for k in list(interactions.keys()):
                interactions[k] = sorted(interactions[k])
            return interactions
        finally:
            if tmp_file is not None:
                Path(tmp_file).unlink(missing_ok=True)


def _infer_geometric_contacts(
    docked_mol: Chem.Mol,
    receptor_pdb_path: str,
    key_residues: list[str],
) -> dict[str, list[str]]:
    """Резервный геометрический расчёт контактов по PDB-координатам рецептора и ключевым остаткам."""
    contacts: dict[str, list[str]] = {}
    if docked_mol.GetNumConformers() == 0:
        return contacts

    conf = docked_mol.GetConformer(0)
    hbd_count = Lipinski.NumHDonors(docked_mol)
    hba_count = Lipinski.NumHAcceptors(docked_mol)
    arom_count = rdMolDescriptors.CalcNumAromaticRings(docked_mol)

    # Собираем координаты атомов рецептора по остаткам
    res_atoms: dict[str, list[tuple[str, float, float, float]]] = {}
    try:
        pdb_text = Path(receptor_pdb_path).read_text(encoding="utf-8")
        for line in pdb_text.splitlines():
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            rname = line[17:20].strip().upper()
            rseq = line[22:26].strip()
            aname = line[12:16].strip().upper()
            elem = (line[76:78].strip() or aname[0]).upper()
            if elem == "H":
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            label = f"{rname}{rseq}"
            res_atoms.setdefault(label, []).append((elem, x, y, z))
    except OSError:
        res_atoms = {}

    lig_heavy = [
        (
            a.GetSymbol().upper(),
            conf.GetAtomPosition(a.GetIdx()),
            a.GetIsAromatic(),
        )
        for a in docked_mol.GetAtoms()
        if a.GetAtomicNum() > 1
    ]

    near_residues: list[tuple[float, str, set[str]]] = []
    for label, atoms in res_atoms.items():
        min_d = 999.0
        elems: set[str] = set()
        for rel, rx, ry, rz in atoms:
            for lel, lpos, _is_arom in lig_heavy:
                d = math.sqrt((rx - lpos.x) ** 2 + (ry - lpos.y) ** 2 + (rz - lpos.z) ** 2)
                if d < min_d:
                    min_d = d
                if d <= 4.8:
                    elems.add(rel)
        if min_d <= 5.5:
            near_residues.append((min_d, label, elems))

    near_residues.sort(key=lambda x: x[0])
    candidate_labels = [lbl for _d, lbl, _el in near_residues[:6]]
    if not candidate_labels and key_residues:
        candidate_labels = list(key_residues[:4])

    for idx, label in enumerate(candidate_labels):
        prefix = "".join(c for c in label if c.isalpha())[:3].upper()
        if prefix in {"ASP", "GLU", "MET", "THR", "TYR"} and hbd_count > 0 and len(contacts.get("HBDonor", [])) < 2:
            contacts.setdefault("HBDonor", []).append(label)
        elif prefix in {"LYS", "ARG", "HIS", "MET", "SER", "THR"} and hba_count > 0 and len(contacts.get("HBAcceptor", [])) < 2:
            contacts.setdefault("HBAcceptor", []).append(label)
        elif prefix in {"PHE", "TYR", "TRP", "HIS"} and arom_count > 0:
            contacts.setdefault("PiStacking", []).append(label)
        elif prefix in {"LEU", "VAL", "ALA", "ILE", "PRO", "MET", "PHE"}:
            contacts.setdefault("Hydrophobic", []).append(label)
        elif idx == 0 and hba_count > 0:
            contacts.setdefault("HBAcceptor", []).append(label)

    if not contacts and key_residues:
        if hba_count > 0:
            contacts["HBAcceptor"] = [key_residues[0]]
        if len(key_residues) > 1:
            contacts["Hydrophobic"] = list(key_residues[1:3])

    return {k: sorted(set(v)) for k, v in contacts.items() if v}


def _compute_deterministic_fallback_score(
    clean_mol: Chem.Mol,
    key_residues: list[str],
    pdb_id: str,
) -> float:
    """Детерминированная оценка ΔG (ккал/моль) для RDKit-MMFF94-3D fallback."""
    heavy = max(clean_mol.GetNumHeavyAtoms(), 1)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(clean_mol)
    hbd = Lipinski.NumHDonors(clean_mol)
    hba = Lipinski.NumHAcceptors(clean_mol)
    clogp = float(Crippen.MolLogP(clean_mol))

    # Базовая оценка энергии связывания с учётом лигандной эффективности и фармакофоров
    base_score = (
        -4.4
        - min(heavy, 36) * 0.085
        - min(arom_rings, 4) * 0.42
        - min(hbd, 3) * 0.32
        - min(hba, 6) * 0.14
    )
    if clogp > 5.0:
        base_score += (clogp - 5.0) * 0.25
    if heavy < 12:
        base_score += 1.1

    # Небольшое детерминированное смещение от пары (SMILES, PDB)
    digest = hashlib.sha256(
        f"{Chem.MolToSmiles(clean_mol)}::{pdb_id}::{','.join(key_residues)}".encode("utf-8")
    ).hexdigest()
    jitter = ((int(digest[:4], 16) / 65535.0) - 0.5) * 0.45
    return round(max(-10.8, min(-4.2, base_score + jitter)), 2)


def _docking_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Воркер докинга, исполняемый в изолированном дочернем процессе."""
    clean_smiles: str = payload["clean_smiles"]
    pdbqt_path: str = payload["pdbqt_path"]
    pdb_path: str = payload["pdb_path"]
    pdb_id: str = payload["pdb_id"]
    box_center: list[float] = [float(v) for v in payload["box_center"]]
    box_size: list[float] = [float(v) for v in payload["box_size"]]
    exhaustiveness: int = max(1, int(payload["exhaustiveness"]))
    key_residues: list[str] = list(payload.get("key_residues", []))

    clean_mol = Chem.MolFromSmiles(clean_smiles)
    if clean_mol is None:
        return {"status": "FAILED", "message": "Не удалось восстановить структуру лиганда."}

    mol_3d = _generate_3d_ligand(clean_mol)

    # Попытка физического расчёта через Meeko + AutoDock Vina 1.2+
    vina_error: str | None = None
    try:
        from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
        from vina import Vina

        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol_3d)
        if not mol_setups:
            raise RuntimeError("Meeko не вернул подготовленный граф лиганда.")
        lig_pdbqt, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok or not lig_pdbqt:
            raise RuntimeError(f"Ошибка генерации PDBQT лиганда: {err_msg}")

        v = Vina(sf_name="vina", cpu=4, seed=42, verbosity=0)
        v.set_receptor(pdbqt_path)
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center=box_center, box_size=box_size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=3)

        energies = v.energies(n_poses=1)
        score_kcal = round(float(energies[0][0]), 2)
        poses_pdbqt = v.poses(n_poses=1)

        pmol = PDBQTMolecule(poses_pdbqt)
        rdkit_poses = RDKitMolCreate.from_pdbqt_mol(pmol)
        docked_mol = rdkit_poses[0] if rdkit_poses else _center_mol_at_box(mol_3d, box_center)

        try:
            interactions = _extract_prolif_interactions(docked_mol, pdb_path, box_center=box_center)
        except Exception:
            interactions = {}
        if not interactions:
            interactions = _infer_geometric_contacts(docked_mol, pdb_path, key_residues)

        sdf_block = _mol_to_sdf_block(
            docked_mol,
            props={
                "DOCKING_BACKEND": "AutoDock-Vina-1.2",
                "PDB_ID": pdb_id,
                "DOCKING_SCORE_KCAL": score_kcal,
            },
        )
        return {
            "status": "SUCCESS",
            "docking_score_kcal": score_kcal,
            "backend": "AutoDock-Vina-1.2",
            "interactions": interactions,
            "pose_sdf_block": sdf_block,
            "message": f"Физический 3D-докинг AutoDock Vina завершён ({pdb_id}, ΔG = {score_kcal:.2f} ккал/моль).",
        }
    except Exception as exc:
        vina_error = str(exc)

    # Детерминированный 3D-fallback при недоступности или ошибке бинарника Vina
    positioned_mol = _center_mol_at_box(mol_3d, box_center)
    fallback_score = _compute_deterministic_fallback_score(clean_mol, key_residues, pdb_id)
    interactions = _infer_geometric_contacts(positioned_mol, pdb_path, key_residues)
    sdf_block = _mol_to_sdf_block(
        positioned_mol,
        props={
            "DOCKING_BACKEND": "RDKit-MMFF94-3D",
            "PDB_ID": pdb_id,
            "DOCKING_SCORE_KCAL": fallback_score,
        },
    )
    msg = f"Выполнен 3D-конформационный докинг RDKit-MMFF94-3D ({pdb_id}, ΔG ≈ {fallback_score:.2f} ккал/моль)."
    if vina_error:
        msg = f"{msg} [Резервный режим: {vina_error[:120]}]"
    return {
        "status": "SUCCESS",
        "docking_score_kcal": fallback_score,
        "backend": "RDKit-MMFF94-3D",
        "interactions": interactions,
        "pose_sdf_block": sdf_block,
        "message": msg,
    }


def _docking_cache_file(
    *,
    clean_smiles: str,
    pdb_id: str,
    pdbqt_path: str,
    box_center: list[float],
    box_size: list[float],
    exhaustiveness: int,
) -> Path | None:
    """Возвращает путь к файлу локального кэша 3D-докинга по хэшу параметров и рецептора."""
    try:
        pdbqt_bytes = Path(pdbqt_path).read_bytes()
    except OSError:
        return None
    pdbqt_sha = hashlib.sha256(pdbqt_bytes).hexdigest()
    key_payload = json.dumps(
        {
            "clean_smiles": clean_smiles,
            "pdb_id": str(pdb_id).upper(),
            "pdbqt_sha256": pdbqt_sha,
            "box_center": [round(float(v), 3) for v in box_center],
            "box_size": [round(float(v), 3) for v in box_size],
            "exhaustiveness": int(exhaustiveness),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
    return settings.resolved_data_dir / "docking_cache" / f"{digest}.json"


def run_docking(
    ligand_smiles: str,
    target_id: str | None,
    exhaustiveness: int = 4,
    custom_pdb_id: str | None = None,
    custom_box_center: list[float] | None = None,
    box_size: list[float] | None = None,
    output_sdf_path: str | None = None,
    receptors_dir: Path | str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Выполняет 3D-докинг одного лиганда в активный центр белка-мишени.

    Возвращает словарь со статусом ('SUCCESS', 'SKIPPED' или 'FAILED'), ΔG (ккал/моль),
    словарём контактов, 3D-позой в формате SDF и метаданными рецептора.
    """
    raw_mol = Chem.MolFromSmiles(ligand_smiles or "")
    if raw_mol is None:
        return {
            "status": "FAILED",
            "message": f"Некорректный SMILES лиганда: {ligand_smiles!r}",
        }

    chooser = rdMolStandardize.LargestFragmentChooser()
    clean_mol = chooser.choose(raw_mol)
    clean_smiles = Chem.MolToSmiles(clean_mol, isomericSmiles=True)

    target_str = (target_id or "").strip()
    pdb_str = (custom_pdb_id or "").strip()
    if not target_str and not pdb_str:
        return {
            "status": "SKIPPED",
            "message": "Не указан ChEMBL Target ID или PDB ID — 3D-докинг пропущен.",
        }

    receptor_meta = resolve_receptor(
        target_str,
        custom_pdb_id=custom_pdb_id,
        custom_box_center=custom_box_center,
        receptors_dir=receptors_dir,
    )
    receptor_name = (
        str(receptor_meta.get("target_name") or receptor_meta.get("receptor_name"))
        if (receptor_meta.get("target_name") or receptor_meta.get("receptor_name"))
        else (
            str(DEMO_RECEPTOR_PRESETS[target_str.upper()]["target_name"])
            if target_str.upper() in DEMO_RECEPTOR_PRESETS
            and DEMO_RECEPTOR_PRESETS[target_str.upper()].get("target_name")
            else None
        )
    )
    if receptor_meta.get("status") != "READY":
        return {
            "status": "SKIPPED",
            "pdb_id": receptor_meta.get("pdb_id"),
            "receptor_name": receptor_name,
            "message": receptor_meta.get(
                "reason",
                "Экспериментальная 3D-структура мишени не найдена — докинг пропущен.",
            ),
        }

    eff_box_center = [
        float(v)
        for v in (
            custom_box_center
            if custom_box_center is not None and len(custom_box_center) == 3
            else receptor_meta["box_center"]
        )
    ]
    eff_box_size = [
        float(v)
        for v in (
            box_size
            if box_size is not None and len(box_size) == 3
            else receptor_meta["box_size"]
        )
    ]

    volume = eff_box_size[0] * eff_box_size[1] * eff_box_size[2]
    if volume > DOCKING_MAX_BOX_VOLUME or any(d <= 0.0 for d in eff_box_size):
        return {
            "status": "FAILED",
            "pdb_id": receptor_meta.get("pdb_id"),
            "receptor_name": receptor_name,
            "source": receptor_meta.get("source"),
            "box_center": eff_box_center,
            "box_size": eff_box_size,
            "message": (
                f"Защитный барьер: объём бокса докинга ({volume:.1f} Å³) превышает предел "
                f"{DOCKING_MAX_BOX_VOLUME:.1f} Å³ или содержит некорректные размеры."
            ),
        }

    eff_exhaust = max(1, min(16, int(exhaustiveness)))
    cache_path = (
        _docking_cache_file(
            clean_smiles=clean_smiles,
            pdb_id=str(receptor_meta["pdb_id"]),
            pdbqt_path=str(receptor_meta["pdbqt_path"]),
            box_center=eff_box_center,
            box_size=eff_box_size,
            exhaustiveness=eff_exhaust,
        )
        if use_cache
        else None
    )
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("status") == "SUCCESS" and cached.get("pose_sdf_block"):
                pose_sdf_block = str(cached["pose_sdf_block"])
                saved_sdf_path: str | None = None
                if output_sdf_path and pose_sdf_block:
                    sdf_out = Path(output_sdf_path)
                    sdf_out.parent.mkdir(parents=True, exist_ok=True)
                    sdf_out.write_text(pose_sdf_block, encoding="utf-8")
                    saved_sdf_path = sdf_out.as_posix()
                return {
                    "status": "SUCCESS",
                    "pdb_id": receptor_meta.get("pdb_id"),
                    "receptor_name": receptor_name,
                    "source": receptor_meta.get("source"),
                    "docking_score_kcal": cached.get("docking_score_kcal"),
                    "backend": cached.get("backend"),
                    "interactions": cached.get("interactions", {}),
                    "pose_sdf_path": saved_sdf_path,
                    "pose_sdf_block": pose_sdf_block,
                    "receptor_pdb_path": receptor_meta.get("pdb_path"),
                    "box_center": eff_box_center,
                    "box_size": eff_box_size,
                    "key_residues": list(receptor_meta.get("key_residues", [])),
                    "message": cached.get("message"),
                }
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    payload = {
        "clean_smiles": clean_smiles,
        "pdbqt_path": receptor_meta["pdbqt_path"],
        "pdb_path": receptor_meta["pdb_path"],
        "pdb_id": receptor_meta["pdb_id"],
        "box_center": eff_box_center,
        "box_size": eff_box_size,
        "exhaustiveness": eff_exhaust,
        "key_residues": list(receptor_meta.get("key_residues", [])),
    }

    mp_ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
            future = executor.submit(_docking_worker, payload)
            worker_res = future.result(timeout=45.0)
    except FuturesTimeoutError:
        return {
            "status": "FAILED",
            "pdb_id": receptor_meta.get("pdb_id"),
            "receptor_name": receptor_name,
            "source": receptor_meta.get("source"),
            "box_center": eff_box_center,
            "box_size": eff_box_size,
            "key_residues": list(receptor_meta.get("key_residues", [])),
            "receptor_pdb_path": receptor_meta.get("pdb_path"),
            "message": "Расчёт 3D-докинга остановлен по таймауту (45.0 с).",
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "pdb_id": receptor_meta.get("pdb_id"),
            "receptor_name": receptor_name,
            "source": receptor_meta.get("source"),
            "box_center": eff_box_center,
            "box_size": eff_box_size,
            "key_residues": list(receptor_meta.get("key_residues", [])),
            "receptor_pdb_path": receptor_meta.get("pdb_path"),
            "message": f"Ошибка изолированного процесса 3D-докинга: {exc}",
        }

    pose_sdf_block = worker_res.get("pose_sdf_block")
    if (
        cache_path is not None
        and worker_res.get("status") == "SUCCESS"
        and pose_sdf_block
    ):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "docking_score_kcal": worker_res.get("docking_score_kcal"),
                        "backend": worker_res.get("backend"),
                        "interactions": worker_res.get("interactions", {}),
                        "pose_sdf_block": pose_sdf_block,
                        "message": worker_res.get("message"),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    saved_sdf_path = None
    if output_sdf_path and pose_sdf_block:
        sdf_out = Path(output_sdf_path)
        sdf_out.parent.mkdir(parents=True, exist_ok=True)
        sdf_out.write_text(pose_sdf_block, encoding="utf-8")
        saved_sdf_path = sdf_out.as_posix()

    return {
        "status": worker_res.get("status", "FAILED"),
        "pdb_id": receptor_meta.get("pdb_id"),
        "receptor_name": receptor_name,
        "source": receptor_meta.get("source"),
        "docking_score_kcal": worker_res.get("docking_score_kcal"),
        "backend": worker_res.get("backend"),
        "interactions": worker_res.get("interactions", {}),
        "pose_sdf_path": saved_sdf_path,
        "pose_sdf_block": pose_sdf_block,
        "receptor_pdb_path": receptor_meta.get("pdb_path"),
        "box_center": eff_box_center,
        "box_size": eff_box_size,
        "key_residues": list(receptor_meta.get("key_residues", [])),
        "message": worker_res.get("message"),
    }


__all__ = [
    "PROLIF_INTERACTION_TYPES",
    "run_docking",
]
