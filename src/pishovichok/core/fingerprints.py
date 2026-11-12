"""Утилиты вычисления молекулярных фингерпринтов и метрик сходства (Слой 1)."""

from __future__ import annotations

from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

_FP_GENERATORS: dict[tuple[str, int, int], Any] = {}


def _get_fp_generator(fp_type: str, fp_radius: int, fp_nbits: int) -> Any:
    key = (fp_type, int(fp_radius), int(fp_nbits))
    gen = _FP_GENERATORS.get(key)
    if gen is not None:
        return gen
    if fp_type == "ecfp":
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(fp_radius), fpSize=int(fp_nbits)
        )
    elif fp_type == "fcfp":
        inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(fp_radius),
            fpSize=int(fp_nbits),
            atomInvariantsGenerator=inv_gen,
        )
    elif fp_type == "rdkit":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=int(fp_nbits))
    else:
        return None
    _FP_GENERATORS[key] = gen
    return gen


def compute_mol_fingerprint(
    mol: Chem.Mol,
    fp_type: str = "ecfp",
    fp_radius: int = 2,
    fp_nbits: int = 2048,
) -> Any:
    """Вычисляет битовый отпечаток молекулы в соответствии с настройками ToolConfig."""
    if fp_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    gen = _get_fp_generator(fp_type, fp_radius, fp_nbits)
    if gen is None:
        gen = _get_fp_generator("ecfp", 2, 2048)
    return gen.GetFingerprint(mol)


def compute_fp_similarity(
    fp1: Any,
    fp2: Any,
    metric: str = "tanimoto",
) -> float:
    """Вычисляет коэффициент сходства между двумя битовыми отпечатками."""
    if metric == "dice":
        return float(DataStructs.DiceSimilarity(fp1, fp2))
    if metric == "cosine":
        return float(DataStructs.CosineSimilarity(fp1, fp2))
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))
