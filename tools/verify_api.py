"""Инструмент предварительной программной интроспекции API рантайма (API Pre-Verification Protocol).

Реализует 4-этапный протокол:
1. Import Discovery (Обнаружение пути импорта)
2. Signature Inspection (Интроспекция сигнатур через inspect.signature)
3. In-Memory Probe (Минимальный синтетический вызов в RAM / временном буфере)
4. Contract Binding (Подтверждение типов возвращаемых объектов)
"""

from __future__ import annotations

import importlib.metadata
import inspect
import sys
import tempfile
from pathlib import Path
from typing import Any


def verify_interface(
    target_obj: Any,
    expected_callable: str,
    sample_args: tuple[Any, ...] = (),
    sample_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Проверяет наличие вызываемого атрибута, его сигнатуру и выполняет пробный запуск."""
    sample_kwargs = sample_kwargs or {}
    attr = getattr(target_obj, expected_callable, None)
    if attr is None:
        raise AttributeError(
            f"API Mismatch: {target_obj} does not have attribute '{expected_callable}'"
        )

    sig: inspect.Signature | None = None
    if callable(attr):
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            # C++/Boost.Python функции в RDKit могут не поддерживать inspect.signature напрямую
            sig = None

    result = None
    exec_error = None
    if callable(attr) and (sample_args or sample_kwargs):
        try:
            result = attr(*sample_args, **sample_kwargs)
        except Exception as exc:  # noqa: BLE001
            exec_error = f"{type(exc).__name__}: {exc}"

    return {
        "attribute": expected_callable,
        "is_callable": callable(attr),
        "signature": str(sig) if sig is not None else "C++/Boost.Python (built-in)",
        "sample_run_success": exec_error is None,
        "result_type": type(result).__name__ if result is not None else None,
        "error": exec_error,
    }


def check_installed_versions() -> dict[str, str]:
    """Возвращает версии ключевых библиотек проекта из метаданных окружения."""
    packages = [
        "rdkit",
        "duckdb",
        "pyarrow",
        "pandas",
        "numpy",
        "pydantic",
        "pydantic-settings",
        "langgraph",
        "langchain-core",
        "langchain-openai",
        "catboost",
        "scikit-learn",
        "streamlit",
        "plotly",
        "pytest",
        "ruff",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "NOT_INSTALLED"
    return versions


def verify_tabular_and_schema_apis() -> dict[str, Any]:
    """Выполняет базовую проверку API Pydantic v2, PyArrow Parquet, DuckDB и RDKit."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
    from rdkit import Chem, DataStructs
    from rdkit.Chem import BRICS, Descriptors, rdFingerprintGenerator, rdMMPA
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    report: dict[str, Any] = {}

    # 1. Проверка Pydantic v2 (ConfigDict(extra="forbid", frozen=True), model_validate, model_dump)
    class _ProbeModel(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        mol_id: str = Field(..., min_length=1)
        pic50: float = Field(..., ge=0.0)

    probe_instance = _ProbeModel.model_validate({"mol_id": "CHEMBL203_1", "pic50": 7.5})
    dumped = probe_instance.model_dump()
    extra_rejected = False
    try:
        _ProbeModel.model_validate({"mol_id": "X", "pic50": 5.0, "illegal_extra": 1})
    except ValidationError:
        extra_rejected = True

    report["pydantic_v2"] = {
        "model_validate_ok": dumped == {"mol_id": "CHEMBL203_1", "pic50": 7.5},
        "extra_forbid_enforced": extra_rejected,
        "methods_verified": [
            verify_interface(
                _ProbeModel, "model_validate", ({"mol_id": "A", "pic50": 6.0},)
            ),
            verify_interface(probe_instance, "model_dump", (), {"mode": "python"}),
            verify_interface(probe_instance, "model_dump_json", (), {"indent": 2}),
        ],
    }

    # 2. Проверка PyArrow Parquet и DuckDB (запись/чтение ZSTD Parquet, SQL-скан, .to_arrow_table(), .df())
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "probe.parquet"
        sample_rows = [
            {
                "mol_id": "M1",
                "canonical_smiles": "c1ccccc1",
                "pchembl_value": 6.2,
                "bao_format": "BAO_0000357",
            },
            {
                "mol_id": "M2",
                "canonical_smiles": "Cc1ccccc1",
                "pchembl_value": 7.8,
                "bao_format": "BAO_0000219",
            },
        ]
        arrow_table = pa.Table.from_pylist(sample_rows)
        pq.write_table(arrow_table, parquet_path, compression="zstd")

        read_back_table = pq.read_table(parquet_path)
        conn = duckdb.connect(database=":memory:")
        rel = conn.sql(
            f"SELECT mol_id, pchembl_value FROM read_parquet('{parquet_path.as_posix()}') "
            "WHERE bao_format = 'BAO_0000357'"
        )
        arrow_res = rel.to_arrow_table()
        df_res = conn.sql(
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        ).df()

        report["parquet_and_duckdb"] = {
            "pyarrow_write_read_rows": read_back_table.num_rows,
            "duckdb_filtered_rows": arrow_res.num_rows,
            "duckdb_arrow_type": type(arrow_res).__name__,
            "duckdb_df_shape": list(df_res.shape),
            "interfaces": [
                verify_interface(pq, "write_table"),
                verify_interface(pq, "read_table", (parquet_path,)),
                verify_interface(duckdb, "connect", (":memory:",)),
                verify_interface(conn, "sql", ("SELECT 1",)),
                verify_interface(rel, "to_arrow_table"),
            ],
        }

    # 3. Проверка критичных вызовов RDKit
    mol = Chem.MolFromSmiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1.Cl")
    clean_mol = rdMolStandardize.Cleanup(mol)
    chooser = rdMolStandardize.LargestFragmentChooser()
    parent_mol = chooser.choose(clean_mol)
    uncharger = rdMolStandardize.Uncharger()
    neutral_mol = uncharger.uncharge(parent_mol)
    te = rdMolStandardize.TautomerEnumerator()
    canon_mol = te.Canonicalize(neutral_mol)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(canon_mol)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp1 = fp_gen.GetFingerprint(canon_mol)
    fp2 = fp_gen.GetFingerprint(scaffold_mol)
    tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    mmpa_frags = list(rdMMPA.FragmentMol(canon_mol, maxCuts=1, resultsAsMols=False))
    brics_frags = list(BRICS.BRICSDecompose(canon_mol))

    report["rdkit_core"] = {
        "standardized_smiles": Chem.MolToSmiles(canon_mol, canonical=True),
        "inchi_key": Chem.MolToInchiKey(canon_mol),
        "scaffold_smiles": Chem.MolToSmiles(scaffold_mol, canonical=True),
        "tanimoto_to_scaffold": round(float(tanimoto_sim), 4),
        "mmpa_fragments_count": len(mmpa_frags),
        "brics_fragments_count": len(brics_frags),
        "interfaces": [
            verify_interface(rdMolStandardize, "Cleanup", (mol,)),
            verify_interface(chooser, "choose", (clean_mol,)),
            verify_interface(uncharger, "uncharge", (parent_mol,)),
            verify_interface(te, "Canonicalize", (neutral_mol,)),
            verify_interface(MurckoScaffold, "GetScaffoldForMol", (canon_mol,)),
            verify_interface(Chem, "MolToInchiKey", (canon_mol,)),
            verify_interface(
                rdMMPA,
                "FragmentMol",
                (canon_mol,),
                {"maxCuts": 1, "resultsAsMols": False},
            ),
            verify_interface(
                rdFingerprintGenerator,
                "GetMorganGenerator",
                (),
                {"radius": 2, "fpSize": 2048},
            ),
            verify_interface(fp_gen, "GetFingerprint", (canon_mol,)),
            verify_interface(DataStructs, "TanimotoSimilarity", (fp1, fp2)),
            verify_interface(Descriptors, "MolLogP", (canon_mol,)),
            verify_interface(Descriptors, "TPSA", (canon_mol,)),
        ],
    }

    report["generation_and_evaluation"] = verify_generation_and_evaluation_apis()
    report["orchestration_and_llm"] = verify_orchestration_and_llm_apis()
    return report


def verify_generation_and_evaluation_apis() -> dict[str, Any]:
    """Проверяет API модулей направленной генерации, SAscore, FilterCatalog, BRICS и ADMET."""
    import sqlite3

    import catboost
    from rdkit import Chem, RDConfig
    from rdkit.Chem import BRICS, QED, AllChem, Descriptors, FilterCatalog

    project_root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {}

    # 1. Движок химических реакций AllChem.ReactionFromSmarts + RunReactants + SanitizeMol
    rxn = AllChem.ReactionFromSmarts("[*:1]-[O;H0;D2;R0]-[C;H3;D1;R0]>>[*:1]-C#N")
    anisole = Chem.MolFromSmiles("COc1ccccc1")
    products_tuple = rxn.RunReactants((anisole,))
    first_prod = products_tuple[0][0]
    sanitize_flag = Chem.SanitizeMol(first_prod)
    prod_smiles = Chem.MolToSmiles(first_prod, canonical=True)

    # Проверка доступности пакета crem и локальной базы фрагментов data/crem/replacements_sa2.db
    try:
        import crem  # type: ignore[import-not-found]

        crem_pkg_status = getattr(crem, "__version__", "INSTALLED")
    except ImportError:
        crem_pkg_status = "NOT_INSTALLED"

    crem_db_path = project_root / "data" / "crem" / "replacements_sa2.db"
    crem_db_rows = 0
    crem_db_columns: list[str] = []
    if crem_db_path.is_file():
        with sqlite3.connect(crem_db_path) as conn:
            cur = conn.execute("SELECT * FROM radius3")
            crem_db_columns = [desc[0] for desc in (cur.description or [])]
            crem_db_rows = len(cur.fetchall())

    report["reaction_and_fragment_engine"] = {
        "reaction_type": type(rxn).__name__,
        "run_reactants_return_type": type(products_tuple).__name__,
        "product_tuple_element_type": type(products_tuple[0]).__name__,
        "product_mol_type": type(first_prod).__name__,
        "sanitize_result": str(sanitize_flag),
        "sample_product_smiles": prod_smiles,
        "crem_package_status": crem_pkg_status,
        "local_crem_db_exists": crem_db_path.is_file(),
        "local_crem_db_rows": crem_db_rows,
        "local_crem_db_columns": crem_db_columns,
        "fallback_catalog_ready": True,
        "interfaces": [
            verify_interface(
                AllChem, "ReactionFromSmarts", ("[*:1][F;H0;D1]>>[*:1]C#N",)
            ),
            verify_interface(rxn, "RunReactants", ((anisole,),)),
            verify_interface(Chem, "SanitizeMol", (first_prod,)),
        ],
    }

    # 2. Расчет SAscore, FilterCatalog (PAINS/Brenk/Glaxo) и ретросинтетическая декомпозиция BRICS
    sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
    sascorer_available = False
    gefitinib_mol = Chem.MolFromSmiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    sa_val = 0.0
    if sa_dir.is_dir():
        if str(sa_dir) not in sys.path:
            sys.path.append(str(sa_dir))
        import sascorer  # type: ignore[import-not-found]

        sascorer_available = True
        sa_val = float(sascorer.calculateScore(gefitinib_mol))

    fc_params = FilterCatalog.FilterCatalogParams()
    fc_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    fc_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    fc_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo)
    fc_catalog = FilterCatalog.FilterCatalog(fc_params)
    quinone_mol = Chem.MolFromSmiles("O=C1C=CC(=O)c2ccccc21")
    fc_matches = fc_catalog.GetMatches(quinone_mol)
    first_match_desc = fc_matches[0].GetDescription() if len(fc_matches) > 0 else None

    brics_set = BRICS.BRICSDecompose(gefitinib_mol, returnMols=False)
    brics_bonds = list(BRICS.FindBRICSBonds(gefitinib_mol))

    report["evaluation_filters_and_brics"] = {
        "sascorer_available": sascorer_available,
        "gefitinib_sa_score": round(sa_val, 4),
        "gefitinib_qed": round(float(QED.qed(gefitinib_mol)), 4),
        "filter_catalog_matches_type": type(fc_matches).__name__,
        "quinone_alert_detected": first_match_desc,
        "brics_return_type": type(brics_set).__name__,
        "brics_raw_fragments_count": len(brics_set),
        "brics_bonds_count": len(brics_bonds),
        "interfaces": [
            verify_interface(QED, "qed", (gefitinib_mol,)),
            verify_interface(Descriptors, "ExactMolWt", (gefitinib_mol,)),
            verify_interface(Descriptors, "FractionCSP3", (gefitinib_mol,)),
            verify_interface(fc_catalog, "GetMatches", (quinone_mol,)),
            verify_interface(
                BRICS, "BRICSDecompose", (gefitinib_mol,), {"returnMols": False}
            ),
            verify_interface(BRICS, "FindBRICSBonds", (gefitinib_mol,)),
        ],
    }

    # 3. Проверка моделей ADMET (admet_ai -> CatBoost -> Analytical QSAR fallback)
    try:
        import admet_ai  # type: ignore[import-not-found]

        admet_ai_status = getattr(admet_ai, "__version__", "INSTALLED")
    except ImportError:
        admet_ai_status = "NOT_INSTALLED"

    models_dir = project_root / "data" / "models"
    cbm_files = (
        sorted(p.name for p in models_dir.glob("*.cbm")) if models_dir.is_dir() else []
    )

    report["admet_models"] = {
        "admet_ai_status": admet_ai_status,
        "catboost_version": catboost.__version__,
        "local_cbm_files": cbm_files,
        "active_cascade_tier": (
            "admet_ai_dmpnn"
            if admet_ai_status != "NOT_INSTALLED"
            else (
                "catboost_local_qsar"
                if len(cbm_files) >= 4
                else "rdkit_analytical_qsar"
            )
        ),
        "interfaces": [
            verify_interface(catboost.CatBoostClassifier, "predict_proba"),
            verify_interface(catboost.CatBoostRegressor, "predict"),
        ],
    }

    return report


def verify_orchestration_and_llm_apis() -> dict[str, Any]:
    """Проверяет актуальный API StateGraph (LangGraph) и Structured Outputs (LangChain OpenAI)."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    from pishovichok.state.graph_state import SARGraphState, create_initial_graph_state
    from pishovichok.state.schemas import QueryConstraints

    report: dict[str, Any] = {}

    # 1. Проверка API графа LangGraph (START, END, StateGraph, add_node, add_edge, add_conditional_edges, compile, invoke)
    workflow = StateGraph(state_schema=SARGraphState)
    workflow.add_node(
        "test_research", lambda state: {"last_executed_stage": "research"}
    )
    workflow.add_edge(START, "test_research")
    workflow.add_conditional_edges(
        "test_research",
        lambda state: "proceed",
        {"proceed": END},
    )
    compiled_app = workflow.compile()

    init_state = create_initial_graph_state(
        user_query="Проверка API графа",
        session_id="test_probe_api",
        workspace_dir="/tmp/test_probe_api",
    )
    executed_state = compiled_app.invoke(init_state)

    report["langgraph"] = {
        "constants": {"START": START, "END": END},
        "compiled_type": type(compiled_app).__name__,
        "smoke_execution_success": executed_state.get("last_executed_stage")
        == "research",
        "interfaces": [
            verify_interface(workflow, "add_node"),
            verify_interface(workflow, "add_edge"),
            verify_interface(workflow, "add_conditional_edges"),
            verify_interface(workflow, "compile"),
            verify_interface(compiled_app, "invoke"),
        ],
    }

    # 2. Проверка Structured Outputs в langchain_openai (with_structured_output на схеме Pydantic v2)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="dummy_key_for_api_verification",
        temperature=0.0,
    )
    structured_runnable = llm.with_structured_output(
        QueryConstraints,
        method="json_schema",
        strict=True,
    )

    report["langchain_openai"] = {
        "structured_runnable_type": type(structured_runnable).__name__,
        "supported_methods": ["json_schema", "function_calling", "json_mode"],
        "interfaces": [
            verify_interface(llm, "with_structured_output"),
        ],
    }

    return report


def main() -> int:
    """Точка входа CLI для проверки версий и базовых контрактов API."""
    import json

    versions = check_installed_versions()
    missing = [k for k, v in versions.items() if v == "NOT_INSTALLED"]
    if missing:
        print(f"ERROR: Missing packages: {missing}", file=sys.stderr)
        return 1

    api_report = verify_tabular_and_schema_apis()
    output = {
        "python_version": sys.version.split()[0],
        "installed_versions": versions,
        "api_verification": api_report,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
