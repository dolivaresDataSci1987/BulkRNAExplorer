from __future__ import annotations

import re

import pandas as pd
import gseapy as gp


FRIENDLY_LIBRARY_PATTERNS = {
    "Hallmark": [r"^MSigDB_Hallmark_", r"Hallmark"],
    "GO Biological Process": [r"^GO_Biological_Process_"],
    "GO Molecular Function": [r"^GO_Molecular_Function_"],
    "GO Cellular Component": [r"^GO_Cellular_Component_"],
    "Reactome": [r"^Reactome_"],
    "KEGG": [r"^KEGG_.*Human$", r"^KEGG_"],
}


def available_enrichr_libraries(species: str = "human") -> list[str]:
    organism = "Human" if species.lower().startswith("human") else "Mouse"
    return list(gp.get_library_name(organism=organism))


def resolve_library(friendly_name: str, species: str = "human") -> str:
    libraries = available_enrichr_libraries(species)
    if friendly_name in libraries:
        return friendly_name

    patterns = FRIENDLY_LIBRARY_PATTERNS.get(friendly_name, [re.escape(friendly_name)])
    matches: list[str] = []
    for pattern in patterns:
        matches = [x for x in libraries if re.search(pattern, x, flags=re.I)]
        if matches:
            break
    if not matches:
        raise ValueError(f"No Enrichr library matching '{friendly_name}' was found for {species}.")

    def year_key(name: str):
        years = re.findall(r"(20\d{2})", name)
        return int(years[-1]) if years else 0

    matches.sort(key=lambda x: (year_key(x), x), reverse=True)
    return matches[0]


def make_prerank_table(res_annotated: pd.DataFrame) -> pd.DataFrame:
    if "stat" not in res_annotated.columns:
        raise ValueError("The differential-expression result has no Wald statistic column.")
    if "gene_symbol" not in res_annotated.columns:
        raise ValueError("Gene symbols are required for GSEA.")

    d = res_annotated[["gene_symbol", "stat"]].copy()
    d["gene_symbol"] = d["gene_symbol"].fillna("").astype(str).str.strip()
    d["stat"] = pd.to_numeric(d["stat"], errors="coerce")
    d = d[(d["gene_symbol"] != "") & d["stat"].notna()]
    if d.empty:
        raise ValueError("No annotated genes with valid ranking statistics are available.")

    d["_abs"] = d["stat"].abs()
    d = d.sort_values("_abs", ascending=False).drop_duplicates("gene_symbol", keep="first")
    d = d.sort_values("stat", ascending=False)
    return d[["gene_symbol", "stat"]].reset_index(drop=True)


def run_prerank_gsea(
    res_annotated: pd.DataFrame,
    library: str,
    species: str = "human",
    permutation_num: int = 500,
    min_size: int = 15,
    max_size: int = 500,
    seed: int = 7,
    threads: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    actual_library = resolve_library(library, species)
    rank = make_prerank_table(res_annotated)

    pre = gp.prerank(
        rnk=rank,
        gene_sets=actual_library,
        organism=species,
        outdir=None,
        permutation_num=int(permutation_num),
        min_size=int(min_size),
        max_size=int(max_size),
        threads=int(threads),
        seed=int(seed),
        no_plot=True,
        verbose=False,
    )
    results = pre.res2d.copy()
    if results is None or results.empty:
        return pd.DataFrame(), rank, actual_library

    for col in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    if "FDR q-val" in results.columns:
        results = results.sort_values(["FDR q-val", "NES"], ascending=[True, False])
    elif "NES" in results.columns:
        results = results.reindex(results["NES"].abs().sort_values(ascending=False).index)
    return results, rank, actual_library
