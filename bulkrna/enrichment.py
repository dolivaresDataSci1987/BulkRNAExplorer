from __future__ import annotations

import re
from typing import Mapping, Sequence

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


def load_library_gene_sets(friendly_name: str, species: str = "human") -> tuple[dict[str, list[str]], str]:
    """Download one Enrichr collection and return {term: genes}, plus the resolved library name."""
    actual_library = resolve_library(friendly_name, species)
    organism = "Human" if species.lower().startswith("human") else "Mouse"
    gene_sets = gp.get_library(name=actual_library, organism=organism)
    cleaned: dict[str, list[str]] = {}
    for term, genes in gene_sets.items():
        vals = [str(g).strip() for g in genes if str(g).strip()]
        if vals:
            cleaned[str(term)] = list(dict.fromkeys(vals))
    if not cleaned:
        raise ValueError(f"The library '{actual_library}' returned no gene sets.")
    return cleaned, actual_library


def subset_gene_sets(
    gene_sets: Mapping[str, Sequence[str]],
    selected_terms: Sequence[str],
) -> dict[str, list[str]]:
    selected = []
    seen = set()
    for term in selected_terms:
        t = str(term)
        if t in gene_sets and t not in seen:
            selected.append(t)
            seen.add(t)
    return {term: list(gene_sets[term]) for term in selected}


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

    # One ranking value per symbol. If several Ensembl IDs map to the same symbol,
    # keep the one with the strongest absolute Wald statistic.
    d["_abs"] = d["stat"].abs()
    d = d.sort_values("_abs", ascending=False).drop_duplicates("gene_symbol", keep="first")
    d = d.sort_values("stat", ascending=False)
    return d[["gene_symbol", "stat"]].reset_index(drop=True)


def _finish_prerank(pre, rank: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    # Some GSEApy runs legitimately return no table after gene-set filtering.
    # Check for None before attempting .copy(), otherwise the UI receives an
    # avoidable AttributeError.
    raw_results = getattr(pre, "res2d", None)
    if raw_results is None:
        return pd.DataFrame(), rank, label
    results = raw_results.copy()
    if results.empty:
        return pd.DataFrame(), rank, label

    for col in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    if "FDR q-val" in results.columns and "NES" in results.columns:
        results = results.sort_values(["FDR q-val", "NES"], ascending=[True, False])
    elif "FDR q-val" in results.columns:
        results = results.sort_values("FDR q-val", ascending=True)
    elif "NES" in results.columns:
        results = results.reindex(results["NES"].abs().sort_values(ascending=False).index)
    return results, rank, label


def _run_prerank(
    rank: pd.DataFrame,
    gene_sets,
    species: str,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
):
    """Memory-conscious GSEApy prerank runner for small cloud instances.

    GSEApy 1.3.x supports the multilevel/fgsea-style estimator. It avoids the
    large classic permutation tensors that can terminate a Streamlit Community
    Cloud process when a collection contains thousands of pathways. One thread
    also avoids multiplying peak memory usage.
    """
    return gp.prerank(
        rnk=rank,
        gene_sets=gene_sets,
        organism=species,
        outdir=None,
        permutation_num=max(50, int(permutation_num)),
        min_size=int(min_size),
        max_size=int(max_size),
        threads=1,
        seed=int(seed),
        no_plot=True,
        verbose=False,
        method="multilevel",
    )


def run_prerank_gsea(
    res_annotated: pd.DataFrame,
    library: str,
    species: str = "human",
    permutation_num: int = 250,
    min_size: int = 15,
    max_size: int = 500,
    seed: int = 7,
    threads: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    actual_library = resolve_library(library, species)
    rank = make_prerank_table(res_annotated)
    pre = _run_prerank(
        rank=rank,
        gene_sets=actual_library,
        species=species,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
    )
    return _finish_prerank(pre, rank, actual_library)


def run_prerank_gsea_gene_sets(
    res_annotated: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    label: str = "Custom gene sets",
    species: str = "human",
    permutation_num: int = 250,
    min_size: int = 10,
    max_size: int = 1000,
    seed: int = 7,
    threads: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run pre-ranked GSEA against an explicit dictionary of selected/custom gene sets."""
    cleaned: dict[str, list[str]] = {}
    for term, genes in gene_sets.items():
        name = str(term).strip()
        vals = [str(g).strip() for g in genes if str(g).strip()]
        vals = list(dict.fromkeys(vals))
        if name and vals:
            cleaned[name] = vals
    if not cleaned:
        raise ValueError("No valid custom gene sets were supplied.")

    rank = make_prerank_table(res_annotated)
    pre = _run_prerank(
        rank=rank,
        gene_sets=cleaned,
        species=species,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
    )
    return _finish_prerank(pre, rank, label)


def matched_gene_set_sizes(
    gene_sets: Mapping[str, Sequence[str]],
    rank: pd.DataFrame,
) -> pd.DataFrame:
    """Report supplied and matched genes for each gene set against the ranked gene universe."""
    universe = set(rank["gene_symbol"].astype(str).str.upper())
    rows = []
    for term, genes in gene_sets.items():
        supplied = list(dict.fromkeys(str(g).strip() for g in genes if str(g).strip()))
        matched = [g for g in supplied if g.upper() in universe]
        rows.append({
            "gene_set": str(term),
            "genes_supplied": len(supplied),
            "genes_matched": len(matched),
        })
    if not rows:
        return pd.DataFrame(columns=["gene_set", "genes_supplied", "genes_matched"])
    return pd.DataFrame(rows).sort_values(["genes_matched", "gene_set"], ascending=[False, True])
