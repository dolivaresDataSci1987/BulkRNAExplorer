from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from bulkrna.enrichment import load_library_gene_sets


@lru_cache(maxsize=12)
def _cached_library(friendly_name: str, species: str):
    """Cache Enrichr gene-set libraries within the running Streamlit process."""
    return load_library_gene_sets(friendly_name, species)


def _bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction without adding another dependency."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def gene_ids_to_symbols(gene_ids: Sequence[str], annotation: pd.DataFrame | None) -> set[str]:
    """Map DE-result gene IDs to unique gene symbols using the session annotation."""
    if annotation is None or annotation.empty or "gene_symbol" not in annotation.columns:
        return set()
    ids = [str(x) for x in gene_ids]
    symbols = annotation.reindex(ids)["gene_symbol"].fillna("").astype(str).str.strip()
    return {s.upper() for s in symbols if s}


def common_tested_background_gene_ids(
    history: Mapping[str, object],
    selected_comparisons: Sequence[str],
) -> set[str]:
    """Genes that were actually tested in every selected DE comparison."""
    tested_sets: list[set[str]] = []
    for name in selected_comparisons:
        bundle = history[name]
        result = bundle.results[bundle.comparison_name]
        tested_sets.append({str(x) for x in result.index})
    return set.intersection(*tested_sets) if tested_sets else set()


def run_ora(
    selected_gene_ids: Sequence[str],
    background_gene_ids: Sequence[str],
    annotation: pd.DataFrame,
    library: str,
    species: str = "human",
    min_overlap: int = 2,
    min_pathway_size: int = 5,
    max_pathway_size: int = 1000,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    """Over-representation analysis using one-sided Fisher exact tests.

    The statistical universe is the set of genes that were actually tested in
    all selected differential-expression comparisons, after mapping to symbols.
    """
    query = gene_ids_to_symbols(selected_gene_ids, annotation)
    background = gene_ids_to_symbols(background_gene_ids, annotation)
    query &= background

    if len(query) < 2:
        raise ValueError("Too few annotated genes from this Venn region are present in the tested background.")
    if len(background) < 20:
        raise ValueError("Too few annotated genes are available in the tested background for enrichment analysis.")

    gene_sets, actual_library = _cached_library(library, species)
    rows: list[dict] = []
    n_query = len(query)
    n_background = len(background)

    for term, genes in gene_sets.items():
        pathway = {str(g).strip().upper() for g in genes if str(g).strip()}
        pathway &= background
        pathway_size = len(pathway)
        if pathway_size < int(min_pathway_size) or pathway_size > int(max_pathway_size):
            continue

        overlap = query & pathway
        a = len(overlap)
        if a < int(min_overlap):
            continue

        b = n_query - a
        c = pathway_size - a
        d = n_background - a - b - c
        if d < 0:
            continue

        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        expected_fraction = pathway_size / n_background
        observed_fraction = a / n_query
        fold_enrichment = observed_fraction / expected_fraction if expected_fraction > 0 else np.nan

        rows.append(
            {
                "Term": str(term),
                "Overlap": a,
                "Query size": n_query,
                "Pathway size in background": pathway_size,
                "Background size": n_background,
                "Fold enrichment": float(fold_enrichment),
                "Odds ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.inf,
                "P-value": float(pvalue),
                "Genes": ";".join(sorted(overlap)),
            }
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "Term", "Overlap", "Query size", "Pathway size in background",
                "Background size", "Fold enrichment", "Odds ratio", "P-value", "FDR", "Genes",
            ]
        )
        return empty, actual_library, {
            "query_genes": n_query,
            "background_genes": n_background,
            "tested_terms": 0,
        }

    out = pd.DataFrame(rows)
    out["FDR"] = _bh_adjust(out["P-value"].to_numpy())
    out = out.sort_values(["FDR", "P-value", "Fold enrichment"], ascending=[True, True, False]).reset_index(drop=True)
    return out, actual_library, {
        "query_genes": n_query,
        "background_genes": n_background,
        "tested_terms": len(out),
    }
