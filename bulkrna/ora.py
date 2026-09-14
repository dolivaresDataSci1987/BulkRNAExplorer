from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
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
    FDR correction is applied across every eligible pathway in the selected
    collection; the minimum-overlap rule is only a reporting filter.
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
            "reported_terms": 0,
        }

    all_terms = pd.DataFrame(rows)
    all_terms["FDR"] = _bh_adjust(all_terms["P-value"].to_numpy())
    tested_terms = len(all_terms)

    out = all_terms.loc[all_terms["Overlap"] >= int(min_overlap)].copy()
    out = out.sort_values(["FDR", "P-value", "Fold enrichment"], ascending=[True, True, False]).reset_index(drop=True)
    return out, actual_library, {
        "query_genes": n_query,
        "background_genes": n_background,
        "tested_terms": tested_terms,
        "reported_terms": len(out),
    }


def render_ora_panel(
    history: Mapping[str, object],
    selected_comparisons: Sequence[str],
    region_name: str,
    region_genes: Sequence[str],
    annotation: pd.DataFrame | None,
    species: str,
    direction_key: str,
    padj_cut: float,
    lfc_cut: float,
) -> None:
    """Render a self-contained ORA panel for the currently selected Venn region."""
    st.divider()
    st.markdown("##### Functional enrichment of this Venn region (ORA)")
    st.caption(
        "Tests whether functional categories are over-represented in the selected genes. "
        "The background is restricted to genes that were actually tested in every selected comparison. "
        "This is over-representation analysis (ORA), not GSEA."
    )

    if annotation is None:
        st.warning("Functional enrichment needs gene-symbol annotation, which is unavailable in this session.")
        return
    if not region_genes:
        st.info("This Venn region contains no genes, so enrichment cannot be calculated.")
        return

    c1, c2, c3 = st.columns(3)
    library = c1.selectbox(
        "Functional collection",
        [
            "GO Biological Process",
            "GO Molecular Function",
            "GO Cellular Component",
            "Reactome",
            "KEGG",
        ],
        key="ora_library",
    )
    display_fdr = c2.number_input(
        "Display FDR ≤",
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.4f",
        key="ora_display_fdr",
    )
    min_overlap = c3.number_input(
        "Minimum overlapping genes",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
        key="ora_min_overlap",
    )

    signature = (
        tuple(selected_comparisons),
        str(direction_key),
        float(padj_cut),
        float(lfc_cut),
        str(region_name),
        str(library),
        int(min_overlap),
    )

    if st.button("▶ RUN FUNCTIONAL ENRICHMENT", use_container_width=True, key="run_ora_button"):
        try:
            background_ids = common_tested_background_gene_ids(history, selected_comparisons)
            with st.spinner("Running over-representation analysis…"):
                results, actual_library, stats = run_ora(
                    selected_gene_ids=list(region_genes),
                    background_gene_ids=list(background_ids),
                    annotation=annotation,
                    library=library,
                    species=species,
                    min_overlap=int(min_overlap),
                    min_pathway_size=5,
                    max_pathway_size=1000,
                )
            st.session_state["ora_last"] = {
                "signature": signature,
                "results": results,
                "library": actual_library,
                "stats": stats,
            }
            st.success(f"Functional enrichment complete using {actual_library}.")
        except Exception as exc:
            st.exception(exc)

    last = st.session_state.get("ora_last")
    if not last or last.get("signature") != signature:
        return

    results = last["results"].copy()
    stats = last["stats"]
    actual_library = last["library"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Genes analysed", stats.get("query_genes", 0))
    m2.metric("Tested background", stats.get("background_genes", 0))
    m3.metric("Terms tested", stats.get("tested_terms", 0))
    st.caption(f"Resolved library: `{actual_library}` · {stats.get('reported_terms', len(results))} terms meet the overlap filter")

    if results.empty:
        st.warning("No functional terms met the pathway-size and overlap criteria.")
        return

    show_mode = st.radio(
        "Show enrichment results",
        ["FDR-significant only", "All reported terms"],
        horizontal=True,
        key="ora_show_mode",
    )
    if show_mode == "FDR-significant only":
        shown = results.loc[results["FDR"] <= float(display_fdr)].copy()
        if shown.empty:
            st.warning(f"No terms pass FDR ≤ {display_fdr:g}. Switch to 'All reported terms' to inspect the complete table.")
    else:
        shown = results.copy()

    if not shown.empty:
        st.dataframe(shown, use_container_width=True, height=430)
        plot_df = shown.head(25).copy()
        fig = px.scatter(
            plot_df,
            x="Fold enrichment",
            y="Term",
            size="Overlap",
            hover_data=["FDR", "P-value", "Genes", "Pathway size in background"],
            title=f"Functional enrichment · {region_name}",
        )
        fig.update_layout(height=max(550, 24 * len(plot_df) + 180))
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download full enrichment results (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="venn_region_functional_enrichment.csv",
        mime="text/csv",
        key="download_ora_results",
    )
