from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from bulkrna.annotation import add_annotation_to_results
from bulkrna.enrichment import (
    load_library_gene_sets,
    make_prerank_table,
    matched_gene_set_sizes,
    run_prerank_gsea_gene_sets,
    subset_gene_sets,
)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _cached_gene_set_library(friendly_name: str, species: str):
    return load_library_gene_sets(friendly_name, species)


def _tokens_from_text(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[\s,;]+", text or "") if x.strip()]


def _annotation_maps(annotation: pd.DataFrame):
    symbol_map: dict[str, str] = {}
    ensembl_map: dict[str, str] = {}
    for gene_id, row in annotation.iterrows():
        symbol = str(row.get("gene_symbol", "") or "").strip()
        ens = str(row.get("ensembl_id", "") or "").strip()
        if symbol:
            symbol_map[symbol.upper()] = symbol
            if ens:
                ensembl_map[ens.upper()] = symbol
            ensembl_map[str(gene_id).split(".")[0].upper()] = symbol
    return symbol_map, ensembl_map


def _normalise_gene_tokens(tokens: list[str], annotation: pd.DataFrame, rank: pd.DataFrame):
    symbol_map, ensembl_map = _annotation_maps(annotation)
    universe = {str(x).upper(): str(x) for x in rank["gene_symbol"].astype(str)}
    resolved: list[str] = []
    unresolved: list[str] = []

    for token in tokens:
        raw = str(token).strip().strip('"').strip("'")
        if not raw:
            continue
        key = raw.upper()
        symbol = None
        if key in universe:
            symbol = universe[key]
        elif key in symbol_map:
            symbol = symbol_map[key]
        else:
            ens = raw.split(".")[0].upper()
            if ens in ensembl_map:
                symbol = ensembl_map[ens]
        if symbol:
            resolved.append(symbol)
        else:
            unresolved.append(raw)

    return list(dict.fromkeys(resolved)), list(dict.fromkeys(unresolved))


def _read_uploaded_gene_sets(uploaded, default_name: str) -> dict[str, list[str]]:
    if uploaded is None:
        return {}
    raw = uploaded.getvalue()
    name = uploaded.name.lower()

    if name.endswith((".csv", ".tsv", ".tabular")):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str).fillna("")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str).fillna("")
        if df.empty:
            return {}

        lower = {str(c).strip().lower(): c for c in df.columns}
        set_col = next((lower[x] for x in ["pathway", "gene_set", "geneset", "set", "process"] if x in lower), None)
        gene_col = next((lower[x] for x in ["gene", "symbol", "gene_symbol", "genes"] if x in lower), None)

        if set_col is not None and gene_col is not None:
            out: dict[str, list[str]] = {}
            for _, row in df.iterrows():
                set_name = str(row[set_col]).strip()
                gene = str(row[gene_col]).strip()
                if set_name and gene:
                    out.setdefault(set_name, []).append(gene)
            return out

        first_col = df.columns[0]
        genes = [str(x).strip() for x in df[first_col].tolist() if str(x).strip()]
        return {default_name: genes} if genes else {}

    text = raw.decode("utf-8", errors="ignore")
    genes = _tokens_from_text(text)
    return {default_name: genes} if genes else {}


def _normalise_gene_sets(raw_sets: dict[str, list[str]], annotation: pd.DataFrame, rank: pd.DataFrame):
    resolved_sets: dict[str, list[str]] = {}
    unresolved_rows: list[dict] = []
    for set_name, tokens in raw_sets.items():
        resolved, unresolved = _normalise_gene_tokens(tokens, annotation, rank)
        if resolved:
            resolved_sets[str(set_name)] = resolved
        for gene in unresolved:
            unresolved_rows.append({"gene_set": str(set_name), "unmatched_input": gene})
    return resolved_sets, pd.DataFrame(unresolved_rows)


def _gmt_bytes(gene_sets: dict[str, list[str]]) -> bytes:
    lines = []
    for name, genes in gene_sets.items():
        lines.append("\t".join([str(name), "BulkRNAExplorer"] + [str(g) for g in genes]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _plot_results(results: pd.DataFrame, title: str):
    if results is None or results.empty:
        st.warning("No gene sets passed the matching/size criteria for this analysis.")
        return

    st.dataframe(results, use_container_width=True, height=520)
    if "Term" not in results.columns or "NES" not in results.columns:
        return

    plot_df = results.copy()
    if "FDR q-val" in plot_df.columns:
        plot_df["FDR q-val"] = pd.to_numeric(plot_df["FDR q-val"], errors="coerce")
        plot_df["_sig"] = -np.log10(plot_df["FDR q-val"].clip(lower=1e-12))
        plot_df = plot_df.sort_values("FDR q-val").head(30)
        fig = px.scatter(
            plot_df,
            x="NES",
            y="Term",
            size="_sig",
            hover_data=[c for c in ["NOM p-val", "FDR q-val", "Lead_genes"] if c in plot_df.columns],
            title=title,
        )
    else:
        plot_df = plot_df.reindex(plot_df["NES"].abs().sort_values(ascending=False).head(30).index)
        fig = px.scatter(plot_df, x="NES", y="Term", title=title)
    fig.update_layout(height=max(550, 23 * len(plot_df) + 180))
    st.plotly_chart(fig, use_container_width=True)


def render_custom_gsea(species: str = "human"):
    st.subheader("Custom GSEA")
    st.caption("Test only pathways you choose, or create your own gene set(s).")

    history = st.session_state.get("analysis_history", {})
    annotation = st.session_state.get("annotation")
    if not history:
        st.info("Run at least one differential-expression comparison first.")
        return
    if annotation is None:
        st.warning("Custom GSEA needs gene-symbol annotation, which is unavailable for this session.")
        return

    names = list(history.keys())
    active = st.session_state.get("active_analysis")
    default_index = names.index(active) if active in names else len(names) - 1
    comparison = st.selectbox("Comparison", names, index=default_index, key="custom_gsea_comparison")
    bundle = history[comparison]
    raw_res = bundle.results[bundle.comparison_name]
    annotated_res = add_annotation_to_results(raw_res, annotation)
    rank = make_prerank_table(annotated_res)

    st.info(
        f"Ranking: **{comparison}** · all annotated genes ordered by the PyDESeq2 Wald statistic. "
        f"Positive NES = enrichment toward **{bundle.group_b_name}**; negative NES = enrichment toward **{bundle.group_a_name}**."
    )

    mode = st.radio(
        "Custom GSEA mode",
        ["Select specific pathways", "My custom gene sets"],
        horizontal=True,
        key="custom_gsea_mode",
    )

    if mode == "Select specific pathways":
        c1, c2 = st.columns(2)
        collection = c1.selectbox(
            "Collection",
            [
                "Hallmark",
                "GO Biological Process",
                "GO Molecular Function",
                "GO Cellular Component",
                "Reactome",
                "KEGG",
            ],
            key="selected_pathway_collection",
        )
        permutations = c2.selectbox("Permutations", [100, 250, 500], index=0, key="selected_pathway_permutations")

        try:
            with st.spinner("Loading pathway names…"):
                library_sets, actual_library = _cached_gene_set_library(collection, species)
            st.caption(f"Library: `{actual_library}` · {len(library_sets):,} pathways available")

            query = st.text_input(
                "Search pathways",
                placeholder="e.g. senesc, apoptosis, DNA repair, epithelial mesenchymal…",
                key="selected_pathway_search",
            ).strip().lower()

            all_terms = sorted(library_sets.keys())
            if query:
                words = [w for w in query.split() if w]
                filtered_terms = [t for t in all_terms if all(w in t.lower() for w in words)]
            elif len(all_terms) <= 250:
                filtered_terms = all_terms
            else:
                filtered_terms = []
                st.info("This collection is large. Type a word above to filter pathways before selecting them.")

            if query:
                st.caption(f"{len(filtered_terms):,} pathways match your search.")

            use_all_filtered = st.checkbox(
                "Use all pathways matching this search",
                value=False,
                disabled=not filtered_terms,
                key="use_all_filtered_pathways",
            )
            if use_all_filtered:
                selected_terms = filtered_terms
            else:
                selected_terms = st.multiselect(
                    "Specific pathways",
                    filtered_terms,
                    key="specific_pathway_multiselect",
                    placeholder="Click the pathways you want to include",
                )

            selected_sets = subset_gene_sets(library_sets, selected_terms)
            if selected_sets:
                preview = matched_gene_set_sizes(selected_sets, rank)
                st.dataframe(preview, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download selected pathways (GMT)",
                    _gmt_bytes(selected_sets),
                    file_name="selected_pathways.gmt",
                    mime="text/plain",
                )

            a1, a2 = st.columns(2)
            min_size = a1.number_input("Minimum matched genes per pathway", min_value=3, max_value=100, value=10, step=1, key="selected_min_size")
            max_size = a2.number_input("Maximum matched genes per pathway", min_value=20, max_value=5000, value=1000, step=10, key="selected_max_size")

            if st.button("▶ RUN SELECTED-PATHWAY GSEA", type="primary", use_container_width=True):
                if not selected_sets:
                    st.error("Select at least one pathway first.")
                else:
                    with st.spinner("Running GSEA only on the selected pathways…"):
                        results, _, label = run_prerank_gsea_gene_sets(
                            annotated_res,
                            selected_sets,
                            label=f"{actual_library} · selected pathways",
                            species=species,
                            permutation_num=int(permutations),
                            min_size=int(min_size),
                            max_size=int(max_size),
                            seed=7,
                            threads=1,
                        )
                    st.session_state["custom_gsea_last"] = {
                        "comparison": comparison,
                        "label": label,
                        "results": results,
                    }
                    st.success(f"Custom GSEA complete for {len(selected_sets)} selected pathway(s).")
        except Exception as e:
            st.exception(e)

    else:
        n1, n2 = st.columns([2, 1])
        custom_name = n1.text_input("Gene-set / process name", value="Senescence", key="custom_gene_set_name").strip() or "Custom_gene_set"
        permutations = n2.selectbox("Permutations", [100, 250, 500], index=0, key="custom_gene_permutations")

        pasted = st.text_area(
            "Paste genes",
            height=180,
            placeholder="CDKN1A\nCDKN2A\nTP53\nSERPINE1\nIL6\nCXCL8",
            key="custom_gene_text",
        )
        uploaded = st.file_uploader(
            "Or upload gene list",
            type=["txt", "csv", "tsv", "tabular"],
            key="custom_gene_file",
            help="One-column files create one gene set. A two-column file with pathway/gene or process/gene creates several gene sets.",
        )

        raw_sets: dict[str, list[str]] = {}
        pasted_tokens = _tokens_from_text(pasted)
        if pasted_tokens:
            raw_sets[custom_name] = pasted_tokens
        if uploaded is not None:
            file_sets = _read_uploaded_gene_sets(uploaded, custom_name)
            for set_name, genes in file_sets.items():
                raw_sets.setdefault(set_name, []).extend(genes)

        resolved_sets, unresolved = _normalise_gene_sets(raw_sets, annotation, rank) if raw_sets else ({}, pd.DataFrame())
        if resolved_sets:
            preview = matched_gene_set_sizes(resolved_sets, rank)
            st.dataframe(preview, use_container_width=True, hide_index=True)
            if (preview["genes_matched"] < 15).any():
                st.warning("At least one custom gene set has fewer than 15 matched genes. Results for very small gene sets can be unstable.")
            if not unresolved.empty:
                with st.expander(f"Unmatched input genes ({len(unresolved)})"):
                    st.dataframe(unresolved, use_container_width=True, hide_index=True)
            st.download_button(
                "Download resolved custom gene sets (GMT)",
                _gmt_bytes(resolved_sets),
                file_name="custom_gene_sets.gmt",
                mime="text/plain",
            )

        a1, a2 = st.columns(2)
        min_size = a1.number_input("Minimum matched genes per gene set", min_value=3, max_value=100, value=10, step=1, key="custom_min_size")
        max_size = a2.number_input("Maximum matched genes per gene set", min_value=20, max_value=5000, value=1000, step=10, key="custom_max_size")

        if st.button("▶ RUN MY CUSTOM GSEA", type="primary", use_container_width=True):
            if not resolved_sets:
                st.error("Add at least one valid custom gene set first.")
            else:
                matched = matched_gene_set_sizes(resolved_sets, rank)
                passing = matched.loc[matched["genes_matched"] >= int(min_size), "gene_set"].tolist()
                if not passing:
                    st.error("No custom gene set has enough matched genes. Reduce the minimum size or check the gene IDs.")
                else:
                    sets_to_run = {name: resolved_sets[name] for name in passing}
                    try:
                        with st.spinner("Running GSEA against your custom gene set(s)…"):
                            results, _, label = run_prerank_gsea_gene_sets(
                                annotated_res,
                                sets_to_run,
                                label="User custom gene sets",
                                species=species,
                                permutation_num=int(permutations),
                                min_size=int(min_size),
                                max_size=int(max_size),
                                seed=7,
                                threads=1,
                            )
                        st.session_state["custom_gsea_last"] = {
                            "comparison": comparison,
                            "label": label,
                            "results": results,
                        }
                        st.success(f"Custom GSEA complete for {len(sets_to_run)} custom gene set(s).")
                    except Exception as e:
                        st.exception(e)

    last = st.session_state.get("custom_gsea_last")
    if last and last.get("comparison") == comparison:
        st.divider()
        st.subheader("Custom GSEA result")
        st.caption(last.get("label", "Custom GSEA"))
        results = last.get("results", pd.DataFrame())
        _plot_results(results, "Custom GSEA pathways")
        if isinstance(results, pd.DataFrame) and not results.empty:
            st.download_button(
                "Download Custom GSEA results (CSV)",
                results.to_csv(index=False).encode("utf-8"),
                file_name=f"{comparison.replace(' ', '_')}_Custom_GSEA.csv",
                mime="text/csv",
            )
