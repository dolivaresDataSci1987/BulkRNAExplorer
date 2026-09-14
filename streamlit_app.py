from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from bulkrna.io import merge_featurecounts
from bulkrna.analysis import (
    run_deseq_comparison,
    pca_table,
    classify_degs,
    top_deg_gene_ids,
    common_deg_sets,
)
from bulkrna.annotation import (
    annotate_ensembl_ids,
    add_annotation_to_results,
    display_labels,
    genes_from_symbols,
)
from bulkrna.enrichment import run_prerank_gsea
from bulkrna.plots import pca_figure, volcano_figure, heatmap_figure


st.set_page_config(page_title="BulkRNA Explorer", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.4rem;}
div[data-testid="stMetric"] {background: #f7f8fb; border: 1px solid #e8eaf0; padding: 10px 14px; border-radius: 12px;}
.small-note {color:#667085;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧬 BulkRNA Explorer")
st.caption("Bulk RNA-seq from raw featureCounts · select samples → compare → explore → download")


for key, default in {
    "counts": None,
    "metadata": None,
    "annotation": None,
    "upload_signature": None,
    "analysis_history": {},
    "gsea_history": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_annotation(gene_ids: tuple[str, ...], species: str) -> pd.DataFrame:
    return annotate_ensembl_ids(gene_ids, species=species)


def to_csv_bytes(df: pd.DataFrame, index=True):
    return df.to_csv(index=index).encode("utf-8")


def reset_analyses():
    st.session_state.analysis_history = {}
    st.session_state.gsea_history = {}


def current_analysis():
    history = st.session_state.analysis_history
    if not history:
        return None, None
    names = list(history.keys())
    if "active_analysis" not in st.session_state or st.session_state.active_analysis not in history:
        st.session_state.active_analysis = names[-1]
    return st.session_state.active_analysis, history[st.session_state.active_analysis]


def infer_group_defaults(md: pd.DataFrame):
    included = md.loc[md["include"].astype(bool)].copy()
    conditions = list(pd.unique(included["condition"].astype(str)))
    if not conditions:
        return [], [], "Group_A", "Group_B"
    a_cond = "Sensitive" if "Sensitive" in conditions else conditions[0]
    b_candidates = [c for c in conditions if c != a_cond]
    b_cond = b_candidates[0] if b_candidates else a_cond
    a = included.loc[included["condition"].astype(str) == a_cond, "sample"].astype(str).tolist()
    b = included.loc[included["condition"].astype(str) == b_cond, "sample"].astype(str).tolist()
    return a, b, a_cond, b_cond


with st.sidebar:
    st.header("Project")
    organism_label = st.selectbox("Organism", ["Human", "Mouse"], index=0)
    organism = organism_label.lower()
    st.write("**1. Upload counts**")
    st.write("**2. Check samples**")
    st.write("**3. Select two sample groups**")
    st.write("**4. Run DE + figures**")
    st.write("**5. GSEA / gene panels**")
    st.divider()
    st.caption("Uploaded count files are processed for the current session. No database is created by this app.")
    if st.button("Reset project", use_container_width=True):
        for k in ["counts", "metadata", "annotation", "upload_signature"]:
            st.session_state[k] = None
        reset_analyses()
        st.rerun()


upload_tab, setup_tab, analysis_tab, panels_tab, gsea_tab, export_tab = st.tabs([
    "1 · Upload",
    "2 · Samples",
    "3 · Differential expression",
    "4 · Gene heatmaps",
    "5 · GSEA",
    "6 · Export",
])


with upload_tab:
    st.subheader("Upload featureCounts files")
    st.write("Select all Galaxy/featureCounts `.tabular`, `.txt`, `.tsv` or `.csv` files for the experiment.")
    uploaded = st.file_uploader(
        "featureCounts files",
        type=["tabular", "txt", "tsv", "csv"],
        accept_multiple_files=True,
        help="One count profile per biological sample/library.",
    )

    if uploaded:
        signature = (organism, tuple((f.name, f.size) for f in uploaded))
        if signature != st.session_state.upload_signature:
            try:
                counts, md, messages = merge_featurecounts(uploaded)
                st.session_state.counts = counts
                st.session_state.metadata = md
                st.session_state.upload_signature = signature
                reset_analyses()

                with st.spinner("Mapping Ensembl IDs to gene symbols…"):
                    try:
                        st.session_state.annotation = cached_annotation(tuple(counts.index.astype(str)), organism)
                    except Exception as annotation_error:
                        st.session_state.annotation = None
                        st.warning(
                            "Counts loaded, but gene-symbol annotation could not be retrieved right now. "
                            f"Differential expression can still run. Annotation error: {annotation_error}"
                        )
                st.success(f"Loaded {counts.shape[1]} samples and {counts.shape[0]:,} genes.")
            except Exception as e:
                st.error(f"Could not import the files: {e}")

        if st.session_state.counts is not None:
            c = st.session_state.counts
            ann = st.session_state.annotation
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Samples", c.shape[1])
            m2.metric("Genes", f"{c.shape[0]:,}")
            m3.metric("Raw counts", "Valid integers")
            if ann is not None:
                mapped = int((ann["gene_symbol"].fillna("").astype(str).str.len() > 0).sum())
                m4.metric("Gene symbols mapped", f"{mapped:,}")
            else:
                m4.metric("Gene symbols mapped", "Unavailable")

            preview = c.head(12).copy()
            if ann is not None:
                preview.insert(0, "Gene name", ann.reindex(preview.index)["gene_name"].fillna(""))
                preview.insert(0, "Symbol", ann.reindex(preview.index)["gene_symbol"].fillna(""))
                preview.insert(0, "Ensembl", ann.reindex(preview.index)["ensembl_id"].fillna(""))
            st.dataframe(preview, use_container_width=True)
    else:
        st.info("Upload featureCounts files to begin.")


with setup_tab:
    if st.session_state.counts is None:
        st.info("Upload featureCounts files first.")
    else:
        st.subheader("Check sample information")
        st.write("Correct sample names, conditions or batch labels if needed. These labels help you identify samples later.")
        edited = st.data_editor(
            st.session_state.metadata,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "include": st.column_config.CheckboxColumn("Use", default=True),
                "sample": st.column_config.TextColumn("Sample"),
                "condition": st.column_config.TextColumn("Condition"),
                "batch": st.column_config.TextColumn("Batch / experiment"),
            },
            key="metadata_editor",
        )

        if not edited.equals(st.session_state.metadata):
            old_names = list(st.session_state.metadata["sample"].astype(str))
            new_names = list(edited["sample"].astype(str))
            if len(set(new_names)) != len(new_names):
                st.error("Sample names must be unique.")
            else:
                rename = dict(zip(old_names, new_names))
                st.session_state.counts = st.session_state.counts.rename(columns=rename)
                st.session_state.metadata = edited.copy()
                reset_analyses()

        md = st.session_state.metadata
        included = md.loc[md["include"].astype(bool)]
        if not included.empty:
            st.write("Samples by condition")
            st.dataframe(
                included.groupby("condition").size().rename("n samples").to_frame(),
                use_container_width=True,
            )


with analysis_tab:
    if st.session_state.counts is None or st.session_state.metadata is None:
        st.info("Upload data and check the samples first.")
    else:
        md = st.session_state.metadata.copy()
        included = md.loc[md["include"].astype(bool)].copy()
        sample_options = included["sample"].astype(str).tolist()
        sample_info = included.copy()
        sample_info.index = sample_info["sample"].astype(str)

        if len(sample_options) < 4:
            st.warning("At least four included samples are recommended so that each comparison group can contain biological replicates.")

        default_a, default_b, default_a_name, default_b_name = infer_group_defaults(md)

        def sample_label(sample):
            row = sample_info.loc[sample]
            return f"{sample}  ·  {row.get('condition', '')}  ·  {row.get('batch', '')}"

        st.subheader("Choose exactly which samples to compare")
        st.caption("Group B vs Group A: positive log2FC means higher expression in Group B.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Group A · reference/control")
            group_a = st.multiselect(
                "Samples in Group A",
                sample_options,
                default=[x for x in default_a if x in sample_options],
                format_func=sample_label,
                key="group_a_samples",
            )
            group_a_name = st.text_input("Group A name", value=default_a_name, key="group_a_name")
        with c2:
            st.markdown("#### Group B · tested")
            group_b = st.multiselect(
                "Samples in Group B",
                sample_options,
                default=[x for x in default_b if x in sample_options],
                format_func=sample_label,
                key="group_b_samples",
            )
            group_b_name = st.text_input("Group B name", value=default_b_name, key="group_b_name")

        overlap = sorted(set(group_a).intersection(group_b))
        if overlap:
            st.error("These samples are in both groups: " + ", ".join(overlap))

        with st.expander("Advanced analysis settings"):
            a1, a2, a3 = st.columns(3)
            min_count = a1.number_input("Minimum count", min_value=0, value=10, step=1)
            selected_n = max(1, len(set(group_a + group_b)))
            min_samples = a2.number_input(
                "Minimum samples with that count",
                min_value=1,
                max_value=selected_n,
                value=min(2, selected_n),
                step=1,
            )
            use_batch = a3.checkbox(
                "Correct for batch",
                value=False,
                help="Use only when Batch/experiment is a genuine nuisance factor and is not confounded with the comparison groups.",
            )

        if st.button("▶ RUN DIFFERENTIAL EXPRESSION", type="primary", use_container_width=True):
            if overlap:
                st.error("Remove overlapping samples before running the analysis.")
            else:
                try:
                    with st.spinner("Running PyDESeq2, normalization, VST, PCA and differential expression…"):
                        bundle = run_deseq_comparison(
                            st.session_state.counts,
                            st.session_state.metadata,
                            group_a_samples=group_a,
                            group_b_samples=group_b,
                            group_a_name=group_a_name,
                            group_b_name=group_b_name,
                            use_batch=use_batch,
                            min_count=int(min_count),
                            min_samples=int(min_samples),
                            alpha=0.05,
                            n_cpus=2,
                        )
                        st.session_state.analysis_history[bundle.comparison_name] = bundle
                        st.session_state.active_analysis = bundle.comparison_name
                    st.success(f"Analysis complete: {bundle.comparison_name}")
                except Exception as e:
                    st.exception(e)

        history = st.session_state.analysis_history
        if history:
            st.divider()
            names = list(history.keys())
            active = st.selectbox(
                "View saved comparison",
                names,
                index=names.index(st.session_state.get("active_analysis", names[-1]))
                if st.session_state.get("active_analysis", names[-1]) in names
                else len(names) - 1,
                key="analysis_selector",
            )
            st.session_state.active_analysis = active
            bundle = history[active]
            raw_res = bundle.results[bundle.comparison_name]
            res = add_annotation_to_results(raw_res, st.session_state.annotation)

            pca_sub, de_sub, heatmap_sub, common_sub = st.tabs([
                "PCA", "Volcano + DEG table", "Top-DEG heatmap", "Common DEGs"
            ])

            with pca_sub:
                pca_df, explained = pca_table(bundle.vst_counts, bundle.metadata, top_n=5000)
                color_options = [c for c in ["comparison_group", "condition", "batch"] if c in pca_df.columns]
                color_by = st.selectbox("Colour by", color_options, key=f"pca_color_{active}")
                st.plotly_chart(pca_figure(pca_df, explained, color_by), use_container_width=True)
                st.dataframe(pca_df, use_container_width=True)

            with de_sub:
                a, b = st.columns(2)
                padj_cut = a.number_input(
                    "Adjusted p-value ≤",
                    min_value=0.000001,
                    max_value=1.0,
                    value=0.05,
                    format="%.4f",
                    key=f"de_padj_{active}",
                )
                lfc_cut = b.number_input(
                    "|log2FC| ≥",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"de_lfc_{active}",
                )

                status = classify_degs(raw_res, padj_cut, lfc_cut)
                rr = res.copy()
                rr.insert(0, "status", status)
                n_up = int((status == "UP").sum())
                n_down = int((status == "DOWN").sum())

                x1, x2, x3 = st.columns(3)
                x1.metric("UP in Group B", n_up)
                x2.metric("DOWN in Group B", n_down)
                x3.metric("Significant", n_up + n_down)

                st.plotly_chart(volcano_figure(res, padj_cut, lfc_cut), use_container_width=True)

                show = st.radio(
                    "Show",
                    ["Significant only", "All genes", "UP only", "DOWN only"],
                    horizontal=True,
                    key=f"show_{active}",
                )
                if show == "Significant only":
                    table = rr[rr["status"].isin(["UP", "DOWN"])]
                elif show == "UP only":
                    table = rr[rr["status"] == "UP"]
                elif show == "DOWN only":
                    table = rr[rr["status"] == "DOWN"]
                else:
                    table = rr

                preferred_cols = [
                    "status", "gene_symbol", "gene_name", "ensembl_id",
                    "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj",
                ]
                table = table[[c for c in preferred_cols if c in table.columns]]
                st.dataframe(table, use_container_width=True, height=520)
                st.download_button(
                    "Download this DEG table (CSV)",
                    to_csv_bytes(table),
                    file_name=f"{active.replace(' ', '_')}_DEGs.csv",
                    mime="text/csv",
                )

            with heatmap_sub:
                st.write("Create a heatmap from the top statistically significant genes in this comparison.")
                h1, h2, h3 = st.columns(3)
                top_n = h1.number_input(
                    "Number of top DEGs",
                    min_value=1,
                    max_value=1000,
                    value=20,
                    step=1,
                    key=f"top_n_{active}",
                )
                hm_padj = h2.number_input(
                    "Heatmap padj ≤",
                    min_value=0.000001,
                    max_value=1.0,
                    value=0.05,
                    format="%.4f",
                    key=f"hm_padj_{active}",
                )
                hm_lfc = h3.number_input(
                    "Heatmap |log2FC| ≥",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    key=f"hm_lfc_{active}",
                )
                hm_z = st.checkbox("Z-score each gene", value=True, key=f"hm_z_{active}")

                genes = top_deg_gene_ids(raw_res, int(top_n), hm_padj, hm_lfc)
                if not genes:
                    st.warning("No genes pass the selected thresholds.")
                else:
                    mat = bundle.vst_counts.loc[genes, bundle.group_a_samples + bundle.group_b_samples]
                    labels = display_labels(genes, st.session_state.annotation)
                    st.caption(f"Showing {len(genes)} genes, ranked by adjusted p-value and then |log2FC|.")
                    st.plotly_chart(
                        heatmap_figure(
                            mat,
                            bundle.metadata,
                            zscore=hm_z,
                            title=f"Top {len(genes)} DEGs · {active}",
                            gene_labels=labels,
                        ),
                        use_container_width=True,
                    )
                    hm_export = mat.copy()
                    hm_export.insert(0, "gene_symbol", labels)
                    st.download_button(
                        "Download heatmap expression (CSV)",
                        to_csv_bytes(hm_export),
                        file_name=f"{active.replace(' ', '_')}_top{len(genes)}_heatmap.csv",
                    )

            with common_sub:
                if len(history) < 2:
                    st.info("Run and save at least two comparisons to calculate common UP/DOWN genes.")
                else:
                    selected_comparisons = st.multiselect(
                        "Comparisons to intersect",
                        list(history.keys()),
                        default=list(history.keys()),
                        key="common_comparisons",
                    )
                    cpadj, clfc = st.columns(2)
                    cp = cpadj.number_input(
                        "Common DEG padj ≤",
                        min_value=0.000001,
                        max_value=1.0,
                        value=0.05,
                        format="%.4f",
                        key="common_padj",
                    )
                    cl = clfc.number_input(
                        "Common DEG |log2FC| ≥",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        key="common_lfc",
                    )
                    if len(selected_comparisons) >= 2:
                        result_dict = {
                            name: history[name].results[history[name].comparison_name]
                            for name in selected_comparisons
                        }
                        _, _, common_up, common_down = common_deg_sets(result_dict, cp, cl)
                        c_up, c_down = st.columns(2)
                        with c_up:
                            st.metric("Common UP", len(common_up))
                            up_ann = add_annotation_to_results(
                                pd.DataFrame(index=sorted(common_up)),
                                st.session_state.annotation,
                            )
                            st.dataframe(
                                up_ann[[c for c in ["gene_symbol", "gene_name", "ensembl_id"] if c in up_ann.columns]],
                                use_container_width=True,
                                height=300,
                            )
                        with c_down:
                            st.metric("Common DOWN", len(common_down))
                            down_ann = add_annotation_to_results(
                                pd.DataFrame(index=sorted(common_down)),
                                st.session_state.annotation,
                            )
                            st.dataframe(
                                down_ann[[c for c in ["gene_symbol", "gene_name", "ensembl_id"] if c in down_ann.columns]],
                                use_container_width=True,
                                height=300,
                            )


with panels_tab:
    active, bundle = current_analysis()
    if bundle is None:
        st.info("Run at least one differential-expression comparison first. Heatmaps use VST expression from the selected analysis.")
    elif st.session_state.annotation is None:
        st.warning("Gene-symbol annotation is unavailable, so symbol-based heatmaps cannot be generated yet.")
    else:
        st.subheader("Heatmap from genes of interest")
        history_names = list(st.session_state.analysis_history.keys())
        panel_analysis = st.selectbox(
            "Use expression from comparison",
            history_names,
            index=history_names.index(active) if active in history_names else 0,
            key="panel_analysis",
        )
        bundle = st.session_state.analysis_history[panel_analysis]

        st.write("Paste **gene symbols** such as `ERBB2`, `ESR1`, `BCL2` or `SOX2`.")
        text = st.text_area(
            "Gene symbols",
            height=170,
            placeholder="ERBB2\nESR1\nPGR\nMKI67\nBCL2",
        )
        zscore = st.checkbox("Z-score each gene", value=True, key="panel_zscore")

        if st.button("Generate gene-symbol heatmap"):
            requested = [x.strip() for x in text.replace(",", "\n").splitlines() if x.strip()]
            genes, missing = genes_from_symbols(requested, st.session_state.annotation)

            if missing:
                st.warning(f"{len(missing)} symbols were not found: " + ", ".join(missing[:30]))
            if not genes:
                st.error("None of the requested symbols were found in this dataset.")
            else:
                genes = [g for g in genes if g in bundle.vst_counts.index]
                mat = bundle.vst_counts.loc[genes, bundle.group_a_samples + bundle.group_b_samples]
                labels = display_labels(genes, st.session_state.annotation)
                st.plotly_chart(
                    heatmap_figure(
                        mat,
                        bundle.metadata,
                        zscore=zscore,
                        title=f"Custom gene panel · {panel_analysis}",
                        gene_labels=labels,
                    ),
                    use_container_width=True,
                )
                panel_export = mat.copy()
                panel_export.insert(0, "gene_symbol", labels)
                st.download_button(
                    "Download panel expression (CSV)",
                    to_csv_bytes(panel_export),
                    "gene_panel_VST_expression.csv",
                )


with gsea_tab:
    active, bundle = current_analysis()
    if bundle is None:
        st.info("Run a differential-expression comparison first.")
    elif st.session_state.annotation is None:
        st.warning("GSEA needs gene symbols. Gene annotation was not available for this session.")
    else:
        st.subheader("Gene Set Enrichment Analysis · pre-ranked GSEA")
        history_names = list(st.session_state.analysis_history.keys())
        gsea_analysis = st.selectbox(
            "Comparison",
            history_names,
            index=history_names.index(active) if active in history_names else 0,
            key="gsea_analysis",
        )
        bundle = st.session_state.analysis_history[gsea_analysis]
        raw_res = bundle.results[bundle.comparison_name]
        annotated_res = add_annotation_to_results(raw_res, st.session_state.annotation)

        st.caption(
            "All annotated genes are ranked by the PyDESeq2 Wald statistic. "
            "Positive enrichment is associated with Group B; negative enrichment with Group A."
        )

        g1, g2 = st.columns(2)
        library = g1.selectbox(
            "Gene-set collection",
            [
                "Hallmark",
                "GO Biological Process",
                "GO Molecular Function",
                "GO Cellular Component",
                "Reactome",
                "KEGG",
            ],
            index=0,
        )
        permutations = g2.selectbox("Permutations", [100, 250, 500, 1000], index=2)

        if st.button("▶ RUN GSEA", type="primary", use_container_width=True):
            try:
                with st.spinner("Running pre-ranked GSEA…"):
                    gsea_results, rank, actual_library = run_prerank_gsea(
                        annotated_res,
                        library=library,
                        species=organism,
                        permutation_num=int(permutations),
                        min_size=15,
                        max_size=500,
                        seed=7,
                        threads=2,
                    )
                    key = f"{gsea_analysis}::{library}"
                    st.session_state.gsea_history[key] = {
                        "results": gsea_results,
                        "rank": rank,
                        "library": actual_library,
                    }
                st.success(f"GSEA complete using {actual_library}.")
            except Exception as e:
                st.exception(e)

        key = f"{gsea_analysis}::{library}"
        if key in st.session_state.gsea_history:
            item = st.session_state.gsea_history[key]
            gres = item["results"].copy()
            st.caption(f"Enrichr library: `{item['library']}`")

            if gres.empty:
                st.warning("No gene sets passed the GSEA size/matching criteria.")
            else:
                st.dataframe(gres, use_container_width=True, height=500)

                if "Term" in gres.columns and "NES" in gres.columns:
                    plot_df = gres.copy()
                    if "FDR q-val" in plot_df.columns:
                        plot_df["_sig"] = -np.log10(
                            pd.to_numeric(plot_df["FDR q-val"], errors="coerce").clip(lower=1e-12)
                        )
                        plot_df = plot_df.sort_values("FDR q-val").head(30)
                        fig = px.scatter(
                            plot_df,
                            x="NES",
                            y="Term",
                            size="_sig",
                            hover_data=[c for c in ["NOM p-val", "FDR q-val", "Lead_genes"] if c in plot_df.columns],
                            title="Top GSEA pathways",
                        )
                    else:
                        plot_df = plot_df.reindex(plot_df["NES"].abs().sort_values(ascending=False).head(30).index)
                        fig = px.scatter(plot_df, x="NES", y="Term", title="Top GSEA pathways")
                    fig.update_layout(height=max(550, 22 * len(plot_df) + 200))
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "Download GSEA results (CSV)",
                    to_csv_bytes(gres, index=False),
                    file_name=f"{gsea_analysis.replace(' ', '_')}_{library.replace(' ', '_')}_GSEA.csv",
                )


with export_tab:
    active, bundle = current_analysis()
    if bundle is None:
        st.info("Run at least one differential-expression comparison first.")
    else:
        st.subheader("Download selected comparison")
        history_names = list(st.session_state.analysis_history.keys())
        export_analysis = st.selectbox(
            "Comparison",
            history_names,
            index=history_names.index(active) if active in history_names else 0,
            key="export_analysis",
        )
        bundle = st.session_state.analysis_history[export_analysis]
        raw_res = bundle.results[bundle.comparison_name]
        annotated_res = add_annotation_to_results(raw_res, st.session_state.annotation)

        st.download_button("Normalized counts", to_csv_bytes(bundle.norm_counts), "normalized_counts.csv")
        st.download_button("VST expression", to_csv_bytes(bundle.vst_counts), "VST_expression.csv")
        st.download_button("Metadata used", to_csv_bytes(bundle.metadata), "metadata_used.csv")
        st.download_button("Annotated DE results", to_csv_bytes(annotated_res), "differential_expression_annotated.csv")
        if st.session_state.annotation is not None:
            st.download_button(
                "Gene annotation",
                to_csv_bytes(st.session_state.annotation),
                "gene_annotation.csv",
            )

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("normalized_counts.csv", bundle.norm_counts.to_csv())
            z.writestr("VST_expression.csv", bundle.vst_counts.to_csv())
            z.writestr("metadata_used.csv", bundle.metadata.to_csv())
            z.writestr("differential_expression_annotated.csv", annotated_res.to_csv())
            if st.session_state.annotation is not None:
                z.writestr("gene_annotation.csv", st.session_state.annotation.to_csv())
        st.download_button(
            "⬇ Download all tables (ZIP)",
            zbuf.getvalue(),
            f"{export_analysis.replace(' ', '_')}_BulkRNAExplorer_results.zip",
            mime="application/zip",
            type="primary",
        )


st.divider()
st.caption("BulkRNA Explorer · exploratory bulk RNA-seq analysis · differential expression: PyDESeq2 · enrichment: GSEApy")
