from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from bulkrna.io import merge_featurecounts, strip_ensembl_version
from bulkrna.analysis import run_deseq, pca_table, classify_degs, common_deg_sets
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
st.caption("Bulk RNA-seq analysis from raw featureCounts. Upload → check groups → analyze → download.")

for key, default in {
    "counts": None, "metadata": None, "bundle": None, "upload_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def to_csv_bytes(df: pd.DataFrame, index=True):
    return df.to_csv(index=index).encode("utf-8")


def reset_analysis():
    st.session_state.bundle = None


with st.sidebar:
    st.header("Project")
    st.write("**1. Upload data**")
    st.write("**2. Check samples**")
    st.write("**3. Run analysis**")
    st.divider()
    st.caption("No database is used by this app. Uploaded data are processed for the current session.")
    if st.button("Reset project", use_container_width=True):
        for k in ["counts", "metadata", "bundle", "upload_signature"]:
            st.session_state[k] = None
        st.rerun()

upload_tab, setup_tab, analysis_tab, panels_tab, export_tab = st.tabs([
    "1 · Upload", "2 · Samples", "3 · Analysis", "4 · Gene panels", "5 · Export"
])

with upload_tab:
    st.subheader("Upload featureCounts files")
    st.write("Select all featureCounts `.tabular`, `.txt` or `.tsv` files belonging to the experiment.")
    uploaded = st.file_uploader(
        "featureCounts files", type=["tabular", "txt", "tsv", "csv"], accept_multiple_files=True,
        help="One featureCounts file per biological sample is ideal."
    )
    if uploaded:
        signature = tuple((f.name, f.size) for f in uploaded)
        if signature != st.session_state.upload_signature:
            try:
                counts, md, messages = merge_featurecounts(uploaded)
                st.session_state.counts = counts
                st.session_state.metadata = md
                st.session_state.upload_signature = signature
                reset_analysis()
                st.success(f"Loaded {counts.shape[1]} samples and {counts.shape[0]:,} genes.")
            except Exception as e:
                st.error(f"Could not import the files: {e}")
        if st.session_state.counts is not None:
            c = st.session_state.counts
            m1, m2, m3 = st.columns(3)
            m1.metric("Samples", c.shape[1])
            m2.metric("Genes", f"{c.shape[0]:,}")
            m3.metric("Raw counts", "Valid integers")
            st.dataframe(c.head(12), use_container_width=True)
    else:
        st.info("Upload the featureCounts files to begin.")

with setup_tab:
    if st.session_state.counts is None:
        st.info("Upload featureCounts files first.")
    else:
        st.subheader("Check sample groups")
        st.write("The app tries to detect sample names, conditions and Exp/batch automatically. Correct anything that is wrong.")
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
            old_names = list(st.session_state.metadata["sample"])
            new_names = list(edited["sample"])
            if len(set(new_names)) != len(new_names):
                st.error("Sample names must be unique.")
            else:
                rename = dict(zip(old_names, new_names))
                st.session_state.counts = st.session_state.counts.rename(columns=rename)
                st.session_state.metadata = edited.copy()
                reset_analysis()
        md = st.session_state.metadata
        included = md.loc[md["include"].astype(bool)]
        if not included.empty:
            st.write("Detected groups")
            st.dataframe(included.groupby("condition").size().rename("n samples").to_frame(), use_container_width=True)

with analysis_tab:
    if st.session_state.counts is None or st.session_state.metadata is None:
        st.info("Upload data and check sample groups first.")
    else:
        st.subheader("Run complete analysis")
        md = st.session_state.metadata
        included = md.loc[md["include"].astype(bool)].copy()
        conditions = list(pd.unique(included["condition"].astype(str)))
        if len(conditions) < 2:
            st.warning("At least two conditions are needed.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            reference = c1.selectbox("Reference condition", conditions, index=conditions.index("Sensitive") if "Sensitive" in conditions else 0)
            min_count = c2.number_input("Min. count", min_value=0, value=10, step=1)
            min_samples = c3.number_input("Min. samples", min_value=1, max_value=max(1, len(included)), value=min(3, len(included)), step=1)
            use_batch = c4.checkbox("Correct for batch", value=False, help="Enable only if the Batch/experiment column represents a real technical/experimental batch.")
            design = "~batch + condition" if use_batch else "~condition"

            st.caption(f"Model: `{design}`. All non-reference conditions will be compared with **{reference}**.")
            if st.button("▶ RUN COMPLETE ANALYSIS", type="primary", use_container_width=True):
                try:
                    with st.spinner("Running normalization, DESeq2, VST, PCA and differential-expression contrasts…"):
                        bundle = run_deseq(
                            st.session_state.counts, st.session_state.metadata,
                            reference=reference, design=design,
                            min_count=int(min_count), min_samples=int(min_samples), alpha=0.05, n_cpus=2,
                        )
                        st.session_state.bundle = bundle
                    st.success("Analysis complete.")
                except Exception as e:
                    st.exception(e)

        bundle = st.session_state.bundle
        if bundle is not None:
            st.divider()
            pca_tab, de_tab, common_tab, qc_tab = st.tabs(["PCA", "Differential expression", "Common DEGs", "QC"])

            with pca_tab:
                pca_df, explained = pca_table(bundle.vst_counts, bundle.metadata, top_n=5000)
                color_options = [c for c in ["condition", "batch"] if c in pca_df.columns]
                color_by = st.selectbox("Colour by", color_options, key="pca_color")
                st.plotly_chart(pca_figure(pca_df, explained, color_by), use_container_width=True)
                st.dataframe(pca_df, use_container_width=True)

            with de_tab:
                contrast = st.selectbox("Contrast", list(bundle.results.keys()), key="de_contrast")
                r = bundle.results[contrast]
                a, b = st.columns(2)
                padj_cut = a.number_input("Adjusted p-value ≤", min_value=0.000001, max_value=1.0, value=0.05, format="%.4f", key="de_padj")
                lfc_cut = b.number_input("|log2FC| ≥", min_value=0.0, value=1.0, step=0.1, key="de_lfc")
                status = classify_degs(r, padj_cut, lfc_cut)
                rr = r.copy()
                rr.insert(0, "status", status)
                n_up = int((status == "UP").sum()); n_down = int((status == "DOWN").sum())
                x1, x2, x3 = st.columns(3)
                x1.metric("UP", n_up); x2.metric("DOWN", n_down); x3.metric("Significant", n_up+n_down)
                st.plotly_chart(volcano_figure(r, padj_cut, lfc_cut), use_container_width=True)
                show = st.radio("Show", ["Significant only", "All genes", "UP only", "DOWN only"], horizontal=True)
                if show == "Significant only":
                    table = rr[rr["status"].isin(["UP", "DOWN"])]
                elif show == "UP only":
                    table = rr[rr["status"] == "UP"]
                elif show == "DOWN only":
                    table = rr[rr["status"] == "DOWN"]
                else:
                    table = rr
                st.dataframe(table, use_container_width=True, height=500)
                st.download_button("Download this DEG table (CSV)", to_csv_bytes(table), file_name=f"{contrast.replace(' ', '_')}_DEGs.csv", mime="text/csv")

            with common_tab:
                if len(bundle.results) < 2:
                    st.info("Common DEG analysis needs at least two contrasts vs the same reference.")
                else:
                    a, b = st.columns(2)
                    cp = a.number_input("Common DEG padj ≤", min_value=0.000001, max_value=1.0, value=0.05, format="%.4f")
                    cl = b.number_input("Common DEG |log2FC| ≥", min_value=0.0, value=1.0, step=0.1)
                    ups, downs, common_up, common_down = common_deg_sets(bundle.results, cp, cl)
                    m1, m2 = st.columns(2)
                    m1.metric("Common UP", len(common_up)); m2.metric("Common DOWN", len(common_down))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Common UP genes**")
                        up_df = pd.DataFrame({"Geneid": sorted(common_up)})
                        st.dataframe(up_df, use_container_width=True, height=300)
                        st.download_button("Download Common UP", to_csv_bytes(up_df, index=False), "common_UP.csv")
                    with col2:
                        st.write("**Common DOWN genes**")
                        down_df = pd.DataFrame({"Geneid": sorted(common_down)})
                        st.dataframe(down_df, use_container_width=True, height=300)
                        st.download_button("Download Common DOWN", to_csv_bytes(down_df, index=False), "common_DOWN.csv")

            with qc_tab:
                libs = bundle.counts_filtered.sum(axis=0).rename("library_size").to_frame()
                libs["sample"] = libs.index
                fig = px.bar(libs, x="sample", y="library_size", title="Library size after gene filtering")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Genes retained for modelling: **{bundle.counts_filtered.shape[0]:,}**")

with panels_tab:
    bundle = st.session_state.bundle
    if bundle is None:
        st.info("Run the analysis first. Gene-panel heatmaps use VST expression.")
    else:
        st.subheader("Gene-panel heatmap")
        st.write("Paste Ensembl IDs (one per line or separated by commas). Gene-symbol mapping will be added in the next version.")
        text = st.text_area("Genes", height=160, placeholder="ENSG00000141736\nENSG00000146648\n...")
        top_n = st.number_input("If no genes are entered, show top variable genes", min_value=5, max_value=200, value=30, step=5)
        zscore = st.checkbox("Z-score each gene", value=True)
        if st.button("Generate heatmap"):
            vst = bundle.vst_counts
            if text.strip():
                requested = [x.strip() for x in text.replace(",", "\n").splitlines() if x.strip()]
                exact = {str(g): g for g in vst.index}
                versionless = {str(g).split(".")[0]: g for g in vst.index}
                found, missing = [], []
                for q in requested:
                    if q in exact:
                        found.append(exact[q])
                    elif q.split(".")[0] in versionless:
                        found.append(versionless[q.split(".")[0]])
                    else:
                        missing.append(q)
                genes = list(dict.fromkeys(found))
                if missing:
                    st.warning(f"{len(missing)} genes were not found: " + ", ".join(missing[:20]))
            else:
                genes = list(vst.var(axis=1).sort_values(ascending=False).head(int(top_n)).index)
            if genes:
                mat = vst.loc[genes]
                st.plotly_chart(heatmap_figure(mat, bundle.metadata, zscore=zscore, title=f"Gene panel ({len(genes)} genes)"), use_container_width=True)
                st.download_button("Download VST expression (CSV)", to_csv_bytes(mat), "gene_panel_VST_expression.csv")

with export_tab:
    bundle = st.session_state.bundle
    if bundle is None:
        st.info("Run the analysis first.")
    else:
        st.subheader("Download project results")
        st.download_button("Normalized counts", to_csv_bytes(bundle.norm_counts), "normalized_counts.csv")
        st.download_button("VST expression", to_csv_bytes(bundle.vst_counts), "VST_expression.csv")
        st.download_button("Metadata", to_csv_bytes(bundle.metadata), "metadata_used.csv")

        # Build one ZIP in memory with all key tables.
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("normalized_counts.csv", bundle.norm_counts.to_csv())
            z.writestr("VST_expression.csv", bundle.vst_counts.to_csv())
            z.writestr("metadata_used.csv", bundle.metadata.to_csv())
            for name, df in bundle.results.items():
                z.writestr(f"DE_{name.replace(' ', '_')}.csv", df.to_csv())
        st.download_button("⬇ Download all tables (ZIP)", zbuf.getvalue(), "BulkRNAExplorer_results.zip", mime="application/zip", type="primary")

st.divider()
st.caption("BulkRNA Explorer · prototype for exploratory bulk RNA-seq analysis. Differential expression: PyDESeq2.")
