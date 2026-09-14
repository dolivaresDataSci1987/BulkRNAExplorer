from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analysis import classify_degs


def pca_figure(pca_df: pd.DataFrame, explained, color="comparison_group"):
    xlab = f"PC1 ({explained[0]*100:.1f}%)" if len(explained) > 0 else "PC1"
    ylab = f"PC2 ({explained[1]*100:.1f}%)" if len(explained) > 1 else "PC2"
    hover_cols = [c for c in ["comparison_group", "condition", "batch"] if c in pca_df.columns]
    fig = px.scatter(
        pca_df.reset_index(names="sample"),
        x="PC1",
        y="PC2",
        color=color,
        text="sample",
        hover_data=hover_cols,
        labels={"PC1": xlab, "PC2": ylab},
    )
    fig.update_traces(textposition="top center", marker={"size": 11})
    fig.update_layout(height=560, legend_title=color)
    return fig


def volcano_figure(res: pd.DataFrame, padj_cut: float, lfc_cut: float):
    d = res.copy()
    d["status"] = classify_degs(d, padj_cut, lfc_cut)
    d["minus_log10_padj"] = -np.log10(d["padj"].clip(lower=1e-300))
    d["Geneid"] = d.index.astype(str)
    if "gene_symbol" not in d.columns:
        d["gene_symbol"] = d["Geneid"]
    if "gene_name" not in d.columns:
        d["gene_name"] = ""

    fig = px.scatter(
        d,
        x="log2FoldChange",
        y="minus_log10_padj",
        color="status",
        hover_name="gene_symbol",
        hover_data={
            "gene_name": True,
            "Geneid": True,
            "baseMean": ":.3g",
            "pvalue": ":.3g",
            "padj": ":.3g",
        },
        category_orders={"status": ["NS", "UP", "DOWN"]},
        labels={
            "log2FoldChange": "log2 fold-change",
            "minus_log10_padj": "−log10 adjusted p-value",
        },
    )
    fig.add_vline(x=lfc_cut, line_dash="dash")
    fig.add_vline(x=-lfc_cut, line_dash="dash")
    if padj_cut > 0:
        fig.add_hline(y=-np.log10(padj_cut), line_dash="dash")
    fig.update_layout(height=600)
    return fig


def heatmap_figure(
    matrix_gxs: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    zscore: bool = True,
    title: str = "Heatmap",
    gene_labels: list[str] | None = None,
):
    d = matrix_gxs.copy().astype(float)
    if zscore:
        sd = d.std(axis=1).replace(0, np.nan)
        d = d.sub(d.mean(axis=1), axis=0).div(sd, axis=0).fillna(0)

    y = gene_labels if gene_labels is not None else [str(x) for x in d.index]
    fig = go.Figure(
        data=go.Heatmap(
            z=d.values,
            x=d.columns,
            y=y,
            colorbar={"title": "Z" if zscore else "Expression"},
            hovertemplate="Gene: %{y}<br>Sample: %{x}<br>Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(500, min(1400, 20 * len(d) + 250)),
        xaxis_title="Samples",
        yaxis_title="Genes",
    )
    return fig
