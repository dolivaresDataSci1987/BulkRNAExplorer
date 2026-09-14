from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from .analysis import classify_degs


def pca_figure(pca_df: pd.DataFrame, explained, color="condition"):
    xlab = f"PC1 ({explained[0]*100:.1f}%)" if len(explained) > 0 else "PC1"
    ylab = f"PC2 ({explained[1]*100:.1f}%)" if len(explained) > 1 else "PC2"
    hover = [c for c in ["comparison_group", "condition", "batch"] if c in pca_df.columns]
    fig = px.scatter(
        pca_df.reset_index(names="sample"),
        x="PC1",
        y="PC2",
        color=color,
        text="sample",
        hover_data=hover,
        labels={"PC1": xlab, "PC2": ylab},
    )
    fig.update_traces(textposition="top center", marker={"size": 11})
    fig.update_layout(height=560, legend_title=color)
    return fig


def volcano_figure(res: pd.DataFrame, padj_cut: float, lfc_cut: float):
    d = res.copy()
    d["status"] = classify_degs(d, padj_cut, lfc_cut)
    d["minus_log10_padj"] = -np.log10(pd.to_numeric(d["padj"], errors="coerce").clip(lower=1e-300))
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
            "Geneid": True,
            "gene_name": True,
            "baseMean": ":.2f",
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


def _hierarchical_order(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.arange(values.shape[0])
    distances = pdist(values, metric="euclidean")
    if distances.size == 0 or not np.isfinite(distances).all() or np.allclose(distances, 0):
        return np.arange(values.shape[0])
    tree = linkage(distances, method="average", optimal_ordering=True)
    return leaves_list(tree)


def prepare_heatmap_matrix(
    matrix_gxs: pd.DataFrame,
    zscore: bool = True,
    cluster_genes: bool = False,
    cluster_samples: bool = False,
) -> pd.DataFrame:
    """Prepare and optionally hierarchically reorder a genes x samples matrix."""
    d = matrix_gxs.copy().astype(float)
    if zscore:
        sd = d.std(axis=1).replace(0, np.nan)
        d = d.sub(d.mean(axis=1), axis=0).div(sd, axis=0).fillna(0)

    if cluster_genes and d.shape[0] > 1:
        order = _hierarchical_order(d.to_numpy())
        d = d.iloc[order, :]
    if cluster_samples and d.shape[1] > 1:
        order = _hierarchical_order(d.to_numpy().T)
        d = d.iloc[:, order]
    return d


def heatmap_figure(
    matrix_gxs: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    zscore=True,
    title="Heatmap",
    gene_labels=None,
    cluster_genes=False,
    cluster_samples=False,
):
    d = prepare_heatmap_matrix(
        matrix_gxs,
        zscore=zscore,
        cluster_genes=cluster_genes,
        cluster_samples=cluster_samples,
    )

    label_map = None
    if gene_labels is not None:
        label_map = {str(g): str(label) for g, label in zip(matrix_gxs.index, gene_labels)}
    y_labels = [label_map.get(str(g), str(g)) if label_map else str(g) for g in d.index]

    md = sample_metadata.copy()
    if "sample" in md.columns:
        md = md.set_index("sample", drop=False)
    x_labels = []
    for s in d.columns:
        if s in md.index and "comparison_group" in md.columns:
            x_labels.append(f"{s} · {md.loc[s, 'comparison_group']}")
        else:
            x_labels.append(str(s))

    heatmap_kwargs = dict(z=d.values, x=x_labels, y=y_labels)
    if zscore:
        heatmap_kwargs.update(
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "Z-score"},
        )
    else:
        heatmap_kwargs.update(
            colorscale="Viridis",
            colorbar={"title": "Expression"},
        )

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    subtitle = []
    if cluster_genes:
        subtitle.append("genes clustered")
    if cluster_samples:
        subtitle.append("samples clustered")
    full_title = title + (" · " + ", ".join(subtitle) if subtitle else "")
    fig.update_layout(
        title=full_title,
        height=max(500, min(1500, 20 * len(d) + 260)),
        xaxis_title="Samples",
        yaxis_title="Genes",
    )
    return fig
