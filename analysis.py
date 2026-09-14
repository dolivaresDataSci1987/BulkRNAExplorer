from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.preprocessing import deseq2_norm


@dataclass
class AnalysisBundle:
    counts_filtered: pd.DataFrame          # genes x samples
    metadata: pd.DataFrame                # indexed by samples
    norm_counts: pd.DataFrame             # genes x samples
    vst_counts: pd.DataFrame              # genes x samples
    dds: object
    results: Dict[str, pd.DataFrame]
    reference: str
    design: str


def filter_counts(counts_gxs: pd.DataFrame, min_count: int = 10, min_samples: int = 3) -> pd.DataFrame:
    keep = (counts_gxs >= int(min_count)).sum(axis=1) >= int(min_samples)
    return counts_gxs.loc[keep].copy()


def quick_normalize(counts_gxs: pd.DataFrame) -> pd.DataFrame:
    # pydeseq2 expects samples x genes
    sxg = counts_gxs.T.astype(int)
    norm, _ = deseq2_norm(sxg)
    if not isinstance(norm, pd.DataFrame):
        norm = pd.DataFrame(norm, index=sxg.index, columns=sxg.columns)
    return norm.T


def run_deseq(
    counts_gxs: pd.DataFrame,
    metadata: pd.DataFrame,
    reference: str,
    design: str = "~condition",
    min_count: int = 10,
    min_samples: int = 3,
    alpha: float = 0.05,
    n_cpus: int = 2,
) -> AnalysisBundle:
    md = metadata.copy()
    md = md.loc[md["include"].astype(bool)].copy()
    if md.empty:
        raise ValueError("No samples are included.")
    md["sample"] = md["sample"].astype(str)
    md = md.set_index("sample", drop=True)
    selected = [s for s in md.index if s in counts_gxs.columns]
    md = md.loc[selected]
    counts = counts_gxs.loc[:, selected].copy()
    if len(selected) < 2:
        raise ValueError("At least two samples are required.")
    if "condition" not in md.columns:
        raise ValueError("Metadata must contain a condition column.")
    levels = list(pd.unique(md["condition"].astype(str)))
    if reference not in levels:
        raise ValueError(f"Reference condition '{reference}' not present in included samples.")
    if len(levels) < 2:
        raise ValueError("At least two conditions are required for differential expression.")

    filtered = filter_counts(counts, min_count=min_count, min_samples=min(min_samples, len(selected)))
    if filtered.shape[0] < 10:
        raise ValueError("Too few genes remain after filtering. Reduce the gene filtering thresholds.")

    sxg = filtered.T.astype(int)
    # Cast categorical columns to string to keep formulaic levels predictable.
    for col in md.columns:
        if col != "include":
            md[col] = md[col].astype(str)

    dds = DeseqDataSet(
        counts=sxg,
        metadata=md,
        design=design,
        refit_cooks=True,
        n_cpus=n_cpus,
        quiet=True,
    )
    dds.deseq2()

    norm = pd.DataFrame(dds.layers["normed_counts"], index=sxg.index, columns=sxg.columns).T
    # VST is ideal for PCA/heatmaps. If it fails for a pathological dataset, fall back to log2 normalized counts.
    try:
        dds.vst(use_design=False)
        vst = pd.DataFrame(dds.layers["vst_counts"], index=sxg.index, columns=sxg.columns).T
    except Exception:
        vst = np.log2(norm + 1.0)

    results = {}
    for tested in levels:
        if tested == reference:
            continue
        stats = DeseqStats(dds, contrast=["condition", tested, reference], alpha=alpha, n_cpus=n_cpus, quiet=True)
        stats.summary()
        res = stats.results_df.copy()
        res.index.name = "Geneid"
        results[f"{tested} vs {reference}"] = res

    return AnalysisBundle(
        counts_filtered=filtered,
        metadata=md,
        norm_counts=norm,
        vst_counts=vst,
        dds=dds,
        results=results,
        reference=reference,
        design=design,
    )


def pca_table(vst_gxs: pd.DataFrame, metadata_indexed: pd.DataFrame, top_n: int = 5000) -> tuple[pd.DataFrame, np.ndarray]:
    x = vst_gxs.T.copy()  # samples x genes
    variances = x.var(axis=0).sort_values(ascending=False)
    use = variances.head(min(top_n, len(variances))).index
    x = x.loc[:, use]
    n_components = min(3, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(x)
    cols = [f"PC{i+1}" for i in range(n_components)]
    out = pd.DataFrame(coords, index=x.index, columns=cols)
    out = out.join(metadata_indexed, how="left")
    return out, pca.explained_variance_ratio_


def classify_degs(res: pd.DataFrame, padj_cut: float, lfc_cut: float) -> pd.Series:
    status = pd.Series("NS", index=res.index, dtype="object")
    valid = res["padj"].notna()
    status.loc[valid & (res["padj"] <= padj_cut) & (res["log2FoldChange"] >= lfc_cut)] = "UP"
    status.loc[valid & (res["padj"] <= padj_cut) & (res["log2FoldChange"] <= -lfc_cut)] = "DOWN"
    return status


def common_deg_sets(results: Dict[str, pd.DataFrame], padj_cut: float, lfc_cut: float):
    ups, downs = {}, {}
    for name, res in results.items():
        s = classify_degs(res, padj_cut, lfc_cut)
        ups[name] = set(s.index[s == "UP"])
        downs[name] = set(s.index[s == "DOWN"])
    common_up = set.intersection(*ups.values()) if ups else set()
    common_down = set.intersection(*downs.values()) if downs else set()
    return ups, downs, common_up, common_down
