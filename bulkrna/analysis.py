from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.preprocessing import deseq2_norm


@dataclass
class AnalysisBundle:
    counts_filtered: pd.DataFrame
    metadata: pd.DataFrame
    norm_counts: pd.DataFrame
    vst_counts: pd.DataFrame
    dds: object
    results: Dict[str, pd.DataFrame]
    comparison_name: str
    group_a_name: str
    group_b_name: str
    group_a_samples: list[str]
    group_b_samples: list[str]
    design: str


def filter_counts(counts_gxs: pd.DataFrame, min_count: int = 10, min_samples: int = 3) -> pd.DataFrame:
    keep = (counts_gxs >= int(min_count)).sum(axis=1) >= int(min_samples)
    return counts_gxs.loc[keep].copy()


def quick_normalize(counts_gxs: pd.DataFrame) -> pd.DataFrame:
    sxg = counts_gxs.T.astype(int)
    norm, _ = deseq2_norm(sxg)
    if not isinstance(norm, pd.DataFrame):
        norm = pd.DataFrame(norm, index=sxg.index, columns=sxg.columns)
    return norm.T


def run_deseq_comparison(
    counts_gxs: pd.DataFrame,
    metadata: pd.DataFrame,
    group_a_samples: Sequence[str],
    group_b_samples: Sequence[str],
    group_a_name: str = "Reference",
    group_b_name: str = "Tested",
    use_batch: bool = False,
    min_count: int = 10,
    min_samples: int = 2,
    alpha: float = 0.05,
    n_cpus: int = 2,
) -> AnalysisBundle:
    """Run one explicit sample-vs-sample-group differential-expression comparison.

    log2FoldChange is Group B / Group A. Positive values mean higher expression
    in Group B; negative values mean higher expression in Group A.
    """
    a = [str(x) for x in group_a_samples]
    b = [str(x) for x in group_b_samples]
    if not a or not b:
        raise ValueError("Select at least one sample in each comparison group.")
    overlap = sorted(set(a).intersection(b))
    if overlap:
        raise ValueError("A sample cannot belong to both groups: " + ", ".join(overlap))
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Differential expression should use at least two biological replicates in each group.")

    group_a_name = str(group_a_name).strip() or "Group_A"
    group_b_name = str(group_b_name).strip() or "Group_B"
    if group_a_name == group_b_name:
        raise ValueError("The two comparison groups need different names.")

    md = metadata.copy()
    md["sample"] = md["sample"].astype(str)
    md = md.set_index("sample", drop=True)
    selected = a + b
    missing = [s for s in selected if s not in counts_gxs.columns or s not in md.index]
    if missing:
        raise ValueError("Selected samples were not found: " + ", ".join(missing))

    md = md.loc[selected].copy()
    a_set = set(a)
    md["comparison_group"] = [group_a_name if s in a_set else group_b_name for s in md.index]
    counts = counts_gxs.loc[:, selected].copy()

    filtered = filter_counts(
        counts,
        min_count=int(min_count),
        min_samples=min(int(min_samples), len(selected)),
    )
    if filtered.shape[0] < 10:
        raise ValueError("Too few genes remain after filtering. Reduce the gene filtering thresholds.")

    sxg = filtered.T.astype(int)
    for col in md.columns:
        if col != "include":
            md[col] = md[col].astype(str)

    design = "~batch + comparison_group" if use_batch else "~comparison_group"
    if use_batch:
        if "batch" not in md.columns:
            raise ValueError("Batch correction was requested but no batch column is available.")
        if md["batch"].nunique() < 2:
            raise ValueError("Batch correction needs at least two batch levels.")

    dds = DeseqDataSet(
        counts=sxg,
        metadata=md,
        design=design,
        refit_cooks=True,
        n_cpus=n_cpus,
        quiet=True,
    )
    dds.deseq2()

    norm = pd.DataFrame(
        dds.layers["normed_counts"],
        index=sxg.index,
        columns=sxg.columns,
    ).T
    try:
        dds.vst(use_design=False)
        vst = pd.DataFrame(
            dds.layers["vst_counts"],
            index=sxg.index,
            columns=sxg.columns,
        ).T
    except Exception:
        vst = np.log2(norm + 1.0)

    stats = DeseqStats(
        dds,
        contrast=["comparison_group", group_b_name, group_a_name],
        alpha=alpha,
        n_cpus=n_cpus,
        quiet=True,
    )
    stats.summary()
    res = stats.results_df.copy()
    res.index.name = "Geneid"

    comparison_name = f"{group_b_name} vs {group_a_name}"
    return AnalysisBundle(
        counts_filtered=filtered,
        metadata=md,
        norm_counts=norm,
        vst_counts=vst,
        dds=dds,
        results={comparison_name: res},
        comparison_name=comparison_name,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        group_a_samples=a,
        group_b_samples=b,
        design=design,
    )


def pca_table(
    vst_gxs: pd.DataFrame,
    metadata_indexed: pd.DataFrame,
    top_n: int = 5000,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = vst_gxs.T.copy()
    variances = x.var(axis=0).sort_values(ascending=False)
    use = variances.head(min(int(top_n), len(variances))).index
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


def top_deg_gene_ids(
    res: pd.DataFrame,
    n: int,
    padj_cut: float = 0.05,
    lfc_cut: float = 0.0,
) -> list[str]:
    """Top N significant DE genes, ranked by padj then |log2FC|."""
    d = res.copy()
    d = d[d["padj"].notna()]
    d = d[(d["padj"] <= float(padj_cut)) & (d["log2FoldChange"].abs() >= float(lfc_cut))]
    if d.empty:
        return []
    d = d.assign(_abs_lfc=d["log2FoldChange"].abs())
    d = d.sort_values(["padj", "_abs_lfc"], ascending=[True, False])
    return [str(x) for x in d.head(int(n)).index]


def common_deg_sets(results: Dict[str, pd.DataFrame], padj_cut: float, lfc_cut: float):
    ups, downs = {}, {}
    for name, res in results.items():
        s = classify_degs(res, padj_cut, lfc_cut)
        ups[name] = set(s.index[s == "UP"])
        downs[name] = set(s.index[s == "DOWN"])
    common_up = set.intersection(*ups.values()) if ups else set()
    common_down = set.intersection(*downs.values()) if downs else set()
    return ups, downs, common_up, common_down
