from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _read_featurecounts_file(file_obj) -> pd.DataFrame:
    """Read a Galaxy/featureCounts single-sample table."""
    name = getattr(file_obj, "name", "uploaded.tabular")
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    df = pd.read_csv(file_obj, sep="\t", comment="#")
    if df.shape[1] < 2:
        raise ValueError(f"{name}: expected Geneid plus a counts column.")
    gene_col = "Geneid" if "Geneid" in df.columns else df.columns[0]
    # Prefer the last numeric-looking column. This also tolerates standard featureCounts annotation columns.
    candidates = [c for c in df.columns if c != gene_col]
    count_col = None
    for c in reversed(candidates):
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().mean() > 0.99:
            count_col = c
            break
    if count_col is None:
        raise ValueError(f"{name}: no numeric count column detected.")
    out = pd.DataFrame({"Geneid": df[gene_col].astype(str), "count": pd.to_numeric(df[count_col], errors="raise")})
    if out["Geneid"].duplicated().any():
        raise ValueError(f"{name}: duplicated Gene IDs detected.")
    if (out["count"] < 0).any() or not np.allclose(out["count"], np.round(out["count"])):
        raise ValueError(f"{name}: counts must be non-negative integers.")
    out["count"] = out["count"].astype(np.int64)
    out.attrs["source_name"] = name
    out.attrs["original_count_column"] = str(count_col)
    return out


def infer_sample_metadata(filename: str, original_column: str | None = None) -> dict:
    text = f"{filename} {original_column or ''}"
    sid = re.search(r"ID\s*([0-9]+)", text, flags=re.I)
    sample_id = f"ID{sid.group(1)}" if sid else Path(filename).stem[:35]

    low = text.lower()
    if "cddp" in low:
        condition = "CDDP_R"
    elif "dtx" in low:
        condition = "DTX_R"
    elif "sensitive" in low or "sensible" in low:
        condition = "Sensitive"
    else:
        condition = "Condition_1"

    exp = re.search(r"Exp\s*[_-]?\s*([0-9]+)", text, flags=re.I)
    batch = f"Exp{exp.group(1)}" if exp else "Batch_1"
    return {"sample": sample_id, "condition": condition, "batch": batch, "include": True}


def merge_featurecounts(uploaded_files: Iterable) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames = []
    meta = []
    messages = []
    reference_genes = None

    for f in uploaded_files:
        df = _read_featurecounts_file(f)
        genes = df["Geneid"].tolist()
        if reference_genes is None:
            reference_genes = genes
        elif genes != reference_genes:
            raise ValueError(
                f"{getattr(f, 'name', 'file')}: Gene IDs/order do not match the other files. "
                "Use featureCounts outputs generated with the same annotation."
            )
        m = infer_sample_metadata(getattr(f, "name", "sample"), df.attrs.get("original_count_column"))
        frames.append(df["count"].rename(m["sample"]))
        meta.append(m)
        messages.append(f"{m['sample']}: {len(df):,} genes")

    if not frames:
        raise ValueError("No files were uploaded.")
    counts = pd.concat(frames, axis=1)
    counts.index = pd.Index(reference_genes, name="Geneid")
    metadata = pd.DataFrame(meta)
    if metadata["sample"].duplicated().any():
        # Make names unique while keeping them readable.
        seen = {}
        names = []
        for s in metadata["sample"]:
            seen[s] = seen.get(s, 0) + 1
            names.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
        metadata["sample"] = names
        counts.columns = names
    return counts, metadata, messages


def strip_ensembl_version(series: pd.Index | pd.Series) -> pd.Index:
    return pd.Index([str(x).split(".")[0] for x in series])
