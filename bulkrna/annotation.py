from __future__ import annotations

from typing import Iterable

import pandas as pd


def strip_ensembl_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def annotate_ensembl_ids(gene_ids: Iterable[str], species: str = "human", chunk_size: int = 1000) -> pd.DataFrame:
    import mygene

    original = [str(x) for x in gene_ids]
    clean = [strip_ensembl_version(x) for x in original]
    unique_clean = list(dict.fromkeys(clean))

    mg = mygene.MyGeneInfo()
    mapping: dict[str, dict] = {}

    for i in range(0, len(unique_clean), int(chunk_size)):
        chunk = unique_clean[i:i + int(chunk_size)]
        hits = mg.querymany(
            chunk,
            scopes="ensembl.gene",
            fields="symbol,name,entrezgene",
            species=species,
            as_dataframe=False,
            returnall=False,
            verbose=False,
        )
        for hit in hits:
            q = str(hit.get("query", ""))
            if not q or hit.get("notfound"):
                continue
            candidate = {
                "gene_symbol": hit.get("symbol"),
                "gene_name": hit.get("name"),
                "entrezgene": hit.get("entrezgene"),
            }
            if q not in mapping or (not mapping[q].get("gene_symbol") and candidate.get("gene_symbol")):
                mapping[q] = candidate

    rows = []
    for original_id, clean_id in zip(original, clean):
        hit = mapping.get(clean_id, {})
        symbol = hit.get("gene_symbol")
        name = hit.get("gene_name")
        rows.append(
            {
                "Geneid": original_id,
                "ensembl_id": clean_id,
                "gene_symbol": "" if pd.isna(symbol) or symbol is None else str(symbol),
                "gene_name": "" if pd.isna(name) or name is None else str(name),
                "entrezgene": hit.get("entrezgene"),
            }
        )
    return pd.DataFrame(rows).set_index("Geneid", drop=True)


def add_annotation_to_results(res: pd.DataFrame, annotation: pd.DataFrame | None) -> pd.DataFrame:
    out = res.copy()
    if annotation is None or annotation.empty:
        out.insert(0, "gene_name", "")
        out.insert(0, "gene_symbol", out.index.astype(str))
        out.insert(0, "ensembl_id", [strip_ensembl_version(x) for x in out.index])
        return out

    ann = annotation.reindex(out.index)
    symbols = ann["gene_symbol"].fillna("").astype(str)
    fallback = pd.Series(out.index.astype(str), index=out.index)
    symbols = symbols.where(symbols.str.len() > 0, fallback)
    names = ann["gene_name"].fillna("").astype(str)
    ens = ann["ensembl_id"].fillna(pd.Series([strip_ensembl_version(x) for x in out.index], index=out.index))
    out.insert(0, "gene_name", names.values)
    out.insert(0, "gene_symbol", symbols.values)
    out.insert(0, "ensembl_id", ens.values)
    return out


def display_labels(gene_ids: Iterable[str], annotation: pd.DataFrame | None) -> list[str]:
    ids = [str(x) for x in gene_ids]
    if annotation is None or annotation.empty:
        return ids
    ann = annotation.reindex(ids)
    symbols = ann["gene_symbol"].fillna("").astype(str).tolist()
    labels = [s if s else gid for gid, s in zip(ids, symbols)]
    counts = pd.Series(labels).value_counts()
    out = []
    for gid, label in zip(ids, labels):
        if counts.get(label, 0) > 1:
            out.append(f"{label} · {strip_ensembl_version(gid)}")
        else:
            out.append(label)
    return out


def genes_from_symbols(symbols: Iterable[str], annotation: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    requested = [str(x).strip() for x in symbols if str(x).strip()]
    if annotation is None or annotation.empty:
        return [], requested

    work = annotation.copy()
    work["_symbol_upper"] = work["gene_symbol"].fillna("").astype(str).str.upper()
    lookup: dict[str, list[str]] = {}
    for gene_id, sym in work["_symbol_upper"].items():
        if sym:
            lookup.setdefault(sym, []).append(str(gene_id))

    found: list[str] = []
    missing: list[str] = []
    for symbol in requested:
        matches = lookup.get(symbol.upper(), [])
        if matches:
            found.extend(matches)
        else:
            missing.append(symbol)

    return list(dict.fromkeys(found)), missing
