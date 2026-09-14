# BulkRNA Explorer — User Guide

**A click-first guide for bulk RNA-seq analysis. No coding required.**

BulkRNA Explorer is designed for collaborators who want to analyse bulk RNA-seq count data in a browser without installing R, Python, Galaxy, or command-line tools.

> **Important:** BulkRNA Explorer is intended for exploratory analysis. For publication-critical results, confirm the final analysis with an established bioinformatics workflow and retain the original raw data and metadata.

---

## Quick start

1. Open BulkRNA Explorer in your browser.
2. Upload one raw featureCounts table per biological sample/library.
3. Check sample names, conditions, and batch labels in **2 · Samples**.
4. In **3 · Differential expression**, choose **Group A** (reference/control) and **Group B** (tested condition).
5. Click **RUN DIFFERENTIAL EXPRESSION**.
6. Explore PCA, volcano plots, DEG tables, top-DEG heatmaps, and overlaps.
7. Use **4 · Gene heatmaps** for any custom gene list.
8. Use **5 · GSEA** for full-library enrichment or **6 · Custom GSEA** for selected pathways or your own gene sets.
9. Download results from the relevant panel or from **7 · Download**.

---

# 1. Upload

## What files do I need?

Upload raw **featureCounts** output tables, ideally one file per biological sample/library. Accepted extensions include `.tabular`, `.txt`, `.tsv`, and `.csv`.

The count values must be **raw non-negative integer counts**.

### Use

- raw featureCounts counts
- one biological sample/library per file
- consistent gene IDs across all uploaded files

### Do not use

- TPM
- FPKM
- CPM
- already log-transformed expression
- R1 and R2 FASTQ reads as if they were separate biological replicates

Paired-end R1/R2 reads belong to the same sequencing library and are not biological replicates.

## Gene annotation

For Human or Mouse datasets, the app attempts to map Ensembl IDs to gene symbols and gene names. This requires internet access from the server. Differential expression can still run if annotation temporarily fails, but symbol-based features may not be available.

---

# 2. Samples

Use **2 · Samples** to check the metadata before running an analysis.

You can edit:

- **Use** — include/exclude the sample
- **Sample** — sample name
- **Condition** — biological condition
- **Batch / experiment** — optional nuisance factor

## Excluding a poor-quality sample

Untick **Use** for that sample. The excluded sample will not be available for the differential-expression comparison.

## Batch labels

Only use batch correction if the batch is a real technical/nuisance factor and the experimental design allows it to be separated from condition.

> If every control sample is in Batch 1 and every treated sample is in Batch 2, batch and condition are confounded. Do not enable batch correction in that situation.

---

# 3. Differential expression

BulkRNA Explorer performs pairwise differential expression using PyDESeq2.

## Group A and Group B

- **Group A** = reference/control
- **Group B** = tested condition

The result is always interpreted as **Group B vs Group A**.

### Direction of log2 fold-change

- **Positive log2FC** → higher expression in **Group B**
- **Negative log2FC** → higher expression in **Group A**

### Example

If Group A is `Sensitive` and Group B is `CDDP_R`:

- positive log2FC = higher in `CDDP_R`
- negative log2FC = higher in `Sensitive`

## Replicates

The app requires at least two samples per group. Biological replication is strongly recommended; more replicates generally provide more reliable estimates.

## Advanced settings

### Minimum count

Genes with extremely low counts can be filtered before DE analysis.

### Minimum samples with that count

Controls how many selected samples must meet the minimum-count threshold.

### Correct for batch

Enable only when batch is a genuine nuisance variable and is not confounded with the comparison groups.

---

## PCA

PCA gives an overview of the main expression differences between samples.

Use it to check whether:

- biological replicates cluster together
- conditions separate as expected
- one sample behaves very differently from the others
- a batch effect may dominate the data

PCA is exploratory. A sample should not be removed solely because it looks different without a biological or technical reason.

---

## Volcano plot and DEG table

Choose thresholds for:

- adjusted p-value (`padj`)
- absolute log2 fold-change (`|log2FC|`)

The table can display:

- significant genes only
- all genes
- genes higher in Group B
- genes higher in Group A

### Typical starting thresholds

A common exploratory starting point is:

- `padj ≤ 0.05`
- `|log2FC| ≥ 1`

These are not universal rules. Choose thresholds appropriate for the biological question and analysis plan.

---

## Top-DEG heatmap

The **Top-DEG heatmap** uses expression from the samples in the selected pairwise comparison.

You can display:

- genes UP in Group B
- genes UP in Group A
- both directions, top N total
- both directions, top N per group

You can also choose hierarchical clustering and Z-score scaling per gene.

### Z-score each gene

This emphasizes the relative expression pattern of each gene across the displayed samples. It is useful for visual pattern comparison, but the colour scale no longer represents absolute expression differences between genes.

---

## Common / Venn

Use this panel to compare DEG lists from two or three saved pairwise analyses.

### Example: genes common to two resistant conditions

1. Run `CDDP_R vs Sensitive` and save it.
2. Change Group B to the DTX-resistant samples.
3. Name the group `DTX_R` and run `DTX_R vs Sensitive`.
4. Return to **Common / Venn**.
5. Select both saved comparisons.
6. Choose one of:
   - **UP in Group B of each comparison**
   - **DOWN in Group B / UP in Group A**
   - **All significant DEGs regardless of direction**
7. Choose the desired Venn region.
8. Inspect or download the corresponding gene list.

This is the correct place to obtain **Common UP**, **Common DOWN**, and genes exclusive to one comparison.

---

# 4. Gene heatmaps

This panel is intentionally generic. It is not restricted to a particular cancer type or predefined panel.

Paste any list of **gene symbols**, for example:

```text
ERBB2
ESR1
PGR
MKI67
BCL2
```

You can therefore use:

- a published gene signature
- a pathway gene list
- a custom panel
- a stemness list
- apoptosis genes
- DNA-repair genes
- any other biologically relevant set

The app reports symbols that could not be found in the dataset.

> **Current behaviour:** the custom gene heatmap uses VST expression from one saved pairwise comparison, so the displayed samples are the Group A + Group B samples from that comparison.

---

# 5. GSEA

Standard GSEA is a **pre-ranked GSEA**. All annotated genes are ranked by the PyDESeq2 Wald statistic.

Available collections include:

- Hallmark
- GO Biological Process
- GO Molecular Function
- GO Cellular Component
- Reactome
- KEGG

## How to interpret NES direction

Because the ranking comes from **Group B vs Group A**:

- **positive NES** → enrichment toward Group B
- **negative NES** → enrichment toward Group A

## FDR

Use the GSEA FDR (`FDR q-val`) to assess pathway-level statistical evidence. A lower FDR indicates stronger evidence after correction for multiple testing.

## Important difference from DEG enrichment

GSEA uses the ranked list of **all annotated genes**, not only genes passing a DEG threshold.

---

# 6. Custom GSEA

Custom GSEA provides two modes.

## A. Select specific pathways

Use this when you want to test only a small number of pathways from Hallmark, GO, Reactome, or KEGG.

### Example: search for senescence-related terms

1. Select the saved comparison.
2. Choose **Select specific pathways**.
3. Choose a collection such as **GO Biological Process**.
4. Type `senesc` in the pathway search box.
5. Select the terms you want.
6. Review the number of matched genes.
7. Click **RUN SELECTED-PATHWAY GSEA**.

For very large collections, type a search word first rather than loading every term into the selection box.

## B. My custom gene sets

Use this when you already have your own gene list/signature.

You can:

- paste gene symbols
- paste Ensembl IDs
- upload TXT/CSV/TSV files
- upload several named gene sets in one file

### One custom gene set

Enter a name such as `Senescence`, then paste genes.

### Several custom gene sets

A CSV can use this format:

```csv
pathway,gene
Senescence,CDKN1A
Senescence,CDKN2A
Senescence,TP53
Stemness,SOX2
Stemness,NANOG
```

The app shows matched and unmatched input genes before running GSEA.

### Small gene sets

Very small gene sets can give unstable enrichment results. If only a few genes match the ranked dataset, interpret the result cautiously.

---

# 7. Download

The download section provides files from a selected saved comparison, including:

- normalized counts
- VST expression
- metadata used in the analysis
- annotated differential-expression results
- gene annotation, when available
- ZIP archive containing the main tables

Individual panels also provide dedicated CSV downloads where relevant.

---

# Practical tutorials

## Tutorial A — Compare two conditions

**Goal:** treated vs control.

1. Upload featureCounts files.
2. Check sample metadata.
3. Put controls in Group A.
4. Put treated samples in Group B.
5. Name both groups clearly.
6. Run differential expression.
7. Check PCA.
8. Inspect the volcano plot.
9. Filter the DEG table.
10. Download the DEG table and/or full results ZIP.

---

## Tutorial B — Find genes common to two comparisons

**Goal:** identify genes changing in the same direction in two conditions relative to the same reference.

1. Run comparison 1 and save it.
2. Run comparison 2 and save it.
3. Open **Common / Venn**.
4. Select the two comparisons.
5. Select **UP** to obtain Common UP genes.
6. Select the shared Venn region and download it.
7. Repeat with **DOWN** for Common DOWN genes.

Use the same padj/log2FC thresholds if the biological goal is to compare the two DEG lists consistently.

---

## Tutorial C — Make a heatmap from your own gene list

1. First run a pairwise comparison containing the samples you want to display.
2. Open **4 · Gene heatmaps**.
3. Select the saved comparison.
4. Paste gene symbols, one per line or separated by commas.
5. Choose Z-score and clustering options.
6. Generate the heatmap.
7. Download the underlying expression matrix if needed.

---

## Tutorial D — Run standard GSEA

1. Run and save a differential-expression comparison.
2. Open **5 · GSEA**.
3. Select the comparison.
4. Choose a gene-set collection.
5. Run GSEA.
6. Sort/inspect NES and FDR.
7. Remember: positive NES points toward Group B; negative NES points toward Group A.

---

## Tutorial E — Run GSEA only for pathways of interest

1. Open **6 · Custom GSEA**.
2. Select **Select specific pathways**.
3. Search a collection with a keyword.
4. Select the relevant terms.
5. Confirm that enough genes match.
6. Run the selected-pathway GSEA.
7. Download the results.

---

# Troubleshooting / FAQ

## The app does not find my genes

Check whether you entered gene symbols rather than aliases or uncommon identifiers. The app maps Ensembl IDs during upload, but symbol-based heatmaps expect gene symbols.

## Annotation is unavailable

Gene annotation requires an online annotation service. If that service is temporarily unavailable, differential expression can still work but symbol-dependent functions may not.

## I get no significant DEGs

Possible explanations include:

- small biological effect
- high variability between replicates
- low sample number
- stringent thresholds
- poor data quality
- incorrect sample grouping

Do not simply relax thresholds until something becomes significant. First inspect PCA, sample labels, count quality, and the biological design.

## GSEA returns no pathways

Check:

- gene annotation is available
- enough genes are ranked
- the selected library is reachable online
- custom gene sets contain enough matched genes
- minimum gene-set size is not too high

## Should I use batch correction?

Only when a genuine technical/nuisance batch exists and it is not completely confounded with the biological condition.

## Can I compare one sample vs one sample?

No. BulkRNA Explorer requires at least two samples per group for differential expression, and proper biological replication is strongly recommended.

## Why do positive and negative directions seem reversed?

Always check which condition is Group A and which is Group B. The app reports **Group B vs Group A**.

## Can I use TPM/FPKM?

No. Differential expression requires raw integer counts.

## Does the app save my project forever?

No. Analyses are stored in the current Streamlit browser/server session. Download important result files before ending the session.

---

# Data privacy

BulkRNA Explorer does not intentionally create a database of uploaded counts. However, when the app is hosted on a third-party cloud platform, uploaded research data are processed on that provider's servers.

For unpublished, confidential, patient-derived, or otherwise sensitive datasets, confirm that cloud processing is compatible with your institutional data policy before uploading data.

---

# What this tool is — and is not

BulkRNA Explorer is intended to make common exploratory bulk RNA-seq analyses accessible to non-coding users.

It does **not** replace:

- experimental design review
- sequencing QC
- biological replication
- expert interpretation
- validation of publication-critical findings

When in doubt, preserve the original files and consult the person responsible for the bioinformatics analysis.
