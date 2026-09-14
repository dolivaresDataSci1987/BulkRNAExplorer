# BulkRNA Explorer — Streamlit edition

A click-first bulk RNA-seq app for collaborators who should not need Python, R, Galaxy or local software installation.

## Current features

- Upload multiple Galaxy/featureCounts tables.
- Auto-detect sample IDs, conditions and Exp/batch labels when present in filenames.
- Edit sample metadata in the browser.
- Select **exactly which samples** belong to Group A and Group B for each differential-expression comparison.
- Run PyDESeq2 on raw integer counts.
- Calculate DESeq2 normalized counts and VST expression.
- PCA for the samples included in each comparison.
- Ensembl ID → gene symbol / gene name annotation for human or mouse.
- Volcano plots showing gene symbols and names.
- Annotated differential-expression tables.
- Top-DEG heatmaps with any requested number of genes.
- Custom gene-panel heatmaps entered by **gene symbol**.
- Save multiple comparisons during the same session and calculate Common UP / Common DOWN genes.
- Pre-ranked GSEA using the PyDESeq2 Wald statistic with Hallmark, GO, Reactome or KEGG gene sets.
- Download normalized counts, VST matrices, annotated DEG tables, gene annotation and results ZIPs.

## Deploy on Streamlit Community Cloud

1. Connect this repository to Streamlit Community Cloud.
2. Entrypoint: `streamlit_app.py`.
3. Use Python 3.12.
4. Deploy and share the resulting Streamlit URL.

No Python installation is required on collaborators' computers.

## Data privacy

The application code does not create a database or intentionally persist uploaded counts. However, when deployed on a third-party cloud, uploaded research data are processed on that provider's server. For unpublished or sensitive datasets, confirm that this is acceptable under your institution's data policy or deploy the same repository on an institutional server.

## Scientific notes

- Differential expression requires **raw integer counts**, not TPM/FPKM/CPM.
- Paired-end R1/R2 FASTQ files are not biological replicates.
- A differential-expression comparison should normally contain biological replicates in both groups; the app requires at least two samples per group.
- Positive log2 fold-change means higher expression in **Group B relative to Group A**.
- Only enable batch correction when the batch variable is real and the design is identifiable.
- GSEA ranks all annotated genes by the PyDESeq2 Wald statistic; it does not restrict the analysis to statistically significant DEGs.
- Gene annotation and online Enrichr gene-set libraries require outbound internet access from the Streamlit server.
- This application is intended for exploratory analysis and should be validated against an established pipeline before publication.

## Planned next additions

- Built-in PAM50, BCL2-family and curated stemness panels.
- GO over-representation analysis directly from Common UP / Common DOWN genes.
- Publication-quality SVG/PDF export.
- Project save/reopen.
