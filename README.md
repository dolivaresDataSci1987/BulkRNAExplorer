# BulkRNA Explorer — Streamlit edition

A click-first bulk RNA-seq app for collaborators who should not need Python, R, Galaxy or local software installation.

## What this version does

- Upload multiple Galaxy/featureCounts tables.
- Auto-detect sample IDs, Sensitive / CDDP_R / DTX_R and Exp2/3/4 when present in filenames.
- Edit sample metadata in the browser.
- Filter low-count genes.
- Run PyDESeq2 on raw integer counts.
- Calculate DESeq2 normalized counts and VST expression.
- PCA.
- Differential-expression tables and volcano plots.
- Automatic contrasts of every condition versus a selected reference.
- Common UP / common DOWN genes across contrasts.
- Gene-panel heatmaps using Ensembl IDs.
- Download normalized counts, VST matrices, DEG tables and a results ZIP.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository, e.g. `BulkRNAExplorer`.
2. Upload **all files and folders from this ZIP** to the repository root.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** and choose the repository.
5. Entrypoint: `streamlit_app.py`.
6. Open **Advanced settings** and select **Python 3.12**.
7. Deploy.
8. Share the resulting `https://....streamlit.app` URL with collaborators.

No Python installation is required on collaborators' computers.

## Data privacy

The application code does not create a database or intentionally persist uploaded counts. However, if you deploy on a third-party cloud, uploaded research data are processed on that provider's server. For unpublished/sensitive datasets, confirm that this is acceptable under your institution's data policy or deploy the same repository on an institutional server.

## Important scientific notes

- Differential expression requires **raw integer counts**, not TPM/FPKM/CPM.
- Paired-end R1/R2 FASTQ files are not biological replicates. featureCounts should yield one count profile per biological sample/library.
- Only enable batch correction when the `batch` field represents a genuine nuisance factor and the design is identifiable.
- This prototype is for exploratory analysis and should be validated against your established pipeline before publication.

## Next planned functions

- Gene symbol ↔ Ensembl annotation bundled for human/mouse.
- Built-in PAM50 and BCL2-family panels.
- Custom panel upload by gene symbol.
- GO over-representation analysis for Common UP/DOWN.
- Pre-ranked GSEA.
- Publication-quality figure export (SVG/PDF).
- Project save/reopen.
