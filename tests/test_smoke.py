import unittest

import numpy as np
import pandas as pd

from bulkrna.analysis import (
    classify_degs,
    filter_counts,
    run_deseq_comparison,
    select_top_deg_gene_ids,
    significant_gene_set,
)
from bulkrna.overlap import common_to_all, exclusive_to_each


class AnalysisUnitTests(unittest.TestCase):
    def test_filter_counts(self):
        counts = pd.DataFrame(
            {
                "S1": [0, 10, 20],
                "S2": [0, 11, 21],
                "S3": [1, 0, 22],
            },
            index=["g0", "g1", "g2"],
        )
        out = filter_counts(counts, min_count=10, min_samples=2)
        self.assertEqual(list(out.index), ["g1", "g2"])

    def test_deg_classification_and_selection(self):
        res = pd.DataFrame(
            {
                "log2FoldChange": [2.0, -2.5, 0.2, 1.5],
                "padj": [0.001, 0.01, 0.001, 0.2],
            },
            index=["UP1", "DOWN1", "NS1", "NS2"],
        )
        status = classify_degs(res, padj_cut=0.05, lfc_cut=1.0)
        self.assertEqual(status.loc["UP1"], "UP")
        self.assertEqual(status.loc["DOWN1"], "DOWN")
        self.assertEqual(status.loc["NS1"], "NS")
        self.assertEqual(significant_gene_set(res, 0.05, 1.0, "up"), {"UP1"})
        self.assertEqual(significant_gene_set(res, 0.05, 1.0, "down"), {"DOWN1"})
        both = select_top_deg_gene_ids(res, n=10, padj_cut=0.05, lfc_cut=1.0, direction="both_total")
        self.assertEqual(set(both), {"UP1", "DOWN1"})

    def test_overlap_helpers(self):
        sets = {
            "A": {"g1", "g2", "g3"},
            "B": {"g2", "g3", "g4"},
            "C": {"g3", "g5"},
        }
        self.assertEqual(common_to_all(sets), {"g3"})
        exclusive = exclusive_to_each(sets)
        self.assertEqual(exclusive["A"], {"g1"})
        self.assertEqual(exclusive["B"], {"g4"})
        self.assertEqual(exclusive["C"], {"g5"})


class ScientificPipelineSmokeTest(unittest.TestCase):
    def test_pydeseq2_pipeline_runs_on_small_dataset(self):
        rng = np.random.default_rng(7)
        genes = [f"gene_{i:03d}" for i in range(60)]
        samples_a = ["A1", "A2", "A3"]
        samples_b = ["B1", "B2", "B3"]
        samples = samples_a + samples_b

        matrix = np.zeros((len(genes), len(samples)), dtype=int)
        for i in range(len(genes)):
            base = 40 + (i % 15) * 3
            matrix[i, :3] = rng.poisson(base, size=3)
            if i < 12:
                matrix[i, 3:] = rng.poisson(base * 4, size=3)
            elif 12 <= i < 24:
                matrix[i, 3:] = rng.poisson(max(5, base // 4), size=3)
            else:
                matrix[i, 3:] = rng.poisson(base, size=3)

        counts = pd.DataFrame(matrix, index=genes, columns=samples)
        metadata = pd.DataFrame(
            {
                "include": [True] * 6,
                "sample": samples,
                "condition": ["A"] * 3 + ["B"] * 3,
                "batch": ["B1", "B2", "B3", "B1", "B2", "B3"],
            }
        )

        bundle = run_deseq_comparison(
            counts,
            metadata,
            group_a_samples=samples_a,
            group_b_samples=samples_b,
            group_a_name="Control",
            group_b_name="Test",
            use_batch=False,
            min_count=1,
            min_samples=2,
            alpha=0.05,
            n_cpus=1,
        )

        self.assertEqual(bundle.comparison_name, "Test vs Control")
        self.assertEqual(bundle.dds, None)
        self.assertEqual(bundle.norm_counts.shape[1], 6)
        self.assertEqual(bundle.vst_counts.shape[1], 6)
        result = bundle.results["Test vs Control"]
        for col in ["baseMean", "log2FoldChange", "pvalue", "padj"]:
            self.assertIn(col, result.columns)
        self.assertGreaterEqual(result.shape[0], 50)


class ImportSmokeTests(unittest.TestCase):
    def test_major_modules_import(self):
        import bulkrna.app  # noqa: F401
        import bulkrna.annotation  # noqa: F401
        import bulkrna.custom_gsea  # noqa: F401
        import bulkrna.enrichment  # noqa: F401
        import bulkrna.io  # noqa: F401
        import bulkrna.plots  # noqa: F401


if __name__ == "__main__":
    unittest.main()
