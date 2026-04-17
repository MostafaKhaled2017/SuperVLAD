from __future__ import annotations

import unittest
from types import SimpleNamespace

import faiss
import numpy as np

import test


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_populate_retrieval_diagnostics_batches_full_search(self):
        original_buffer_bytes = test.RETRIEVAL_DIAGNOSTICS_MAX_BUFFER_BYTES
        test.RETRIEVAL_DIAGNOSTICS_MAX_BUFFER_BYTES = 64
        self.addCleanup(
            setattr,
            test,
            "RETRIEVAL_DIAGNOSTICS_MAX_BUFFER_BYTES",
            original_buffer_bytes,
        )

        database_features = np.array(
            [
                [0.0, 0.0],
                [5.0, 5.0],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        )
        query_features = np.array(
            [
                [0.1, 0.1],
                [4.9, 5.2],
                [9.5, 9.5],
            ],
            dtype=np.float32,
        )

        index = faiss.IndexFlatL2(database_features.shape[1])
        index.add(database_features)

        eval_ds = SimpleNamespace(
            database_num=database_features.shape[0],
            retrieval_query_diagnostics=None,
            retrieval_per_bin_metrics=None,
            retrieval_query_debug=None,
            retrieval_cluster_mass_stats=None,
            retrieval_diagnostics_meta=None,
            get_positives=lambda: [
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ],
        )
        args = SimpleNamespace(
            return_debug_metrics=False,
            attack_name="none",
            attack_severity=None,
            attack_seed=0,
            attack_eps=None,
            attack_steps=None,
            attack_step_size=None,
            attack_mask_mode="none",
            attack_keep_ratio=None,
        )

        test._populate_retrieval_diagnostics(
            args=args,
            eval_ds=eval_ds,
            faiss_index=index,
            query_features=query_features,
            test_method="hard_resize",
            query_debug_rows=None,
            cluster_mass_rows=None,
        )

        self.assertEqual(len(eval_ds.retrieval_query_diagnostics), 3)
        self.assertEqual(
            [row["best_positive_rank"] for row in eval_ds.retrieval_query_diagnostics],
            [1, 1, 1],
        )
        self.assertTrue(all(row["top1_correct"] for row in eval_ds.retrieval_query_diagnostics))
        self.assertEqual(eval_ds.retrieval_per_bin_metrics[0]["query_count"], 3)


if __name__ == "__main__":
    unittest.main()
