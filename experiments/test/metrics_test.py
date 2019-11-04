import unittest

import numpy as np
from ortho import metrics


class MetricsTest(unittest.TestCase):
    def test_rank_1(self):
        rank_1_matrix = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        # Columns are all perfectly similar, so mean should be 1
        self.assertAlmostEqual(metrics.mean_cosine_similarity(rank_1_matrix), 1)

        # Three columns with perfect similarity means total should be 3
        self.assertAlmostEqual(metrics.total_cosine_simiarity(rank_1_matrix), 3)

        # All elements of the similarity matrix should be 1
        cosine_similarity_matrix = metrics.cosine_similarity_matrix(rank_1_matrix)
        for row, col in zip(range(0, 3), range(0, 3)):
            self.assertAlmostEqual(cosine_similarity_matrix[row, col], 1)

    def test_orthogonal(self):
        orthogonal_matrix = np.array([
            [0, 2, 0],
            [1, 0, 0],
            [0, 0, -5],
        ])
        # Columns are orthogonal, so mean should be 0
        self.assertAlmostEqual(metrics.mean_cosine_similarity(orthogonal_matrix), 0)

        # All orthogonal means total should be 0
        self.assertAlmostEqual(metrics.total_cosine_simiarity(orthogonal_matrix), 0)

        # All off-diagonal elements should be 0, on-diagonal are still 1
        cosine_similarity_matrix = metrics.cosine_similarity_matrix(orthogonal_matrix)
        for row, col in zip(range(0, 3), range(0, 3)):
            if row == col:
                self.assertAlmostEqual(cosine_similarity_matrix[row, col], 1)
            else:
                self.assertAlmostEqual(cosine_similarity_matrix[row, col], 0)

    def test_bunched_up(self):
        bunched_up_matrix = np.array([
            [11, 12, 13],
            [11, 12, 14],
            [11, 13, 14],
        ])
        # Columns are close to dependent, so mean should be high
        self.assertGreater(metrics.mean_cosine_similarity(bunched_up_matrix), 0.5)

        # All off-diagonal elements of the similarity matrix should be high
        cosine_similarity_matrix = metrics.cosine_similarity_matrix(bunched_up_matrix)
        for row, col in zip(range(0, 3), range(0, 3)):
            if row == col:
                continue
            self.assertGreater(cosine_similarity_matrix[row, col], 0.5)

    def test_all_equal(self):
        bunched_up_matrix = np.array([
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5],
        ])
        # Columns are exactly dependent, so mean similarity should be maximized (1.0)
        self.assertAlmostEqual(metrics.mean_cosine_similarity(bunched_up_matrix), 1)

        # Same goes for elements of the similarity matrix
        cosine_similarity_matrix = metrics.cosine_similarity_matrix(bunched_up_matrix)
        for row, col in zip(range(0, 3), range(0, 3)):
            self.assertAlmostEqual(cosine_similarity_matrix[row, col], 1)


if __name__ == '__main__':
    unittest.main()
