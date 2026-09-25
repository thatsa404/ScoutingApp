import unittest
from unittest.mock import patch

import numpy as np

from rtrack.appear import appearance_crops
from rtrack.embed import embed_appearance
from rtrack.reid import _gate_team_scores, _pooled_queries, _team_scores
from rtrack.robots import normalize_counter, retally


class AppearancePolicyTests(unittest.TestCase):
    def test_full_crop_neutralizes_bumper_hue_but_keeps_upper_crop(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[10:54, 10:90] = (10, 80, 160)
        image[54:90, 10:90] = (255, 0, 0)
        upper, whole = appearance_crops(image, [10, 10, 90, 90])
        self.assertGreater(upper.size, 0)
        self.assertTrue(np.any(upper[..., 2] != upper[..., 0]))
        bumper = whole[int(0.55 * len(whole)):]
        self.assertTrue(np.array_equal(bumper[..., 0], bumper[..., 1]))
        self.assertTrue(np.array_equal(bumper[..., 1], bumper[..., 2]))

    def test_fusion_is_weighted_concatenation_of_normalized_branches(self):
        fake = [np.array([[3.0, 4.0]], np.float32),
                np.array([[0.0, 2.0]], np.float32)]
        with patch("rtrack.embed.embed", side_effect=fake):
            fused = embed_appearance([np.zeros((6, 6, 3), np.uint8)],
                                     [np.zeros((6, 6, 3), np.uint8)])
        want = np.sqrt(0.5) * np.array([0.6, 0.8, 0.0, 1.0])
        np.testing.assert_allclose(fused[0], want, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(fused[0])), 1.0, places=6)

    def test_pooling_uses_only_local_samples_on_one_track(self):
        features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
        pooled = _pooled_queries(features, np.array([0.0, 1.0, 2.0]),
                                 np.array([1]), radius=1.0)
        want = np.array([[2.0, 1.0]], np.float32)
        want /= np.linalg.norm(want, axis=1, keepdims=True)
        np.testing.assert_allclose(pooled, want, atol=1e-6)

    def test_top_three_mean_resists_one_lucky_prototype(self):
        similarities = np.array([[0.99, 0.10, 0.10, 0.10,
                                  0.80, 0.79, 0.78]], np.float32)
        scores = _team_scores(similarities,
                              ["A", "A", "A", "A", "B", "B", "B"],
                              ["A", "B"], top=3)
        self.assertGreater(scores[0, 1], scores[0, 0])

    def test_alliance_gate_has_low_confidence_fallback(self):
        scores = np.array([[0.1, 0.9], [0.1, 0.9]], np.float32)
        gated = _gate_team_scores(
            scores, np.array(["red", "red"]), np.array([1.0, 0.2]),
            ["R", "B"], {"R": "red", "B": "blue"}, minimum_confidence=0.5)
        self.assertTrue(np.isneginf(gated[0, 1]))
        self.assertAlmostEqual(float(gated[1, 1]), 0.9, places=6)

    def test_dense_votes_are_redistributed_then_capped_per_fragment(self):
        votes = [[float(t), "A" if t != 15 else "B"] for t in range(20)]
        ident = {
            "tallyVoteCap": 4,
            "tracks": {"1": {"voteList": votes, "tally": {"A": 19, "B": 1}}},
        }
        rows = [
            {"t": float(t), "dets": [{"tid": 1 if t < 10 else 2}]}
            for t in range(20)
        ]

        result = retally(ident, rows, {1: 1, 2: 1})

        self.assertEqual(len(result["tracks"]["1"]["voteList"]), 10)
        self.assertEqual(len(result["tracks"]["2"]["voteList"]), 10)
        self.assertEqual(result["tracks"]["1"]["votes"], 4)
        self.assertEqual(result["tracks"]["2"]["votes"], 4)
        self.assertEqual(sum(result["tracks"]["1"]["tally"].values()), 4)
        self.assertEqual(sum(result["tracks"]["2"]["tally"].values()), 4)

    def test_vote_normalization_preserves_share_and_exact_budget(self):
        result = normalize_counter({"A": 3, "B": 1}, 24)
        self.assertEqual(result, {"A": 18, "B": 6})
        self.assertEqual(sum(result.values()), 24)


if __name__ == "__main__":
    unittest.main()
