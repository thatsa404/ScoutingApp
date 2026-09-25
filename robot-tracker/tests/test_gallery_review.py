import unittest

from rtrack.gallery_review import (
    _candidate_images,
    _candidate_id,
    _choose_views,
    _solver_proposals,
    manifest_version,
    _unreviewed_candidates,
    validate_answer,
)


class GalleryReviewTests(unittest.TestCase):
    def test_solver_proposals_join_exact_frame_and_source_track(self):
        rows = [
            {"f": 10, "dets": [{"tid": 101, "source_tid": 7, "team": "190"}]},
            {"f": 11, "dets": [{"tid": 102, "source_tid": 7, "team": "157"}]},
            {"f": 12, "dets": [{"tid": 103, "source_tid": 8, "team": "190"},
                                  {"tid": 104, "source_tid": 8, "team": "157"}]},
        ]
        proposals = _solver_proposals(rows)
        self.assertEqual(proposals[(10, 7)], "190")
        self.assertEqual(proposals[(11, 7)], "157")
        self.assertNotIn((12, 8), proposals)

    def test_view_selection_is_bounded_and_temporally_spread(self):
        views = [{"f": i, "t": float(i), "xyxy": [0, 0, 40, 60]}
                 for i in range(20)]
        chosen = _choose_views(views, 5)
        self.assertEqual(len(chosen), 5)
        self.assertEqual([v["f"] for v in chosen], [0, 4, 9, 14, 19])

    def test_manifest_version_is_order_independent_for_decisions(self):
        a = {"season": 2026, "teamStates": {}, "decisions": [
            {"candidateId": "b", "action": "accept", "team": "190"},
            {"candidateId": "a", "action": "reject"},
        ]}
        b = {**a, "decisions": list(reversed(a["decisions"]))}
        self.assertEqual(manifest_version(a), manifest_version(b))

    def test_answer_must_match_bundle_and_views(self):
        bundle = {"kind": "galleryReviewBundle", "schemaVersion": 2,
                  "reviewId": "r", "bundleHash": "b", "teams": [{
                      "team": "190", "currentGallery": [], "candidates": [{
                          "candidateId": "c", "cropHash": "v",
                      }],
                  }]}
        validate_answer({"kind": "galleryReviewAnswer", "schemaVersion": 2,
                         "reviewId": "r", "bundleHash": "b", "selections": [{
                             "team": "190", "include": ["v"],
                         }]}, bundle)
        with self.assertRaises(SystemExit):
            validate_answer({"kind": "galleryReviewAnswer", "schemaVersion": 2,
                             "reviewId": "r", "bundleHash": "b", "selections": [{
                                 "team": "190", "include": ["not-in-bundle"],
                             }]}, bundle)

    def test_candidate_selection_starts_with_time_endpoints(self):
        items = [{"tid": 1, "f": 1, "t": 1.0, "boxArea": 100,
                  "boxWidth": 10, "boxHeight": 10},
                 {"tid": 1, "f": 2, "t": 2.0, "boxArea": 900,
                  "boxWidth": 30, "boxHeight": 30},
                 {"tid": 2, "f": 3, "t": 3.0, "boxArea": 200,
                  "boxWidth": 14, "boxHeight": 14}]
        selected = _candidate_images(items, 2)
        self.assertEqual({item["tid"] for item in selected}, {1, 2})
        self.assertEqual(selected[0]["tid"], 1)

    def test_field_coverage_beats_detection_size(self):
        items = [
            {"tid": 1, "f": 1, "t": 0.0, "xyxy": [0, 0, 10, 10],
             "fieldXY": [0.0, 0.0], "boxArea": 10},
            {"tid": 2, "f": 2, "t": 10.0, "xyxy": [0, 0, 10, 10],
             "fieldXY": [1.0, 1.0], "boxArea": 10},
            {"tid": 3, "f": 3, "t": 5.0, "xyxy": [0, 0, 10, 10],
             "fieldXY": [0.0, 1.0], "boxArea": 10},
            {"tid": 4, "f": 4, "t": 5.0, "xyxy": [0, 0, 10, 10],
             "fieldXY": [0.5, 0.5], "boxArea": 9000},
        ]
        selected = _candidate_images(items, 3)
        self.assertIn(3, {item["tid"] for item in selected})

    def test_reviewed_candidates_are_not_reissued(self):
        items = [
            {"tid": 1, "f": 10, "t": 1.0, "boxArea": 100},
            {"tid": 2, "f": 20, "t": 2.0, "boxArea": 100},
        ]
        reviewed = {_candidate_id(items[0], season=2026, match="m",
                                  team="190", tracks_source="tracks",
                                  appearance_source="appearance")}
        remaining = _unreviewed_candidates(
            items, reviewed, season=2026, match="m", team="190",
            tracks_source="tracks", appearance_source="appearance")
        self.assertEqual([item["tid"] for item in remaining], [2])


if __name__ == "__main__":
    unittest.main()
