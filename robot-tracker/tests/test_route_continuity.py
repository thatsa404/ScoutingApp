import unittest

from rtrack.continuity import candidates


class ContinuityCandidateTests(unittest.TestCase):
    def test_strict_policy_requires_close_geometry_and_matching_hints(self):
        info = {
            1: {"t0": 0.0, "t1": 1.0, "start": (0.0, 0.0),
                "end": (0.0, 0.0), "alliance": "red"},
            2: {"t0": 1.1, "t1": 2.0, "start": (0.05, 0.0),
                "end": (0.1, 0.0), "alliance": "red"},
            3: {"t0": 1.1, "t1": 2.0, "start": (0.06, 0.0),
                "end": (0.1, 0.0), "alliance": "red"},
            4: {"t0": 1.1, "t1": 2.0, "start": (3.0, 0.0),
                "end": (3.1, 0.0), "alliance": "red"},
        }
        edges = candidates(
            info, {}, max_score=0.25,
            team_hints={1: "A", 2: "A", 3: "B", 4: "A"},
            require_same_hint=True)

        self.assertEqual([(edge["a"], edge["b"]) for edge in edges], [(1, 2)])


if __name__ == "__main__":
    unittest.main()
