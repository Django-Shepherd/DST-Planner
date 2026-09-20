"""Corrupt real captured episodes and ensure damaged samples are rejected."""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_episode import validate, validate_decision


class CapturedEpisodeValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(sys.argv) < 2:
            raise unittest.SkipTest("Pass a captured episode directory")
        cls.source = Path(sys.argv[1])
        cls.report = validate(cls.source)
        with (cls.source / "decisions.jsonl").open() as stream:
            cls.decisions = []
            cls.action = None
            for line in stream:
                d = json.loads(line)
                if len(cls.decisions) < 2:
                    cls.decisions.append(d)
                if cls.action is None and d["candidates"]:
                    cls.action = d
                if len(cls.decisions) == 2 and cls.action is not None:
                    break
        if cls.action is None:
            raise AssertionError("No actionable decision in test episode")

    def test_wrong_expert_label(self):
        d = copy.deepcopy(self.action)
        d["selected_candidate_id"] = len(d["candidates"])
        with self.assertRaisesRegex(ValueError, "Invalid selected candidate"):
            validate_decision(d, d["decision_id"])

    def test_foreign_candidate_node(self):
        d = copy.deepcopy(self.action)
        d["candidates"][0]["node_id"] = len(d["nodes"])
        with self.assertRaisesRegex(ValueError, "Invalid candidate node"):
            validate_decision(d, d["decision_id"])

    def test_nonfinite_state(self):
        d = copy.deepcopy(self.action)
        d["position"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid position"):
            validate_decision(d, d["decision_id"])

    def test_dropped_decision(self):
        self.assertGreater(len(self.decisions), 1)
        d = self.decisions[1]
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            validate_decision(d, 0)

    def test_unflushed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "metadata.json",
                "summary.json",
                "events.jsonl",
                "decisions.jsonl",
                "decisions.jsonl.status.json",
            ]:
                shutil.copyfile(self.source / name, root / name)
            status = json.loads((root / "decisions.jsonl.status.json").read_text())
            status["written"] -= 1
            (root / "decisions.jsonl.status.json").write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "Missing/unflushed"):
                validate(root)

    def test_timeout_not_completion(self):
        if self.report["reason"] != "time_limit":
            self.skipTest("This episode completed naturally")
        with self.assertRaisesRegex(ValueError, "Truncated episode"):
            validate(self.source, require_complete=True)


if __name__ == "__main__":
    # Retain the episode argument for setUpClass but keep unittest from interpreting it.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
