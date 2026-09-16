"""The bounded B controller must never dispatch another experimental arm."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.training.sft.stage3 import run_residual_joint_b as launcher


class ResidualBLauncherTest(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(**{key: Path("/tmp") / key for key in
            ("python", "worktree", "model", "train", "val", "preprocess", "dino", "run_root")})

    def test_exact_scope_and_same_schedule_for_gate(self):
        for phase in launcher.PHASES:
            command = launcher.command(self.args, phase, 29500)
            def value(key):
                return command[command.index("--" + key) + 1]
            for key, expected in {"epochs": "2", "grid-predictor-kind": "residual",
                "lr-qwen-start": "2e-07", "lr-qwen-peak": "2e-07", "query-lr": "1e-05",
                "protocol-lr": "2e-06", "state-proj-lr": "8e-06", "wm-predictor-lr": "0.0003",
                "lambda-sigreg": "0", "lambda-outcome": "0", "lambda-dino": "0.5",
                "batch-size": "1", "grad-accum": "8"}.items():
                self.assertEqual(value(key), expected)
            self.assertNotIn("--outcome-eval-dir", command)
            self.assertEqual("--resume" in command, phase == "resume")
            self.assertEqual("--eval-only" in command, phase == "baseline")
            self.assertEqual("--checkpoint-latest-only" in command, phase == "formal")
            if phase == "formal":
                self.assertNotIn("--stop-after-steps", command)

    def test_default_does_not_run_processes(self):
        argv = []
        for key in ("python", "worktree", "model", "train", "val", "preprocess", "dino", "run-root"):
            argv.extend(["--" + key, "/tmp/" + key])
        argv.extend(["--commit", "approved"])
        output = io.StringIO()
        with patch.object(launcher, "execute") as execute, contextlib.redirect_stdout(output):
            self.assertEqual(launcher.main(argv), 0)
        execute.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())["commands"]), 4)

    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):
            launcher.command(self.args, "treatment", 29500)


if __name__ == "__main__":
    unittest.main()
