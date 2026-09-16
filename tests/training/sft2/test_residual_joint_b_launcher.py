"""The bounded B controller must never dispatch another experimental arm."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
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

    def test_combined_variant_changes_only_authorized_controls(self):
        for phase in launcher.PHASES:
            self.args.variant = "b"
            original = launcher.command(self.args, phase, 29500)
            self.args.variant = "dino2_backbone_stopgrad"
            combined = launcher.command(self.args, phase, 29500)
            expected = original.copy()
            expected[expected.index("--lambda-dino") + 1] = "2.0"
            expected[expected.index("--wm-value-backbone-grad")] = "--no-wm-value-backbone-grad"
            name_index = expected.index("--wandb-run-name") + 1
            expected[name_index] = combined[name_index]
            self.assertEqual(combined, expected)

    def test_variant_checkpoint_mismatch_stops_controller(self):
        self.args.variant = "dino2_backbone_stopgrad"
        metadata = {"training_invariants": {"grid_predictor_kind": "residual", "lambda_sigreg": 0,
                    "dino_weight": 2., "wm_value_backbone_grad": False}, "has_optimizer": True}
        with patch.object(launcher.common, "verify_phase", return_value="/tmp/stop_step_000001"), \
             patch.object(launcher.common, "checkpoint_metadata", return_value=metadata):
            self.assertIn("checkpoint", launcher.verify(self.args, "canary", Path("unused")))
            for key, wrong in (("dino_weight", .5), ("wm_value_backbone_grad", True)):
                correct = metadata["training_invariants"][key]
                metadata["training_invariants"][key] = wrong
                with self.assertRaisesRegex(RuntimeError, "variant checkpoint identity"):
                    launcher.verify(self.args, "canary", Path("unused"))
                metadata["training_invariants"][key] = correct

    def test_combined_parser_and_unchanged_disk_guard(self):
        argv = []
        for key in ("python", "worktree", "model", "train", "val", "preprocess", "dino", "run-root"):
            argv.extend(["--" + key, "/tmp/" + key])
        argv.extend(["--commit", "approved", "--variant", "dino2_backbone_stopgrad"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            launcher.main(argv)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["variant_invariants"], {"dino_weight": 2., "wm_value_backbone_grad": False})
        self.assertEqual(len(plan["commands"]), 4)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.main(argv + ["--min-free-gib", "123"])

    def test_existing_run_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.run_root = Path(directory)
            with self.assertRaises(FileExistsError):
                launcher.execute(self.args)


if __name__ == "__main__":
    unittest.main()
