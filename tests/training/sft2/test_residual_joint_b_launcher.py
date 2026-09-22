"""The bounded B controller must never dispatch another experimental arm."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.training.sft.stage3 import run_residual_joint_b as launcher


class ResidualBLauncherTest(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(**{key: Path("/tmp") / key for key in
            ("python", "worktree", "model", "train", "val", "preprocess", "dino", "run_root")})

    def test_exact_scope_and_same_schedule_for_gate(self):
        for phase in launcher.PHASES:
            command = launcher.command(self.args, phase, 29500)
            def value(key, command=command):
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

    def test_outcome_fresh_budget_and_guard(self):
        argv = self.continuation_argv()[:18] + ["--variant", "outcome"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            launcher.main(argv)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["min_free_gib"], 109)
        self.assertEqual(len(plan["commands"]), 1)
        command = plan["commands"][0]
        for key, value in (("epochs", "5"), ("schedule-total-steps", "46"),
                           ("lambda-outcome", "1"), ("lambda-dino", "2.0")):
            self.assertEqual(command[command.index("--" + key) + 1], value)
        for forbidden in ("--resume", "--eval-only", "--early-stop-metric", "--stop-after-steps"):
            self.assertNotIn(forbidden, command)
        self.assertIn("--no-wm-value-backbone-grad", command)
        self.assertIn("--checkpoint-latest-only", command)
        start = command.index("--diagnostic-steps") + 1
        self.assertEqual(command[start:start+6], list(map(str, range(0, 116, 23))))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.main(argv + ["--min-free-gib", "108"])

    def test_outcome_completion_requires_head_and_full_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.run_root = Path(directory)
            self.args.variant = "outcome"
            root = self.args.run_root / "formal"
            checkpoint = root / "epoch_005"
            checkpoint.mkdir(parents=True)
            for name in ("training_state.pt", "state_proj.pt", "selected_token_rows.pt", "vision_ema.pt",
                         "wm_predictor/predictor.pt", "value_head/value_head.pt", "outcome_head.pt"):
                target = checkpoint / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(b"fixture")
            complete = {"epoch": 5, "step": 115, "checkpoint": "epoch_005",
                        "schedule_total_steps": 46, "reason": "epoch_limit", "early_stop_state": None}
            (root / "training_complete.json").write_text(json.dumps(complete))
            metrics = {"outcome_unique_transition_count": 4}
            for prefix in ("outcome_all", "outcome_h1", "outcome_h2", "outcome_h3", "outcome_h4"):
                for key in ("count", "success_count", "failure_count", "failure_tp", "failure_fp",
                            "failure_fn", "failure_tn", "bce", "accuracy", "always_success_accuracy"):
                    metrics[f"{prefix}_{key}"] = 1
                for key in ("failure_precision", "failure_recall", "failure_f1"):
                    metrics[f"{prefix}_{key}_defined"] = 0
            log = root / "formal.log"
            log.write_text("\n".join(json.dumps({"epoch": e, "global_step": e*23, "val_metrics": metrics}) for e in range(1, 6)))
            self.assertEqual(launcher.verify(self.args, "formal", log)["checkpoint"], str(checkpoint))
            (checkpoint / "outcome_head.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete outcome checkpoint"):
                launcher.verify(self.args, "formal", log)

    def test_existing_run_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.run_root = Path(directory)
            with self.assertRaises(FileExistsError):
                launcher.execute(self.args)

    def continuation_argv(self):
        argv = []
        for key in ("python", "worktree", "model", "train", "val", "preprocess", "dino", "run-root"):
            argv.extend(["--" + key, "/tmp/" + key])
        return argv + ["--commit", "approved", "--variant", "dino2_backbone_stopgrad",
                       "--continue-from", str(launcher.CONTINUE_SOURCE), "--additional-epochs", "10",
                       "--early-stop-metric", "wm_mse", "--early-stop-baseline", "0.235"]

    def test_continuation_is_single_resume_preserving_schedule(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            launcher.main(self.continuation_argv())
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["min_free_gib"], 120)
        self.assertEqual(len(plan["commands"]), 1)
        command = plan["commands"][0]
        for key, value in (("epochs", "12"), ("resume-from", str(launcher.CONTINUE_SOURCE)),
                           ("schedule-total-steps", "46"), ("early-stop-patience", "2")):
            self.assertEqual(command[command.index("--" + key) + 1], value)
        self.assertIn("--resume", command)
        self.assertNotIn("--eval-only", command)
        self.assertNotIn("--stop-after-steps", command)
        index = command.index("--diagnostic-steps") + 1
        self.assertEqual(command[index:index+11], list(map(str, range(46, 277, 23))))

    def test_continuation_requires_explicit_approved_scope(self):
        for extra in (["--additional-epochs", "11"], ["--variant", "b"], ["--min-free-gib", "119"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.main(self.continuation_argv() + extra)

    def test_verify_early_stop_uses_actual_epoch_not_epoch12(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.run_root = Path(directory)
            self.args.continue_from = launcher.CONTINUE_SOURCE
            self.args.early_stop_metric = "wm_mse"
            self.args.early_stop_baseline = .24
            root = self.args.run_root / "formal"
            checkpoint = root / "epoch_004"
            checkpoint.mkdir(parents=True)
            for name in ("training_state.pt", "state_proj.pt", "selected_token_rows.pt",
                         "wm_predictor/predictor.pt", "value_head/value_head.pt"):
                target = checkpoint / name
                target.parent.mkdir(exist_ok=True)
                target.touch()
            complete = {"epoch": 4, "step": 92, "checkpoint": "epoch_004", "schedule_total_steps": 46, "reason": "early_stop",
                        "early_stop_state": {"metric": "wm_mse", "patience": 2,
                                             "relative_improvement": .01, "bad_epochs": 2, "previous": .24,
                                             "history": [{"epoch": e, "value": .24, "relative_improvement": 0.} for e in (3, 4)]}}
            (root / "training_complete.json").write_text(json.dumps(complete))
            log = root / "formal.log"
            log.write_text("\n".join(json.dumps({"epoch": e, "global_step": e*23, "val_metrics": {"wm_mse": .24}}) for e in (3, 4)))
            self.assertEqual(launcher.verify(self.args, "formal", log)["checkpoint"], str(checkpoint))
            complete["early_stop_state"]["bad_epochs"] = 1
            (root / "training_complete.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(RuntimeError, "premature convergence"):
                launcher.verify(self.args, "formal", log)


if __name__ == "__main__":
    unittest.main()
