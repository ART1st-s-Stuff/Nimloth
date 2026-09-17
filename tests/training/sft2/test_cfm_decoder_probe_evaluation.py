import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.training.sft.stage3.evaluate_cfm_decoder_probe import (
    aligned_rows, image_metrics, paired_noise, validate_decoder_identity,
)
from experiments.training.sft.stage3.render_continuation_features import (
    column_layout, load_probe, parse_probe_specs, validate_manifests,
)


class CFMProbeEvaluationTests(unittest.TestCase):
    def test_noise_is_identity_paired_not_order_or_column_dependent(self):
        keys = [("a", 3), ("b", 4), ("a", 3)]
        noise = paired_noise(keys, 20260931, size=12)
        self.assertTrue(torch.equal(noise[0], noise[2]))
        self.assertTrue(torch.equal(noise[1], paired_noise([keys[1]], 20260931, size=12)[0]))
        self.assertFalse(torch.equal(noise[0], paired_noise([keys[0]], 20260932, size=12)[0]))

    def test_real_gaussian_ssim_identity_and_error(self):
        target = torch.rand(2, 3, 16, 16)*2-1
        same = image_metrics(target, target)
        self.assertTrue(torch.equal(same["mse"], torch.zeros(2)))
        self.assertTrue(torch.allclose(same["ssim"], torch.ones(2), atol=1e-5))
        changed = image_metrics(-target, target)
        self.assertTrue(torch.all(changed["mse"] > 0))
        self.assertTrue(torch.all(changed["ssim"] < same["ssim"]))

    def test_cache_alignment_rejects_action_or_successor_change(self):
        dino = torch.arange(5.).reshape(5, 1, 1).expand(5, 64, 1024).clone()
        cached = {"a": {"dino": dino, "states": dino+1, "actions": torch.tensor([1, 2, 3, 4])}}
        item = {"actions": torch.tensor([1., 2., 3., 4.]), "current_dino": dino[0], "dino": dino[1:],
                "online_direct": dino[1:]+2, "predicted": dino[1:]+3, "online_current": dino[0]+2}
        probes = {label: {("a", 0): copy.deepcopy(item)} for label in ("old_stage3_e5", "new_outcome_e5")}
        rows = aligned_rows(probes, cached)
        self.assertEqual([row["observation"] for row in rows], [1, 2, 3, 4])
        self.assertTrue(torch.equal(rows[0]["gt_state"], cached["a"]["states"][1]))
        probes["new_outcome_e5"][("a", 0)]["actions"][0] = 0
        with self.assertRaisesRegex(ValueError, "actions mismatch"):
            aligned_rows(probes, cached)
        probes["new_outcome_e5"] = copy.deepcopy(probes["old_stage3_e5"])
        cached["a"]["dino"] = dino+5
        with self.assertRaisesRegex(ValueError, "current DINO"):
            aligned_rows(probes, cached)

    def test_named_probe_specs_preserve_order_and_reject_ambiguity(self):
        probes = parse_probe_specs(["old_stage3_e5=/old", "new_outcome_e5=/new"])
        self.assertEqual(list(probes), ["old_stage3_e5", "new_outcome_e5"])
        self.assertEqual(probes["new_outcome_e5"].as_posix(), "/new")
        for invalid in ("missing_path", "bad label=/tmp", "same=/a"):
            specs = [invalid] if invalid != "same=/a" else [invalid, "same=/b"]
            with self.assertRaises(ValueError):
                parse_probe_specs(specs)

    def test_named_probe_manifests_allow_same_arbitrary_step_and_bind_fixed_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {label: root / label for label in ("old_stage3_e5", "new_outcome_e5")}
            for label, directory in paths.items():
                directory.mkdir()
                for rank in range(8):
                    artifact = directory / f"rank_{rank:03d}_batch_000.pt"
                    artifact.write_bytes(f"{label}-{rank}".encode())
                    manifest = {
                        "schema": "stage3_fixed_batch_probe_v1", "rank": rank, "step": 115,
                        "batches": [{"keys": [[f"trajectory-{rank}", 0]]}], "identity": {"run": label},
                        "files": [{"name": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
                    }
                    (directory / f"rank_{rank:03d}_COMPLETE.json").write_text(json.dumps(manifest))
            result = validate_manifests(paths)
            self.assertEqual(result["new_outcome_e5"]["step"], 115)
            changed = paths["new_outcome_e5"] / "rank_003_COMPLETE.json"
            manifest = json.loads(changed.read_text())
            manifest["batches"][0]["keys"][0][1] = 1
            changed.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fixed input batch mismatch"):
                validate_manifests(paths)

    def test_probe_manifests_reject_mixed_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(8):
                artifact = directory / f"rank_{rank:03d}_batch_000.pt"
                artifact.write_bytes(str(rank).encode())
                manifest = {
                    "schema": "stage3_fixed_batch_probe_v1", "rank": rank, "step": 115,
                    "batches": [{"keys": [[f"trajectory-{rank}", 0]]}],
                    "identity": {"run": "wrong" if rank == 4 else "one"},
                    "files": [{"name": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
                }
                (directory / f"rank_{rank:03d}_COMPLETE.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "rank identities differ"):
                validate_manifests({"one": directory})

    def test_probe_manifest_rejects_unlisted_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(8):
                artifact = directory / f"rank_{rank:03d}_batch_000.pt"
                artifact.write_bytes(str(rank).encode())
                manifest = {
                    "schema": "stage3_fixed_batch_probe_v1", "rank": rank, "step": 115,
                    "batches": [{"keys": [[f"trajectory-{rank}", 0]]}], "identity": {"run": "one"},
                    "files": [{"name": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
                }
                (directory / f"rank_{rank:03d}_COMPLETE.json").write_text(json.dumps(manifest))
            (directory / "rank_000_batch_stale.pt").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "artifact set"):
                validate_manifests({"one": directory})

    def test_probe_loader_rejects_tensor_row_or_grid_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {
                "schema": "stage3_dino_feature_batch_v1", "keys": [["a", 0]],
                "actions": torch.zeros(1, 4), "dino": torch.zeros(1, 4, 64, 1024),
                "current_dino": torch.zeros(1, 64, 1024), "predicted": torch.zeros(1, 4, 64, 1024),
                "online_direct": torch.zeros(1, 4, 64, 1024), "online_current": torch.zeros(2, 64, 1024),
            }
            torch.save(payload, directory / "rank_000_batch_000.pt")
            with self.assertRaisesRegex(ValueError, "online_current shape mismatch"):
                load_probe(directory)

    def test_column_layout_expands_for_named_probe_headers(self):
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        labels = ("raw", "state_a_very_long_named_probe_predicted")
        offsets, width = column_layout(draw, labels)
        self.assertEqual(offsets[0], 0)
        self.assertGreaterEqual(offsets[1], 144)
        self.assertGreaterEqual(width-offsets[1], draw.textbbox((0, 0), labels[1])[2]+8)

    def test_checkpoint_family_and_split_are_bound(self):
        evaluation = {"condition": "state", "split": "eval", "split_sha256": "right"}
        identity = {"train": {"condition": "state"}, "eval": evaluation, "steps": 4000,
                    "batch": 32, "seed": 20260921, "decoder_family": "spatial_grid_v1"}
        payload = {"step": 4000, "identity": identity}
        normalized = validate_decoder_identity(payload, evaluation, "state")
        with self.assertRaisesRegex(ValueError, "family/evaluation"):
            validate_decoder_identity(payload, evaluation, "dino")
        other = copy.deepcopy(payload)
        other["identity"]["train"]["condition"] = "dino"
        other["identity"]["eval"]["condition"] = "dino"
        validate_decoder_identity(other, evaluation, "dino", normalized)
        other["identity"]["eval"]["split_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "family/evaluation"):
            validate_decoder_identity(other, evaluation, "dino", normalized)


if __name__ == "__main__":
    unittest.main()
