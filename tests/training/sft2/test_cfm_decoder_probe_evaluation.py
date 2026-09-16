import copy
import unittest

import torch

from experiments.training.sft.stage3.evaluate_cfm_decoder_probe import (
    aligned_rows, image_metrics, paired_noise, validate_decoder_identity,
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
        probes = {e: {("a", 0): copy.deepcopy(item)} for e in (2, 4, 5)}
        rows = aligned_rows(probes, cached)
        self.assertEqual([row["observation"] for row in rows], [1, 2, 3, 4])
        self.assertTrue(torch.equal(rows[0]["gt_state"], cached["a"]["states"][1]))
        probes[4][("a", 0)]["actions"][0] = 0
        with self.assertRaisesRegex(ValueError, "actions mismatch"):
            aligned_rows(probes, cached)
        probes[4] = copy.deepcopy(probes[2])
        cached["a"]["dino"] = dino+5
        with self.assertRaisesRegex(ValueError, "current DINO"):
            aligned_rows(probes, cached)

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
