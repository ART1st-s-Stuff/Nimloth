"""Pure selection checks; tensor metrics additionally require torch."""
import json
from pathlib import Path
import tempfile
import unittest
from query_ablation import select_unique, donor_indices, metrics


class SelectionTests(unittest.TestCase):
    def test_last_duplicate_preserves_its_full_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for episode in ('base_000001','common_sense_000001'):
                directory = Path(tmp)/episode
                directory.mkdir()
                turns = []
                for step in range(2):
                    (directory/f'observation_{step:03d}.png').write_bytes(bytes([step]))
                    turns.append(dict(step=step,messages=[dict(content='<image>'*(step+1))]))
                path = directory/'record.json'
                path.write_text(json.dumps(dict(stage='stage2',turns=turns,terminal_generation={'messages':[{'content':'<image>'}], 'step':2})))
                paths.append(path)
            selected = select_unique(paths)
            self.assertEqual(len(selected),2)
            self.assertEqual(selected[1]['episode'],'common_sense_000001')
            self.assertEqual(len(selected[1]['paths']),2)

    def test_donors_exclude_shared_seed_and_hash(self):
        samples = [dict(episode=e,image_sha256=h) for e,h in
                   [('base_000001','a'),('common_sense_000001','b'),('base_000002','c')]]
        donors = donor_indices(samples)
        self.assertEqual(donors[:2],[2,2])
        self.assertIn(donors[2],[0,1])
        with self.assertRaises(ValueError):
            donor_indices(samples[:2])

    def test_metrics_use_observation_axis(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        target = torch.arange(24,dtype=torch.float32).reshape(3,2,4)
        fixed = target.mean(0,keepdim=True).expand_as(target)
        self.assertEqual(metrics(fixed,target)['prediction_variance'],0)
        self.assertEqual(metrics(target,target)['variance_ratio'],1)
        self.assertIn('centered_flattened_cosine', metrics(target,target))
        with self.assertRaises(ValueError):
            metrics(target, torch.full_like(target, float('nan')))


if __name__ == '__main__':
    unittest.main()
