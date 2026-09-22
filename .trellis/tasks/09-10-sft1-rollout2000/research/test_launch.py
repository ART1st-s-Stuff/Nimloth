"""Policy and syntax checks only; no real training or cache validation claims."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('launch.sh').read_text()
BLOCKS = dict(re.findall(r"<<'(PY[^']*)'\n(.*?)\n\1(?:\n|$)", SCRIPT, re.S))


class LaunchPolicy(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)
        (self.run/'PREPROCESS_SUCCEEDED').write_text('0')
        self.policy = dict(until_converged=True, min_epochs=2, patience_epochs=2,
                           min_relative_improvement=0.01,
                           budget='6h per segment; pause and resume until convergence')
        (self.run/'launch.json').write_text(json.dumps(dict(
            run_output=str(self.run), commit='a'*40, args=['--no-wandb'])))
        self.policy_path = self.run/'approved.json'

    def execute(self, phase='train'):
        self.policy_path.write_text(json.dumps(self.policy))
        with patch.object(sys, 'argv', ['-', str(self.run), 'a'*40,
                                       str(self.policy_path), phase, '--no-wandb']):
            exec(compile(BLOCKS['PY_POLICY'], 'PY_POLICY', 'exec'), {})

    def test_no_epoch_cap_and_resume_policy_matches(self):
        self.execute()
        args = (self.run/'policy_args.nul').read_bytes().split(b'\0')
        self.assertIn(b'--until-converged', args)
        self.assertNotIn(b'--epochs', args)
        (self.run/'TRAIN_STARTED').write_text('started')
        self.execute('resume')
        self.assertIn(b'--resume', (self.run/'policy_args.nul').read_bytes().split(b'\0'))

    def test_unapproved_policy_rejected(self):
        self.policy['patience_epochs'] = 5
        with self.assertRaises(AssertionError):
            self.execute()
        self.assertFalse((self.run/'training_policy.json').exists())

    def test_changed_source_rejected(self):
        metadata = json.loads((self.run/'launch.json').read_text())
        metadata['commit'] = 'b'*40
        (self.run/'launch.json').write_text(json.dumps(metadata))
        with self.assertRaises(AssertionError):
            self.execute()

    def test_all_embedded_python_compiles(self):
        for name, code in BLOCKS.items():
            compile(code, name, 'exec')


if __name__ == '__main__':
    unittest.main()
