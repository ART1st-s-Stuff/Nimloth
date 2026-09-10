"""Isolated safety fixtures; these do not validate real checkpoint readability."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('cleanup', Path(__file__).with_name('cleanup_checkpoints.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CleanupSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'run'
        self.run.mkdir()
        (self.run/'TRAIN_SUCCEEDED').write_text('0\n')
        (self.run/'launch.json').write_text(json.dumps({'run_output': str(self.run)}))
        for name in ('epoch_001', 'final', 'best', 'resume_step_00000010'):
            (self.run/name).mkdir()
        (self.run/'epoch_001'/'COMMITTED').write_text('{"epoch":1,"step":27}')
        (self.run/'resume_step_00000010'/'COMMITTED').write_text('{"schema":"nimloth_early_stage_resume_v1","step":10}')
        (self.run/'resume_step_00000010'/'weights').write_text('fixture')

    def validate(self, path):
        return {'epoch': 1, 'step': 27, 'identity': {'fixture': True}}

    def test_removes_only_steps_and_records_manifest(self):
        module.cleanup(self.run, self.validate)
        self.assertFalse((self.run/'resume_step_00000010').exists())
        for name in ('epoch_001', 'final', 'best', 'cleanup_manifest.json', 'cleanup_complete.json'):
            self.assertTrue((self.run/name).exists())

    def test_failure_preserves_steps(self):
        (self.run/'TRAIN_SUCCEEDED').write_text('1')
        with self.assertRaises(ValueError):
            module.cleanup(self.run, self.validate)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_symlink_preserves_steps_and_external_file(self):
        target = Path(self.tmp.name)/'outside'
        target.write_text('retain')
        (self.run/'resume_step_00000010'/'link').symlink_to(target)
        with self.assertRaises(ValueError):
            module.cleanup(self.run, self.validate)
        self.assertEqual(target.read_text(), 'retain')
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_checkpoint_validation_failure_preserves_steps(self):
        def invalid(path):
            raise ValueError('incomplete final')
        with self.assertRaises(ValueError):
            module.cleanup(self.run, invalid)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_uncommitted_step_preserves_all(self):
        (self.run/'resume_step_00000020').mkdir()
        with self.assertRaises(FileNotFoundError):
            module.cleanup(self.run, self.validate)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_invalid_step_range_preserves_all(self):
        for step in (0, 15, 30):
            with self.subTest(step=step):
                path = self.run / f'resume_step_{step:08d}'
                path.mkdir()
                (path/'COMMITTED').write_text(json.dumps({
                    'schema': 'nimloth_early_stage_resume_v1', 'step': step}))
                with self.assertRaises(ValueError):
                    module.cleanup(self.run, self.validate)
                self.assertTrue((self.run/'resume_step_00000010').exists())
                (path/'COMMITTED').unlink()
                path.rmdir()

    def test_wrong_owner_preserves_steps(self):
        (self.run/'launch.json').write_text('{"run_output":"/another/run"}')
        with self.assertRaises(ValueError):
            module.cleanup(self.run, self.validate)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_symlink_run_rejected(self):
        alias = Path(self.tmp.name) / 'alias'
        alias.symlink_to(self.run, target_is_directory=True)
        with self.assertRaises(ValueError):
            module.cleanup(alias, self.validate)
        self.assertTrue((self.run/'resume_step_00000010').exists())


if __name__ == '__main__':
    unittest.main()
