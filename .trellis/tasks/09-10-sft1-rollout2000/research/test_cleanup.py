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
        module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
        self.assertFalse((self.run/'resume_step_00000010').exists())
        for name in ('epoch_001', 'final', 'best', 'cleanup_manifest.json', 'cleanup_complete.json'):
            self.assertTrue((self.run/name).exists())

    def test_terminal_epoch_selected_from_final_state(self):
        terminal = self.run/'epoch_003'
        terminal.mkdir()
        (terminal/'COMMITTED').write_text('{"epoch":3,"step":81}')
        visited = []
        def validate(path):
            visited.append(path.name)
            return {'epoch': 3, 'step': 81, 'identity': {'fixture': True}}
        module.cleanup(self.run, validate, step_validator=lambda *args: None)
        self.assertEqual(visited, ['final', 'epoch_003'])
        manifest = json.loads((self.run/'cleanup_manifest.json').read_text())
        self.assertIn('epoch_003', manifest['retained'])
        self.assertTrue((self.run/'epoch_001').is_dir())

    def test_final_epoch_mismatch_preserves_steps(self):
        def validate(path):
            return {'epoch': 1, 'step': 27 if path.name == 'final' else 26,
                    'identity': {'fixture': True}}
        with self.assertRaises(ValueError):
            module.cleanup(self.run, validate, step_validator=lambda *args: None)
        self.assertTrue((self.run/'resume_step_00000010').is_dir())

    def test_completed_state_requires_format_only_contract(self):
        state = dict(epoch=3, step=81, training_stage='format',
                     format_objective='format_answer_ce_v2', world_size=8,
                     lora=True, best_val=0.3, optimizer={'a': 1},
                     scheduler={'a': 1}, identity={'a': 1},
                     latent_token_count=None, latent_query_mode=None,
                     mask_latent_query_labels=None,
                     convergence_state=dict(schema='adjacent_epoch_lm_v1', best_loss=0.3,
                         previous_loss=0.3, bad_epochs=2, last_epoch=3, converged=True),
                     rank_rng_states=[dict(python=1, numpy=1, torch_cpu=1, torch_cuda=1) for _ in range(8)])
        module.validate_completed_state(state, self.run/'final')
        for field, value in [('format_objective', None), ('latent_token_count', 16),
                             ('latent_query_mode', 'generate'),
                             ('mask_latent_query_labels', False), ('epoch', 0),
                             ('best_val', float('nan')), ('rank_rng_states', []),
                             ('convergence_state', {})]:
            with self.subTest(field=field):
                invalid = dict(state, **{field: value})
                with self.assertRaises(ValueError):
                    module.validate_completed_state(invalid, self.run/'final')

    def test_epoch_cleanup_retains_newer_steps_without_final(self):
        (self.run/'TRAIN_SUCCEEDED').unlink()
        (self.run/'final').rmdir()
        (self.run/'resume_step_00000030').mkdir()
        module.cleanup(self.run, self.validate, through_epoch=1, step_validator=lambda *args: None)
        self.assertFalse((self.run/'resume_step_00000010').exists())
        self.assertTrue((self.run/'resume_step_00000030').exists())
        self.assertTrue((self.run/'cleanup_epoch_001_complete.json').exists())
        module.cleanup(self.run, self.validate, through_epoch=1, step_validator=lambda *args: None)

    def test_failure_preserves_steps(self):
        (self.run/'TRAIN_SUCCEEDED').write_text('1')
        with self.assertRaises(ValueError):
            module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_symlink_preserves_steps_and_external_file(self):
        target = Path(self.tmp.name)/'outside'
        target.write_text('retain')
        (self.run/'resume_step_00000010'/'link').symlink_to(target)
        with self.assertRaises(ValueError):
            module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
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
            module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_invalid_step_range_preserves_all(self):
        for step in (0, 30):
            with self.subTest(step=step):
                path = self.run / f'resume_step_{step:08d}'
                path.mkdir()
                (path/'COMMITTED').write_text(json.dumps({
                    'schema': 'nimloth_early_stage_resume_v1', 'step': step}))
                with self.assertRaises(ValueError):
                    module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
                self.assertTrue((self.run/'resume_step_00000010').exists())
                (path/'COMMITTED').unlink()
                path.rmdir()

    def test_committed_off_cadence_pause_checkpoint_can_be_cleaned(self):
        path = self.run/'resume_step_00000015'
        path.mkdir()
        (path/'COMMITTED').write_text(json.dumps({
            'schema': 'nimloth_early_stage_resume_v1', 'step': 15}))
        module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
        self.assertFalse(path.exists())

    def test_wrong_owner_preserves_steps(self):
        (self.run/'launch.json').write_text('{"run_output":"/another/run"}')
        with self.assertRaises(ValueError):
            module.cleanup(self.run, self.validate, step_validator=lambda *args: None)
        self.assertTrue((self.run/'resume_step_00000010').exists())

    def test_symlink_run_rejected(self):
        alias = Path(self.tmp.name) / 'alias'
        alias.symlink_to(self.run, target_is_directory=True)
        with self.assertRaises(ValueError):
            module.cleanup(alias, self.validate)
        self.assertTrue((self.run/'resume_step_00000010').exists())


if __name__ == '__main__':
    unittest.main()
