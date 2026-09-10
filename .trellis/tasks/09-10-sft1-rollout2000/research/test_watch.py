"""Isolated watcher tests; deletion itself is covered by cleanup tests."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watch', Path(__file__).with_name('watch_checkpoints.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WatchSafety(unittest.TestCase):
    def test_ignores_uncommitted_and_reports_failure_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for epoch in ('001', '002', '003'):
                (run/f'epoch_{epoch}').mkdir()
            (run/'epoch_001'/'COMMITTED').write_text('{}')
            (run/'epoch_003'/'COMMITTED').write_text('{}')
            (run/'cleanup_epoch_003_complete.json').write_text('{}')
            stop = run/'stop'
            stop.touch()
            with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as execute:
                module.watch(run, stop)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.args[0][-2:], ['--through-epoch', '1'])
            self.assertTrue((run/'epoch_001').is_dir())


if __name__ == '__main__':
    unittest.main()
