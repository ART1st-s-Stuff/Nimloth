"""Deterministic process doubles test control decisions, not real DDP pausing."""
import importlib.util
from pathlib import Path
import signal
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('segments', Path(__file__).with_name('run_segments.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SegmentControl(unittest.TestCase):
    def test_only_exact_training_command_matches(self):
        run = Path('/run/owned')
        argv = ['python', '-m', module.MODULE, '--output-dir', str(run)]
        self.assertTrue(module.rank_command(argv, run))
        self.assertFalse(module.rank_command(argv, Path('/run/other')))
        self.assertFalse(module.rank_command(['python', '-m', 'torch.distributed.run']+argv[1:], run))

    def test_ancestry_does_not_include_other_processes(self):
        self.assertTrue(module.descendant(12, 10, {12: 11, 11: 10}))
        self.assertFalse(module.descendant(12, 10, {12: 11, 11: 12}))
        self.assertFalse(module.descendant(12, 10, {12: 9}))

    def test_pause_requires_planned_nonzero_exit_75(self):
        self.assertTrue(module.paused_exit(1, True, 'exitcode  : 75 (pid: 55)'))
        for code, planned, text in [(0, True, 'exitcode:75'), (1, False, 'exitcode:75'),
                                    (1, True, 'exitcode:1'), (1, True, 'exitcode:750'),
                                    (1, True, 'exitcode:75\nexitcode:1'),
                                    (1, True, 'exitcode:75\nCUDA out of memory')]:
            self.assertFalse(module.paused_exit(code, planned, text))

    def test_requests_usr1_only_to_eight_owned_ranks(self):
        process = Mock(pid=100, returncode=1)
        process.poll.side_effect = [None, 1]
        ranks = [(200+i, i) for i in range(8)]
        send = Mock()
        clock = Mock(side_effect=[0, module.PAUSE_SECONDS])
        result = module.wait_segment(process, Path('/run'), Mock(), clock=clock,
            sleep=Mock(), rank_finder=Mock(return_value=ranks), send=send)
        self.assertEqual(result, (1, True))
        self.assertEqual([call.args for call in send.call_args_list],
                         [(pid, signal.SIGUSR1) for pid, _ in ranks])
        self.assertNotIn(100, [call.args[0] for call in send.call_args_list])

    def test_deadline_stops_group_without_resume(self):
        process = Mock(pid=100)
        process.poll.return_value = None
        with patch.object(module, 'terminate_group') as stop:
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                module.wait_segment(process, Path('/run'), Mock(),
                    clock=Mock(side_effect=[0, module.SEGMENT_SECONDS]), sleep=Mock())
        self.assertTrue(stop.called)

    def test_no_new_boundary_fails_closed(self):
        with patch.object(module, 'boundaries', return_value={'/run/resume_step_00000010': 'same'}):
            with self.assertRaisesRegex(RuntimeError, 'no new committed'):
                module.new_boundary(Path('/run'), {'/run/resume_step_00000010': 'same'})


if __name__ == '__main__':
    unittest.main()
