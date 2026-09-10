"""CPU ownership checks for the experiment's bounded segment controller."""
import importlib.util
import signal
import unittest
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[3] / '.trellis/tasks/09-10-sft1-rollout2000/research/run_segments.py'
spec = importlib.util.spec_from_file_location('segment_controller', path)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ProcessOwnershipTests(unittest.TestCase):
    def test_ranks_exclude_inherited_dataloader_and_unrelated_trainers(self):
        parents = {10: 1, 11: 10, 900: 1}
        candidates = [(900, 0)]
        expected = []
        for rank in range(8):
            pid = 100 + rank
            parents[pid] = 11
            parents[pid + 100] = pid
            parents[pid + 200] = pid + 100
            candidates.extend([(pid, rank), (pid + 100, rank), (pid + 200, rank)])
            expected.append((pid, rank))
        # No PGID input: real torchrun ranks use distinct groups.
        self.assertEqual(controller.select_ranks(10, candidates, parents), expected)

    def test_missing_or_duplicate_rank_fails_closed(self):
        parents = {pid: 10 for pid in range(100, 108)}
        with self.assertRaisesRegex(RuntimeError, 'exactly 8'):
            controller.select_ranks(10, [(100 + i, i) for i in range(7)], parents)
        with self.assertRaisesRegex(RuntimeError, 'exactly 8'):
            controller.select_ranks(10, [(100 + i, 0) for i in range(8)], parents)

    def test_cleanup_tracks_reparented_children_and_skips_reused_pids(self):
        current = {10: (1, 1000, 'S'), 20: (10, 2000, 'S'),
                   30: (20, 3000, 'S'), 40: (10, 4000, 'S'),
                   900: (1, 9000, 'S')}
        sent, waited = [], []
        now = [0.0]

        def send(pid, sig):
            sent.append((pid, sig))
            if pid == 10 and sig == signal.SIGTERM:
                del current[10]
                current[20] = (1, 2000, 'S')  # Own worker survives launcher.
                current[40] = (1, 9999, 'S')  # PID reused by unrelated process.
            if sig == signal.SIGKILL:
                del current[pid]

        process = SimpleNamespace(pid=10, wait=lambda timeout: waited.append(timeout))
        controller.terminate_group(process, snapshotter=lambda: dict(current),
                                   send=send, clock=lambda: now[0],
                                   sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
        self.assertEqual(sent, [(10, signal.SIGTERM), (20, signal.SIGTERM),
                                (30, signal.SIGTERM), (20, signal.SIGKILL),
                                (30, signal.SIGKILL)])
        self.assertEqual(set(current), {40, 900})
        self.assertEqual(waited, [5])
        self.assertGreaterEqual(now[0], 30)

    def test_cleanup_fails_bounded_if_owned_process_survives_kill(self):
        now = [0.0]
        process = SimpleNamespace(pid=10, wait=lambda timeout: None)
        with self.assertRaisesRegex(RuntimeError, 'survived SIGKILL'):
            controller.terminate_group(
                process, snapshotter=lambda: {10: (1, 1000, 'D')},
                send=lambda pid, sig: None, clock=lambda: now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            )
        self.assertGreaterEqual(now[0], 35)
        self.assertLess(now[0], 36)

    def test_failed_launcher_cleans_retained_reparented_worker(self):
        current = {10: (1, 1000, 'S'), 20: (10, 2000, 'S'), 99: (1, 9900, 'S')}
        process = SimpleNamespace(pid=10, returncode=None, wait=lambda timeout: 1)
        now, sent = [0.0], []

        def sleep(seconds):
            now[0] += seconds
            current.pop(10, None)
            current[20] = (1, 2000, 'S')
            process.returncode = 1

        def send(pid, sig):
            sent.append((pid, sig))
            current.pop(pid)

        def terminate(process, *, owned):
            controller.terminate_group(process, owned=owned,
                                       snapshotter=lambda: dict(current), send=send)

        process.poll = lambda: process.returncode
        self.assertEqual(controller.wait_segment(
            process, Path('/run/test'), lambda *args, **kwargs: None,
            clock=lambda: now[0], sleep=sleep, snapshotter=lambda: dict(current),
            terminator=terminate,
        ), (1, False))
        self.assertEqual(sent, [(20, signal.SIGTERM)])
        self.assertEqual(set(current), {99})

    def test_retained_leader_identity_blocks_reused_leader_descendants(self):
        owned = {10: 1000, 20: 2000}
        controller.remember_owned(10, {10: (1, 9999, 'S'), 99: (10, 9900, 'S')}, owned)
        self.assertEqual(owned, {10: 1000, 20: 2000})

    def test_zombies_are_finished(self):
        self.assertFalse(controller.same_process(1, 55, {1: (0, 55, 'Z')}))
        self.assertFalse(controller.same_process(1, 55, {1: (0, 56, 'S')}))
        self.assertTrue(controller.same_process(1, 55, {1: (0, 55, 'S')}))


if __name__ == '__main__':
    unittest.main()
