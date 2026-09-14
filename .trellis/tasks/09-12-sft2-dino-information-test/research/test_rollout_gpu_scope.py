"""Readiness must ignore unrelated devices without changing process ownership."""
import unittest
from run_epoch2_rollout import compute_query_command


class ComputeQueryTests(unittest.TestCase):
    def test_explicit_physical_ids_not_cuda_logical_ordinals(self):
        contract = {'physical_gpu_indices':[2,3], 'env':{'CUDA_VISIBLE_DEVICES':'2,3'}}
        self.assertEqual(compute_query_command(contract),[
            'nvidia-smi','-i','2,3','--query-compute-apps=pid','--format=csv,noheader,nounits'])

    def test_omitted_scope_preserves_whole_host_check(self):
        self.assertEqual(compute_query_command({}),[
            'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])

    def test_invalid_scope_fails_closed(self):
        for indices in ([],None,'2,3',[True],[2,2],[-1],[2.0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                compute_query_command({'physical_gpu_indices':indices})


if __name__ == '__main__':
    unittest.main()
