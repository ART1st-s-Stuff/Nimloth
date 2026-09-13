"""Synthetic CPU checks; these do not validate GPU replay or rollout quality."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('features',Path(__file__).with_name('rollout_features.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Tokenizer:
    def decode(self,ids,**kwargs):
        return ''.join({1:'<think>',2:'reason',3:'</think>',4:'<|action_start|>'}[i] for i in ids)

class Tests(unittest.TestCase):
    def test_parameter_cast_preserves_rotary_buffers(self):
        import torch
        model=torch.nn.Module()
        model.weight=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
        model.register_buffer('inv_freq',torch.ones(2,dtype=torch.float32))
        before=model.inv_freq.clone()
        module.cast_replay_parameters(model)
        self.assertEqual(model.weight.dtype,torch.bfloat16)
        self.assertEqual(model.inv_freq.dtype,torch.float32)
        self.assertTrue(torch.equal(model.inv_freq,before))

    def test_query_boundary_excludes_action(self):
        generation=dict(sampled_token_ids=[1,2,3,4],inserted_token_ids=[9,10])
        self.assertEqual(module.query_prefix(generation,Tokenizer(),[9,10]),[1,2,3,9,10])
        generation['inserted_token_ids']=[10,9]
        with self.assertRaises(ValueError): module.query_prefix(generation,Tokenizer(),[9,10])

    def test_missing_boundary_fails(self):
        with self.assertRaises(ValueError):
            module.query_prefix(dict(sampled_token_ids=[1,2],inserted_token_ids=[9]),Tokenizer(),[9])

    def test_target_only_shared_projection(self):
        target=np.zeros((2,2,2,3)); target[...,0]=np.arange(8).reshape(2,2,2)
        center,basis,lo,hi,ratio=module.fit_pc1(target)
        np.testing.assert_allclose(basis,[1,0,0])
        np.testing.assert_allclose(center,[3.5,0,0])
        prediction=target.copy(); prediction[...,0]+=2
        np.testing.assert_allclose((prediction-center)@basis-(target-center)@basis,2)
        self.assertAlmostEqual(ratio,1)
        self.assertLess(lo,hi)

    def test_constant_and_nonfinite_fail(self):
        for value in (np.zeros((2,2,2,3)),np.full((2,2,2,3),np.nan)):
            with self.assertRaises(ValueError): module.fit_pc1(value)

if __name__=='__main__': unittest.main()
