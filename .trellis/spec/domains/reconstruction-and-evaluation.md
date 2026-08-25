# Reconstruction and Evaluation

## Ownership

`nimloth.recon` contains post-hoc CFM/RCDM reconstruction models that consume frozen Nimloth state representations. `nimloth.training.reconstruction` owns decoder training/evaluation entry behavior. `nimloth.eval` owns model-dependent offline evaluation and reconstruction diagnostics; online environment rollout remains in the environment/Agent path.

Reconstruction must not silently enter SFT2/RL optimization, alter the state source, or substitute a service/decoder object for the real model graph. Record the exact source checkpoint/state representation and whether inputs are oracle, predicted, copied, shuffled-action, cached, or re-encoded.

## Evaluation evidence

Every report names:

- exact checkpoint/component and configuration;
- dataset asset, verified split, overlap evidence, and sample/statistical unit;
- command/commit/output provenance;
- metric definition and aggregation;
- failures/exclusions and validity limits.

Static dataset success rates are not model evaluation. Results from one eval set do not generalize to another. Training rollouts, smoke samples, reconstruction image quality, WM MSE, action-value loss, average reward, and held-out success answer different questions and must not be relabeled.

## Sources

- [`recon/README.md`](../../../src/nimloth/recon/README.md)
- [`recon/cfm/README.md`](../../../src/nimloth/recon/cfm/README.md)
- [`recon/rcdm/README.md`](../../../src/nimloth/recon/rcdm/README.md)
- [`training/reconstruction/README.md`](../../../src/nimloth/training/reconstruction/README.md)
- [`eval/README.md`](../../../src/nimloth/eval/README.md)
- [`configs/eval/`](../../../configs/eval/)

Evaluation or reconstruction execution is an experiment and requires the experiment task/lifecycle contract.
