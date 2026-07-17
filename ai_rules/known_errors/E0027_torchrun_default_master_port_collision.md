# E0027: Do not use torchrun's default rendezvous port on shared nodes

## Error

RL dynamic-rollout smoke job `477199` failed before worker/model initialization on dgx-32:

```text
DistNetworkError: ... port: 29500 ... EADDRINUSE ... address already in use
```

The launch used `python -m torch.distributed.run --nproc_per_node=2` without an explicit standalone rendezvous, so torchrun tried the shared default TCP port29500 already owned by another process.

## Correct practice

For single-node multi-process jobs on shared Slurm nodes, launch with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=<N> -- ...
```

`--standalone` chooses a free local rendezvous endpoint. Do not kill or inspect the owner of an occupied port. For intentional multi-node jobs, configure a unique explicit rendezvous endpoint instead. Do not derive the port only from a bounded job-ID modulo: ID12 job478559 used `31000 + job_id % 10000 = 39559` and collided on a shared trainer node. Probe a free port from the rank-0/master node immediately before launching all ranks, record it in topology metadata, and still fail closed on any rendezvous error.
