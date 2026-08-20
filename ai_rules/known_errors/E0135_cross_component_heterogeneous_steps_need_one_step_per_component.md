# E0135 — Cross-component heterogeneous work needs one step per component

## Error

ID187 retry9 Job `525646` again obtained the intended 6+2 allocation. Removing the explicit base `--jobid` was necessary but insufficient: a nested command using `srun --het-group=0,1 --nodes=2` still applied the two-node request to a one-node heterogeneous component and failed with `Only allocated 1 nodes asked for 2`.

## Rule

For this Slurm 23.02 cluster, do not treat comma-selected heterogeneous groups as one flat allocation for nested utility steps. Launch one node-scoped step per component:

- map each allocated node to its exact heterogeneous group;
- use the heterogeneous leader JobId with one exact offset: `srun --jobid=<leader> --het-group=<offset> -w <node>`; the sibling component JobId is not itself an allocation leader, while inherited steps without the leader were unreliable;
- run fabric probes, runtime setup/audit, Ray-log capture, and cleanup once per node;
- explicitly set each node-scoped step's CPU and memory requests; otherwise a group1 step may inherit component0's larger `cpus-per-task` or memory and become unschedulable;
- aggregate their outputs and statuses in the controller.

Ray may still combine both raylets into one logical 6+2 cluster after the component-local Slurm steps start them.
