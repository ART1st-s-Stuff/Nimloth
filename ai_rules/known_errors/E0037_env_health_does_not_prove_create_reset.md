# E0037: Environment HTTP health does not prove create/reset works

## Error

ID16 job `478689` started the VAGEN server on dgx-37 and `/health` passed. The Unity `thor-CloudRendering` process also launched and used99% GPU SM. Nevertheless rank0 blocked inside `create_environments_batch`; after600 seconds the VAGEN client raised `ReadTimeout`. Ranks1–7 were correctly waiting for rank0's Gloo broadcast, so the GPU-idle pattern was initially misread as an FSDP forward deadlock.

ID14 and ID15 showed the same apparent stall but lacked stack diagnostics. ID16 SIGUSR1 dumps proved no policy input or model forward had started.

## Correct practice

An AI2-THOR env node is healthy only after a bounded semantic preflight completes `create_environments_batch`, system prompt retrieval, `reset_batch`, observation schema validation, and close—not merely HTTP `/health` or Unity process startup. Keep stack-dump diagnostics available and inspect rank0 before attributing waiting ranks to NCCL/FSDP. Prefer a node that has reached a real policy turn in the current runtime; ID13 demonstrated that on dgx-51.
