# Preflight evidence

- 2026-09-11 16:10-16:15 UTC: training process group stopped; all eight A100 40GB GPUs report zero compute memory use.
- epoch 5/6 each contain `COMMITTED`, 1.84GB adapter weights and 3.67GB training state; identity records the expected Stage 1 objective, action weight 8, success-only prepared data hashes and FSDP world size 8.
- Host has 494GiB available RAM and 319GiB available disk.
- Ports 18005/18006 and both output roots are unused.
- `/mnt/nimloth/venv` imports vLLM 0.8.5.post1 and the VAGEN Batch server. The accepted VAGEN source is `/mnt/nimloth/sources/vagen` at `844378c`; `/mnt/nimloth/sources/vagen-step60-runtime` is `170a673d` and is explicitly excluded.
