# Progress

## 2026-09-11 implementation and review

- Added formal `format_eval_batch_size`; standard Stage 1 value is 4.
- Batched the fixed 32 prompt generation while preserving order, FSDP synchronized calls, and strict sampled-token validation.
- Added per-batch progress plus separate validation and format-generation timing.
- Independent review fixed text-only multimodal input handling and the EOS trailing-content strictness bug.
- Verification: focused 10 passed; all SFT1 172 passed with 9 pre-existing warnings; compileall and diff check passed. Stage 2 had 152 passed, 1 skipped, and 3 sandbox-only Gloo address-resolution failures. Ruff and static type tools were unavailable.
- Remote old-code run remains active in epoch 4 validation. Do not interrupt before a complete `epoch_004/COMMITTED` or later boundary. After the code commit, refresh remote state and resume from the newest complete epoch for the real batch/FSDP timing gate.
