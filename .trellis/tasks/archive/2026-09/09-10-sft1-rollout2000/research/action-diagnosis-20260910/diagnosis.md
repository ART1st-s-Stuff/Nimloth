# Missing action output diagnosis

Training stopped at user request. Final verification at 2026-09-10 14:35:38 UTC: controller/launcher/ranks absent, no GPU compute processes. Latest logged training step54; latest committed resume step50; epoch001 step27 preserved. Epoch2 validation not committed. Diagnostic GPU run exited0 in22.657s, no optimizer updates.

## Findings

1. No missing action supervision in actual cache: all1709 training records,20212 action blocks; all193 validation records,2152 action blocks; every supervised action_start is immediately followed by an action label. Targets include action_start, action index, action_end and EOS. Stage1 has no query/latent.
2. Real base tokenizer151665 has no action tokens. Added10 action tokens use IDs151665..151674. Base model padded vocab151936 is shrunk to151675, so these IDs initially reuse preexisting padding rows. This establishes novel-token cold start, not by itself an initialization defect.
3. Real saved embedding/head trainable modules and AdamW moments are BF16, LR5e-6, no FP32 master update. Step50 action head only2730/20480 elements differ from base; embedding2690/20480. All20480 first-moment elements per module are nonzero. Read-only bias-corrected update arithmetic from saved moments rounds18359/20480 head and18549/20480 embedding values back to the same BF16 weight. This and real-library reproduction establish a substantial lost-update precision mechanism; it does not prove sole causality or that a fix alone guarantees convergence.
4. Real GPU FA2 exact adapter reload confirms output behavior. On first heldout reference-think prefix, action_start probability baseline1.9019e-8, epoch0011.0837e-7, step508.2360e-7. BaselineEOS .95747, epoch001EOS .78597; step50 ordinary character0 is top1(.56435), EOS .08130. After action_start, reference action3 probability step50 only7.6482e-8. New action token logits remain ~2.7-3 while competing known tokens are far higher.
5. Actual greedy baseline response ends </think><|im_end|>; step50 ends </think>0<|im_end|>. No length truncation. Prior epoch0014sampleCPU output omitted all actionblocks. Both GPU in-training0/32 and reloaded real generation corroborate, so this is not merely loose regex or CPU decoding.
6. All saved702 tensors loaded exactly, no unexpected keys. Generation config has EOS151645, pad151643 and no action suppression. No evidence of wrong checkpoint load or stopping on action token.

## Conclusion and scope

Confirmed implementation issue: BF16 trainable embedding/head + tiny learning rate loses a large fraction of updates, especially harmful for new action tokens which begin with negligible probability. The model still prefers pretrained EOS/ordinary characters over correct new action tokens; total LM loss can improve without fixing this sparse answer component. Full causal contribution versus duration/objective weighting needs a controlled corrected run; not yet performed.

Recommended next implementation: maintain FP32 trainable saved embedding/head and AdamW states/master updates while keeping frozen backbone BF16 and compatible FSDP forward; verify real action-row updates, save/load and FSDP memory gate. Validate initialization of genuinely new IDs in padded vocabulary, then compare action-token CE/probability and free generation. Do not simply force generation tokens or increase epochs to hide the issue. No trainer fixes, data changes, learning-rate changes or training restart performed in this diagnosis.
