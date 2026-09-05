# E0144: WM DINO audits must use the original observation input path

## Error

ID56 created `real_pils` by bicubic-resizing the real next observation to the ID45 CFM decoder image size (`128×128`) before passing it to frozen DINO. This was appropriate for comparing decoder outputs at one image resolution, but `dino_target_copy_rmse` and `dino_target_predicted_rmse` were then described as if they used the exact WM DINO supervision target. The WM training/runtime teacher receives the original raw observation directly and lets the frozen DINO processor perform its own preprocessing.

The ID56 decoder-space DINO comparisons remain valid at the explicitly stated 128×128 reconstruction resolution. Its direct state-to-DINO numbers are only a legacy-resolution sensitivity result, not proof against the exact original-observation WM teacher target.

## Rule

A state/WM audit that claims parity with DINO supervision must pass the original archived observation through the same frozen teacher preprocessing path used by training/runtime. Record the input resolution and every resize before the teacher. Decoder-resolution images may be evaluated separately, but their metrics must be labeled as decoder-resolution sensitivity and must not be called the canonical WM DINO target.
