# E0034 — BatchNorm scratch buffers must start from current running statistics

## Error

SFT2 world2 smoke `482289` completed ten finite train steps, then every validation metric became NaN. `SafeBatchNorm1d` avoided inplace buffer-version conflicts by passing scratch running buffers to `torch.batch_norm`, but allocated them with `new_empty`. BatchNorm applies its momentum update to the supplied prior values, so uninitialized memory produced huge and sometimes negative running variances. Train mode hid the corruption by using batch statistics; eval mode exposed it.

The invalid checkpoint's online encoder had 37 negative `running_var` entries. All ID31 checkpoints are forbidden.

## Prevention

- Initialize scratch mean/variance by cloning the module's current running statistics before calling `F.batch_norm(training=True)`.
- Regression-test running stats against standard `nn.BatchNorm1d` after one update and test repeated forwards before backward.
- Fail training/evaluation immediately when any loss or metric is non-finite; a Slurm exit code of zero is not sufficient evidence of a valid experiment.
