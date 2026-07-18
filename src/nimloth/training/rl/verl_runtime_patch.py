"""Idempotent runtime compatibility patches imported inside every VERL worker."""

from nimloth.training.rl.verl_critic_455 import (
    install_verl_transformers455_critic_patch,
)
from nimloth.training.rl.verl_gate import install_verl_zero_warmup_scheduler_patch


def install_nimloth_verl_runtime_patches() -> None:
    install_verl_zero_warmup_scheduler_patch()
    install_verl_transformers455_critic_patch()


# VERL's ``external_lib`` hook imports this module before model/optimizer build.
install_nimloth_verl_runtime_patches()
