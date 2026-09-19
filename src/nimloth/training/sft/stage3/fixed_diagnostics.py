"""Read-only fixed first-validation-batch probes during joint training."""
from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from nimloth.training.sft.stage3.diagnostics import DINOFeatureWriter, _file_sha256
from nimloth.training.sft.stage3.evaluate import evaluate
from nimloth.training.sft.stage3.utils import preserve_module_modes


@contextmanager
def preserve_probe_state(modules):
    """Probes may consume RNG, but must not advance the training streams."""
    modules = list(modules)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Snapshot only devices owned by this rank/model. Enumerating all visible
    # GPUs would create unrelated CUDA contexts in every distributed process.
    devices = sorted({tensor.device.index for module in modules
                      for tensor in (*module.parameters(), *module.buffers())
                      if tensor.device.type == "cuda"})
    if torch.cuda.is_available() and torch.cuda.current_device() not in devices:
        devices.append(torch.cuda.current_device())
    try:
        with torch.random.fork_rng(devices=devices), preserve_module_modes(modules, training=False):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def run_fixed_diagnostic(loop) -> None:
    if loop.diagnostic_dir is None or not loop.diagnostic_identity:
        raise ValueError("fixed diagnostics require an output directory and run identity")
    step = int(loop.state.global_step)
    root = Path(loop.diagnostic_dir)
    directory = root / f"step_{step:06d}"
    manifest = directory / f"rank_{loop.rank:03d}_COMPLETE.json"
    identity = json.loads(json.dumps(loop.diagnostic_identity, sort_keys=True))
    complete = manifest.is_file()
    if dist.is_available() and dist.is_initialized():
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, complete)
        if len(set(statuses)) != 1:
            raise ValueError("partial distributed diagnostic completion; refusing overwrite")
    if complete:
        saved = json.loads(manifest.read_text())
        if saved["identity"] != identity or saved["step"] != step:
            raise ValueError("fixed diagnostic resume identity mismatch")
        for entry in saved["files"]:
            if _file_sha256(directory / entry["name"]) != entry["sha256"]:
                raise ValueError("fixed diagnostic artifact hash mismatch")
        return
    if any(directory.glob(f"rank_{loop.rank:03d}_*")):
        raise FileExistsError(f"incomplete fixed diagnostic: {directory}")
    reference = root / f"rank_{loop.rank:03d}_inputs.json"
    with preserve_probe_state(loop.model_runtime.agent.trainable_modules):
        # Use the same first validation batch, never the training iterator.
        first_batch = next(iter(loop.val_loader))
        writer = DINOFeatureWriter(
            directory,
            rank=loop.rank,
            state_layout=getattr(loop.model_runtime.agent.wm, "state_layout", None),
        )
        metrics = evaluate(loop.algorithm, loop.model_runtime, [first_batch],
                           batch_builder=loop.batch_builder, max_batches=1, on_batch=writer)
        batch_identity = json.loads(json.dumps(writer.batch_identities))
        reference_payload = {"identity": identity, "batches": batch_identity}
        if reference.exists():
            if json.loads(reference.read_text()) != reference_payload:
                raise ValueError("fixed diagnostic validation inputs changed")
        else:
            with reference.open("x") as stream:
                json.dump(reference_payload, stream, indent=2)
        payload = {"schema": "stage3_fixed_batch_probe_v1", "step": step,
                   "rank": loop.rank, "identity": identity, "metrics": metrics,
                   "batches": batch_identity,
                   "files": [{"name": path.name, "sha256": _file_sha256(path)} for path in writer.paths]}
        temporary = manifest.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        temporary.replace(manifest)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
