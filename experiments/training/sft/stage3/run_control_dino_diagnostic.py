"""Bounded Stage2 versus completed control diagnostic; never launches treatment."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import torch
from render_dino_feature_comparison import load_export


def decompose(error: torch.Tensor) -> dict:
    """Exact MSE decomposition across observations, retaining slot/channel axes."""
    error = error.double()
    mean = error.mean(0, keepdim=True)
    return {"mse": float(error.square().mean()),
            "mean_bias_mse": float(mean.square().mean()),
            "centered_error_mse": float((error - mean).square().mean())}


def summarize(stage2: dict, control: dict) -> dict:
    if not stage2 or stage2.keys() != control.keys():
        raise ValueError("feature identities differ or are empty")
    unique = {}
    for key in sorted(stage2):
        left, right = stage2[key], control[key]
        for field in ("actions", "dino", "current_dino"):
            if not torch.equal(left[field], right[field]):
                raise ValueError(f"{field} mismatch: {key}")
        for offset in range(len(left["dino"])):
            identity = (key[0], key[1] + offset + 1)
            values = (left["online_direct"][offset], right["online_direct"][offset], left["dino"][offset])
            if identity in unique and any(not torch.equal(a, b) for a, b in zip(unique[identity], values)):
                raise ValueError(f"inconsistent duplicate observation: {identity}")
            unique[identity] = values
    initial, final, target = (torch.stack([v[i] for v in unique.values()]).double() for i in range(3))
    result = {"scope": "unique observed successor states represented by exported windows; excludes initial observations",
              "observations": len(unique), "trajectories": len({key[0] for key in unique}),
              "models": {}, "output_shift": decompose(final - initial)}
    for name, prediction in (("stage2", initial), ("control", final)):
        centered = prediction - prediction.mean(0, keepdim=True)
        result["models"][name] = dict(decompose(prediction - target),
            cosine=float(torch.nn.functional.cosine_similarity(prediction.flatten(1), target.flatten(1)).mean()),
            mean=float(prediction.mean()), rms=float(prediction.square().mean().sqrt()),
            mean_grid_l2=float(prediction.flatten(1).norm(dim=1).mean()),
            observation_variance=float(centered.square().mean()))
    result["target"] = {"mean": float(target.mean()), "rms": float(target.square().mean().sqrt()),
                        "observation_variance": float((target-target.mean(0)).square().mean())}
    result["output_shift"]["relative_l2"] = float((final-initial).norm()/initial.norm()) if initial.norm() else None
    return result


def build_commands(prior: dict, output: Path, checkpoint: Path) -> list[dict]:
    phases = [p for p in prior["phases"] if p["phase"] == "formal" and p["arm"] == "control"]
    if len(phases) != 1:
        raise ValueError("expected exactly one formal control phase")
    commands = []
    for name in ("stage2", "control"):
        command = phases[0]["argv"].copy()
        for flag in ("--outcome-eval-dir", "--wandb-run-name", "--resume-from"):
            if flag in command:
                index = command.index(flag)
                del command[index:index+2]
        if "--resume" in command:
            command.remove("--resume")
        command[command.index("--output-dir")+1] = str(output/name/"runtime")
        port_index = next(i for i, value in enumerate(command) if value.startswith("--master_port="))
        with socket.socket() as sock:
            sock.bind(("", 0))
            command[port_index] = f"--master_port={sock.getsockname()[1]}"
        command += ["--eval-only", "--no-wandb", "--max-val-batches", "1",
                    "--feature-export-dir", str(output/name/"features")]
        if name == "control":
            command += ["--resume", "--resume-from", str(checkpoint)]
        commands.append({"name": name, "argv": command})
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != args.commit:
        raise ValueError("source commit mismatch")
    checkpoint = args.run_root/"control/epoch_002"
    for relative in ("training_state.pt", "vision_ema.pt", "selected_token_rows.pt", "state_proj.pt",
                     "wm_predictor/predictor.pt", "model.safetensors.index.json"):
        if not (checkpoint/relative).is_file():
            raise FileNotFoundError(checkpoint/relative)
    state = torch.load(checkpoint/"training_state.pt", map_location="cpu", weights_only=False)
    if state.get("epoch") != 2 or not state.get("epoch_complete", False):
        raise ValueError("control epoch2 checkpoint is incomplete")
    del state
    prior = json.loads((args.run_root/"controller/progress.json").read_text())
    commands = build_commands(prior, args.output, checkpoint)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    if len(gpu.strip().splitlines()) != 8 or any(int(line.split(",")[1]) > 10 for line in gpu.strip().splitlines()):
        raise RuntimeError(f"GPUs occupied: {gpu}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    contract = {"commit": args.commit, "started": started, "deadline": started+900,
                "controller_pid": os.getpid(), "commands": commands, "status": "running", "phases": []}
    def save():
        temporary = args.output/"status.tmp"
        temporary.write_text(json.dumps(contract, indent=2))
        temporary.replace(args.output/"status.json")
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", WANDB_MODE="disabled",
               PYTHONPATH=str(Path.cwd()/"src"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    child = None
    try:
        for item in commands:
            with (args.output/(item["name"]+".log")).open("x") as log:
                child = subprocess.Popen(item["argv"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                phase = {"name": item["name"], "pid": child.pid, "started": time.time()}
                contract["phases"].append(phase)
                save()
                code = child.wait(timeout=max(1, contract["deadline"]-time.time()))
                phase.update(returncode=code, finished=time.time())
                save()
                if code:
                    raise RuntimeError(f"{item['name']} exited {code}")
        # Summary runs in a bounded child so CPU diagnostics cannot overrun the deadline.
        with (args.output/"summary.log").open("x") as log:
            child = subprocess.Popen([commands[0]["argv"][0], str(Path(__file__).resolve()),
                                      "--summarize", str(args.output)], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait(timeout=max(1, contract["deadline"]-time.time())):
                raise RuntimeError("summary failed")
        contract["status"] = "complete"
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        contract.update(status="failed", error=str(error))
        raise
    finally:
        contract["finished"] = time.time()
        save()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        output = Path(sys.argv[2])
        metrics = summarize(load_export(output/"stage2/features"), load_export(output/"control/features"))
        (output/"metrics.json").write_text(json.dumps(metrics, indent=2))
    else:
        main()
