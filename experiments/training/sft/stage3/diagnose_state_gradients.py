"""CPU subset state-space SIGReg/DINO compatibility; not parameter gradients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from render_dino_feature_comparison import load_export

from nimloth.wm.sigreg import SequenceSIGReg


def gradient_geometry(dino_grad, sigreg_grad):
    dino, sigreg = dino_grad.double().flatten(), sigreg_grad.double().flatten()
    dn, sn = dino.norm(), sigreg.norm()
    dot = (dino * sigreg).sum()
    return {"weighted_dino_gradient_norm": float(dn),
            "weighted_sigreg_gradient_norm": float(sn),
            "gradient_dot": float(dot),
            "gradient_cosine": float(dot/(dn*sn)) if dn > 0 and sn > 0 else None,
            "dino_directional_derivative_along_unit_negative_sigreg": float(-dot/sn) if sn > 0 else None}


def extract_observations(rows):
    observed = {}
    for (trajectory, start), row in sorted(rows.items()):
        for offset in range(len(row["dino"])):
            identity = (trajectory, start+offset+1)
            value = (row["online_direct"][offset], row["dino"][offset])
            if identity in observed and any(not torch.equal(a, b) for a, b in zip(observed[identity], value)):
                raise ValueError(f"inconsistent duplicate observation {identity}")
            observed[identity] = value
    identities = sorted(observed)
    indices = {key: index for index, key in enumerate(identities)}
    pairs = [(index, indices[(key[0], key[1]+1)]) for key, index in indices.items()
             if (key[0], key[1]+1) in indices]
    if len(pairs) < 2:
        raise ValueError("at least two distinct adjacent pairs required")
    return identities, torch.stack([observed[k][0] for k in identities]), torch.stack([observed[k][1] for k in identities]), torch.tensor(pairs)


def evaluate(states, targets, pairs, *, seed, num_proj=1024, knots=17):
    # Detached clones prevent any alteration of exported data or model parameters.
    state = states.detach().float().clone().requires_grad_(True)
    target = targets.detach().float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        regularizer = SequenceSIGReg(knots=knots, num_proj=num_proj)
        # Production pools slots, detaches current occurrences, and differentiates successors.
        sequence = torch.stack((state[pairs[:, 0]].detach().mean(-2), state[pairs[:, 1]].mean(-2)), dim=1)
        raw_sigreg = regularizer(sequence)
        raw_dino = (state-target).square().mean()
        dino_grad = torch.autograd.grad(.5*raw_dino, state, retain_graph=True)[0]
        sigreg_grad = torch.autograd.grad(.1*raw_sigreg, state)[0]
    if not all(torch.isfinite(t).all() for t in (raw_dino, raw_sigreg, dino_grad, sigreg_grad)):
        raise ValueError("nonfinite diagnostic")
    geometry = gradient_geometry(dino_grad, sigreg_grad)
    # A normalized illustrative state perturbation, never an optimizer/model update.
    direction = -sigreg_grad / sigreg_grad.norm().clamp_min(1e-30)
    epsilon = 1e-3
    # Evaluate the small loss difference in float64 to avoid float32 cancellation.
    baseline_error = state.detach().double() - target.double()
    changed = .5*(baseline_error+epsilon*direction.double()).square().mean()
    baseline = .5*baseline_error.square().mean()
    geometry.update(seed=seed, raw_dino=float(raw_dino.detach()), raw_sigreg=float(raw_sigreg.detach()),
                    perturbation_l2=epsilon,
                    finite_difference_weighted_dino_derivative=float((changed-baseline)/epsilon))
    return geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = extract_observations(load_export(args.stage2))
    second = extract_observations(load_export(args.control))
    if first[0] != second[0] or not torch.equal(first[2], second[2]) or not torch.equal(first[3], second[3]):
        raise ValueError("matched state identities/targets/pairs differ")
    result = {"scope": "CPU leaf-state subset; unique adjacent pairs among exported successors; missing initial observations; not full training or parameter gradients",
              "observations": len(first[0]), "pairs": len(first[3]),
              "current_occurrences_detached": True, "slot_pooling": "mean", "dino_weight": .5,
              "sigreg_weight": .1, "num_proj": 1024, "knots": 17,
              "direction_sign": "positive derivative means SIGReg descent locally increases DINO loss",
              "models": {}}
    for name, states in (("stage2", first[1]), ("control", second[1]), ("dino_targets", first[2])):
        result["models"][name] = [evaluate(states, first[2], first[3], seed=seed) for seed in (42, 43, 44)]
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
