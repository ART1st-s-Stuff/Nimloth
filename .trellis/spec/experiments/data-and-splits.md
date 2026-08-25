# Data and Split Evidence

## Hard rules

- Verify split semantics from the actual dataset, configuration, loader code, manifest, or metadata. A name such as `all`, `eval_set`, `heldout`, a task category, or `train` is not evidence by itself.
- Training-data collection and rollout-train use only a verified training split.
- Generalization evaluation uses a validation/test/heldout split demonstrated not to overlap training data at the relevant unit (for example scenes, task identities, trajectories, or records).
- Record source/version, selection fields, seeds together with their scene/split context, conversion manifests, filtering, and overlap checks.
- Preserve the distinction among environment assets, project train/val/test partitions, and success subsets. A successful-training subset is not a generalization split.
- Validate the persisted rollout/data schema before conversion or training. Conversion may declare missing source semantics but must not invent CoT, hidden states, transitions, reward provenance, or action mappings.

If semantics are absent or cannot be verified, stop and ask the human. Do not continue based on a filename or previous run.

## Required evidence in the task/run record

- exact source paths/IDs and immutable version or hash where available;
- code/config/metadata lines that establish split meaning;
- overlap key and measured overlap result;
- transformations and their manifests/hashes;
- record counts before/after filtering and the statistical unit;
- limitations that prevent a claim from being generalized.

## Source ownership

- [`src/nimloth/rollout/README.md`](../../../src/nimloth/rollout/README.md): trajectory schema, validation, migration, windows, provenance.
- [`src/nimloth/environment/navigation/README.md`](../../../src/nimloth/environment/navigation/README.md): navigation environment adapter ownership.
- [`configs/training/`](../../../configs/training/) and [`configs/eval/`](../../../configs/eval/): reusable selection fields; values still require source verification.
