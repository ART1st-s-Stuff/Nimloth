# E0032: Nonempty JSONL is not proof of a complete rollout shard

## What happened

During SFT1 formal rollout ID9, VAGEN wrote `0.jsonl` incrementally in batches: the file was nonempty at 25, 50, 75, and 99 rows before reaching the required 120 rows. The wrapper's resume guard currently checks only `-s`, so an interrupted partial shard would be incorrectly skipped on restart.

## Required prevention

Before resuming this collection, validate each existing shard against its expected row count and verify all referenced images exist. Only a 120-row shard with complete images may be skipped. Isolate or remove a partial shard before restart.

Future wrapper refactoring should replace the nonempty check with a completeness validator or atomic finalization marker, but do not edit a wrapper while its current Slurm process is running.
