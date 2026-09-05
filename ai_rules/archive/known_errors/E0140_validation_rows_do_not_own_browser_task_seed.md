# E0140: Validation rows do not own Browser task seed

## Error

ID189 retry4 completed all 120 rollouts and all Browser artifacts, but its final gate read `row['seed']` from `validation/20.jsonl`. Those summary rows do not contain `seed`, so the post-run gate raised `KeyError: 'seed'` after data generation had completed.

## Rule

For Rollout Browser dataset coverage, read seed from each identity-aligned `rollout.json` (`record['seed']`). Validation JSONL may be used for row count, data-source count and `rollout_sample_id`, but fields absent from its schema must not be assumed.

The gate must prove that Browser and validation rollout ID sets are equal before using Browser seed coverage as evidence.

## Detection

Before production, test the finalizer against a representative validation row and Browser record, including explicit absence of `seed` in the validation row.
