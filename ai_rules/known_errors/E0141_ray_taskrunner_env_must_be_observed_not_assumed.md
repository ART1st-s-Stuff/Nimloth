# E0141: Ray TaskRunner environment must be observed, not assumed

## Error

ID189 retry4 exported `VAGEN_ROLLOUT_BROWSER_PACK_WORKERS=8` in the shell runner, but production TaskRunner logs reported `workers=1`. The variable did not reach the Ray TaskRunner environment. Static launcher checks incorrectly treated shell export as proof that worker parallelism was active.

## Rule

A Ray-owned runtime option is enabled only when the owning Ray process logs the effective value and a production-shaped runtime test observes it. Shell export alone is insufficient.

Pass required environment variables through the Ray runtime environment/configuration path used to create TaskRunner, then fail closed if the effective runtime value differs from the contract.

## Impact

Retry4 artifacts remain correct because worker count changes scheduling only. Exact binary transport reduced all three production batches to 35–46 seconds of packing each, so no repack is required.
