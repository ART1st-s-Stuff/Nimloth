# E0143: New Slurm scripts must set the project account

## Error

The first ID54 submission was rejected before allocation because its new Slurm script omitted `#SBATCH --account=peilab`. Current cluster policy requires an explicit account.

## Rule

Every new superpod Slurm script must declare `#SBATCH --account=peilab`, and its static launcher test must assert the account. A submission rejected before allocation does not consume experiment output or W&B identity, but the runtime commit must still be replaced after fixing the script.
