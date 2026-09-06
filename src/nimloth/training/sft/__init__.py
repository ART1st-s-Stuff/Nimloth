"""Three-stage supervised training and environment rollout evaluation.

stage1: response-format supervision; stage2: query/DINO alignment;
stage3: migrated historical SFT2 world-model/value training.
Import stage modules explicitly so CLI discovery does not load model runtimes.
"""
