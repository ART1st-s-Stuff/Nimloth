from pathlib import Path
from build_rerun_contract import build
from prepare_test import source_base_model


def test_explicit_base_survives_transferred_checkpoint_and_preserves_budget():
    inputs = ['--model', '/old/base', '--train-jsonl', '/old/data/train.jsonl',
              '--val-jsonl', '/old/data/val.jsonl', '--dino-cache-root', '/old/dino_cache']
    old = {
        'root': '/old',
        'train_argv': inputs + ['--nproc-per-node=8', '-m', 'nimloth.training.sft.stage2',
            '--epochs', '2', '--grid-size', '8', '--distributed-strategy', 'fsdp',
            '--save-initial-checkpoint', '--output-dir', '/old/train',
            '--lr', '1e-6', '--embedding-lr', '5e-6'],
        'eval_argv_template': inputs + ['--grid-size', '8', '--nproc-per-node=8',
                                        '{checkpoint}', '{output}'],
        'total_seconds': 21600,
        'evaluation_reserve_seconds': 3600,
    }
    result = build(old, commit='abc', source=Path('/transferred/epoch_002'),
                   marker={'epoch': 2, 'step': 36}, root=Path('/new'), base=Path('/original/base'))
    assert source_base_model(result) == Path('/original/base')
    argv = result['train_argv']
    for flag, value in (('--lr', '5e-5'), ('--embedding-lr', '5e-5'),
                        ('--projector-lr', '1e-6'), ('--embedding-master-dtype', 'float32')):
        assert argv[argv.index(flag) + 1] == value
    assert result['total_seconds'] == old['total_seconds']
    assert result['evaluation_reserve_seconds'] == old['evaluation_reserve_seconds']
    assert result['input_root'] == old['root']
    assert result['source_marker'] == {'epoch': 2, 'step': 36}
