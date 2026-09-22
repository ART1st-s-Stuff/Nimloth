"""Verify an already successful export and finish input proof; does not launch GPUs."""
import argparse
import json
import sys
from pathlib import Path

from prepare_test import digest, validate_manifest, verify_split_identities, source_base_model
from run_test import argument, validate_contract
from reuse_audit import reuse_legacy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    validate_contract(contract)
    checkout = Path(contract['checkout'])
    sys.path.insert(0, str(checkout / 'src'))
    from run_query_gate import validate_checkout
    validate_checkout(checkout, contract['commit'])
    root = Path(contract['root'])
    if (root / 'controller_events.jsonl').exists() or (root / 'train').exists():
        raise ValueError('training already started; not a preparation continuation')
    files = json.loads(args.manifest.read_text())
    validate_manifest(files)
    for entry in files:
        path = Path(entry['path'])
        if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError('input changed: ' + str(path))
    source = Path(contract['source_checkpoint'])
    if json.loads((source / 'COMMITTED').read_text()) != contract['source_marker']:
        raise ValueError('source identity changed')
    events = [json.loads(line) for line in (root / 'preparation_events.jsonl').read_text().splitlines()]
    exports = [e for e in events if e.get('phase') == 'export']
    if (len(exports) != 2 or exports[0]['event'] != 'phase_started'
            or exports[1]['event'] != 'phase_finished' or exports[1]['returncode'] != 0):
        raise ValueError('no unambiguous successful export')
    argv = exports[0]['argv']
    for flag, expected in (('--base-model', source_base_model(contract)),
                           ('--adapter-dir', source), ('--out-dir', root / 'base')):
        if Path(argument(argv, flag)).resolve() != expected.resolve():
            raise ValueError('successful export lineage differs: ' + flag)
    config = json.loads((root / 'base/config.json').read_text())
    if config.get('nimloth_embedding_master_dtype') != 'float32':
        raise ValueError('export does not declare FP32 embedding master')
    # Check actual storage, not just configuration. Export phase verified all adapter
    # tensors before merging; exact source embedding rows are retained after merge.
    import torch
    from safetensors import safe_open
    source_file = source / 'adapter_model.safetensors'
    with safe_open(source_file, framework='pt', device='cpu') as original:
        for suffix in ('model.embed_tokens.weight', 'lm_head.weight'):
            source_keys = [key for key in original.keys() if key.endswith(suffix)]
            if len(source_keys) != 1:
                raise ValueError('ambiguous source embedding ' + suffix)
            found = False
            for shard in (root / 'base').glob('*.safetensors'):
                with safe_open(shard, framework='pt', device='cpu') as exported:
                    for key in exported.keys():
                        if key.endswith(suffix):
                            actual = exported.get_tensor(key)
                            expected = original.get_tensor(source_keys[0])
                            if actual.dtype != torch.float32 or not torch.equal(actual, expected):
                                raise ValueError('exported embedding differs: ' + suffix)
                            found = True
            if not found:
                raise ValueError('exported embedding missing: ' + suffix)
    verify_split_identities(root, contract['expected_records'])
    options = dict(grid_size=8, query_count=64, max_length=20000, min_pixels=3136, max_pixels=100352)
    reuse_legacy(contract, files, options)
    print(json.dumps({'status': 'preparation_verified', 'root': str(root), 'gpu_started': False}))


if __name__ == '__main__':
    main()
