"""Create the authorized epoch2 FP32 embedding rerun contract; never launch."""
import argparse
import json
from pathlib import Path
from run_test import validate_contract


def build(old, *, commit, source, marker, root, base):
    result = json.loads(json.dumps(old))
    previous = result['root']
    result['root'] = str(root)
    result['commit'] = commit
    for key in ('train_argv', 'eval_argv_template'):
        result[key] = [x.replace(previous, str(root)) for x in result[key]]
    argv = result['train_argv']
    for flag in ('--lr', '--embedding-lr'):
        argv[argv.index(flag) + 1] = '5e-5'
    if '--embedding-master-dtype' in argv:
        argv[argv.index('--embedding-master-dtype') + 1] = 'float32'
    else:
        argv += ['--embedding-master-dtype', 'float32']
    if '--projector-lr' in argv:
        argv[argv.index('--projector-lr') + 1] = '1e-6'
    else:
        argv += ['--projector-lr', '1e-6']
    result.update(source_checkpoint=str(source), source_marker=marker,
                  source_base_model=str(base), input_root=previous,
                  reuse_legacy_contract=previous + '.contract.json')
    result.pop('reuse_input_audit', None)
    validate_contract(result)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('old-contract', 'source', 'root', 'base', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    marker = json.loads((a.source / 'COMMITTED').read_text())
    if marker.get('epoch') != 2:
        raise ValueError('expected Stage1 epoch2')
    result = build(json.loads(a.old_contract.read_text()), commit=a.commit,
                   source=a.source, marker=marker, root=a.root, base=a.base)
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
