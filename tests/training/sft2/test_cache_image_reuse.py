"""Immutable image reuse rejects mismatches before linking source storage."""
import json

import pytest
import torch

from nimloth.util.cache.image_reuse import validate_image_reuse, link_verified_images, file_sha256
from nimloth.util.cache.schema import COMPACT_CACHE_FORMAT


def source_cache(tmp_path):
    root = tmp_path / 'old'
    (root / 'images').mkdir(parents=True)
    (root / 'transitions').mkdir()
    (root / 'transitions/old.pt').write_bytes(b'old-text-must-not-be-reused')
    manifest = dict(format=COMPACT_CACHE_FORMAT, base_fingerprint='old-base',
                    image_source_fingerprint='image-fingerprint', image_dtype='bfloat16',
                    max_pixels=100352, min_pixels=3136, image_shard_size=2,
                    image_shards=1, unique_images=2)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    rows = [dict(path=f'/image-{i}', shard=0, index=i, grid_thw=[1, 2, 2]) for i in range(2)]
    (root / 'image_index.json').write_text(json.dumps(dict(format=COMPACT_CACHE_FORMAT, images=rows)))
    torch.save(dict(pixel_values=torch.zeros(8, 12, dtype=torch.bfloat16),
                    image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
                    offsets=torch.tensor([0, 4, 8])), root / 'images/shard_00000.pt')
    kwargs = dict(paths=[r['path'] for r in rows], source_fingerprint='image-fingerprint',
                  base_fingerprint='old-base', image_dtype='bfloat16', max_pixels=100352,
                  min_pixels=3136, image_shard_size=2, pixel_width=12, merge_size=2)
    return root, kwargs


def test_links_only_images_and_leaves_old_bytes_unchanged(tmp_path):
    root, kwargs = source_cache(tmp_path)
    before = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()}
    identity = validate_image_reuse(root, **kwargs)
    dest = tmp_path / 'new'
    link_verified_images(identity, dest)
    link_verified_images(identity, dest)  # Exact tracked-image resume is idempotent.
    assert (dest / 'images/shard_00000.pt').samefile(root / 'images/shard_00000.pt')
    assert not (dest / 'transitions').exists() and not (dest / 'manifest.json').exists()
    assert {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('key,value', [('image_dtype','float32'), ('max_pixels',42),
                                       ('min_pixels',1), ('source_fingerprint','other'),
                                       ('base_fingerprint','other'), ('image_shard_size',1),
                                       ('paths',['/image-1','/image-0']), ('pixel_width',13)])
def test_rejects_source_contract_mismatches(tmp_path, key, value):
    root, kwargs = source_cache(tmp_path)
    kwargs[key] = value
    with pytest.raises(ValueError):
        validate_image_reuse(root, **kwargs)


@pytest.mark.parametrize('field', ['grid', 'offset', 'index', 'unfinished'])
def test_rejects_inconsistent_shards_or_index(tmp_path, field):
    root, kwargs = source_cache(tmp_path)
    if field == 'unfinished':
        (root / 'build_state.json').write_text('{}')
    elif field == 'index':
        path = root / 'image_index.json'
        payload = json.loads(path.read_text())
        payload['images'][0]['grid_thw'] = [1, 4, 4]
        path.write_text(json.dumps(payload))
    else:
        path = root / 'images/shard_00000.pt'
        payload = torch.load(path, weights_only=True)
        if field == 'grid':
            payload['image_grid_thw'][0] = torch.tensor([1, 3, 2])
        else:
            payload['offsets'][1] = 3
        torch.save(payload, path)
    with pytest.raises(ValueError):
        validate_image_reuse(root, **kwargs)


def test_rejects_source_change_after_validation(tmp_path):
    root, kwargs = source_cache(tmp_path)
    identity = validate_image_reuse(root, **kwargs)
    path = root / 'images/shard_00000.pt'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed after verification'):
        link_verified_images(identity, tmp_path / 'new')
