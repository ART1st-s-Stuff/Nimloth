"""Update directory metadata in a copied cache; never modify its source."""
import json
import sys
from pathlib import Path


def relocate(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError('source and destination must differ')
    for path in destination.glob('*/manifest.json'):
        original = source / path.parent.name / path.name
        if path.is_symlink() or path.samefile(original):
            raise ValueError('manifest must be an independent copy')
        metadata = json.loads(path.read_text())
        if metadata != json.loads(original.read_text()):
            raise ValueError('copied manifest differs from source')
        if metadata['dir'] != str(original.parent):
            raise ValueError('source manifest directory mismatch')
        metadata['dir'] = str(path.parent)
        path.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    relocate(Path(sys.argv[1]), Path(sys.argv[2]))
