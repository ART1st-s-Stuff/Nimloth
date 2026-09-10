import json
import shutil
import tempfile
import unittest
from pathlib import Path

from relocate_cache import relocate


class RelocateTest(unittest.TestCase):
    def test_updates_only_copied_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'source', Path(tmp)/'dest'
            (source/'train').mkdir(parents=True)
            manifest = source/'train/manifest.json'
            manifest.write_text(json.dumps({'dir': str(source/'train'), 'count': 1}))
            original = manifest.read_bytes()
            (source/'train/sample.pt').write_bytes(b'unchanged payload')
            shutil.copytree(source, dest)
            relocate(source, dest)
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(json.loads((dest/'train/manifest.json').read_text())['dir'], str(dest/'train'))
            self.assertEqual((dest/'train/sample.pt').read_bytes(), b'unchanged payload')
            with self.assertRaises(ValueError):
                relocate(source, source)


if __name__ == '__main__':
    unittest.main()
