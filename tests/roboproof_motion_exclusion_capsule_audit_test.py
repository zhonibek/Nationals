import base64
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from roboproof.motion_learning.exclusion_capsule_audit import assert_archive_current, audit_capsule, decode_pairs, geometry_key, mapping_identity, round_javascript


class ExclusionCapsuleAuditTests(unittest.TestCase):
    def test_exact_javascript_half_rounding_and_negative_zero_key(self):
        for value, expected in [(0.5, 1), (-0.5, 0), (1.5, 2), (-1.5, -1),
                                (math.nextafter(0.5, 0), 0), (math.nextafter(-0.5, -1), -1)]:
            self.assertEqual(round_javascript(value), expected)
        with self.assertRaises(ValueError):
            round_javascript(True)

    def test_body_relative_geometry_and_heading_wrap(self):
        task = {'start': {'xIn': 0, 'yIn': 0, 'headingDeg': 0}, 'goal': {'xIn': 0, 'yIn': 24, 'headingDeg': 180}, 'deadlineSeconds': 10}
        self.assertEqual(geometry_key(task), hashlib.sha256(b'0:240:-180').hexdigest())
        rotated = {'start': {'xIn': 3, 'yIn': 4, 'headingDeg': 90}, 'goal': {'xIn': 27, 'yIn': 4, 'headingDeg': -90}, 'deadlineSeconds': 6}
        self.assertEqual(geometry_key(task), geometry_key(rotated))

    def test_invalid_task_does_not_create_a_geometry_exclusion(self):
        task = {'start': {'xIn': False, 'yIn': 0, 'headingDeg': 0}, 'goal': {'xIn': 0, 'yIn': 24, 'headingDeg': 0}, 'deadlineSeconds': 10}
        with self.assertRaises(ValueError):
            geometry_key(task)

    def test_canonical_paired_vectors_preserve_duplicate_order(self):
        seeds = struct.pack('>III', 7, 2, 7)
        geometries = bytes.fromhex('ab' * 32 + 'cd' * 32 + 'ab' * 32)
        vectors = {'encoding': 'uint32be-sha256-base64-v1', 'worldSeedCount': 3, 'bodyGeometryCount': 3,
                   'worldSeeds': base64.b64encode(seeds).decode(), 'bodyGeometries': base64.b64encode(geometries).decode()}
        self.assertEqual(decode_pairs(vectors, 3), [seeds, geometries])
        for changes in [{'worldSeedCount': True}, {'bodyGeometryCount': 2}, {'worldSeeds': vectors['worldSeeds'][:-4]},
                        {'worldSeeds': vectors['worldSeeds'] + '='}, {'extra': True}]:
            with self.assertRaises((ValueError, TypeError)):
                decode_pairs(dict(vectors, **changes), 3)

    def test_full_audit_refuses_local_execution_before_reading_inputs(self):
        with patch('roboproof.motion_learning.exclusion_capsule_audit.platform.system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'remote Linux CPython3.12.3'):
                audit_capsule(None, None, None, None)

    def test_mapping_helpers_cannot_self_certify_source_changes(self):
        root = Path(__file__).resolve().parents[1]
        geometry = (root / 'roboproof/motion-exposures.js').read_bytes()
        motion = (root / 'simulator/motion.js').read_bytes()
        self.assertEqual(mapping_identity(geometry, motion)['fixedDt'], 0.01)
        with self.assertRaisesRegex(ValueError, 'Immutable original'):
            mapping_identity(geometry, motion.replace(b'> 1e-9', b'> 1e-8'))

    @unittest.skipUnless(os.name == 'posix', 'Descriptor-relative pathname test requires remote POSIX')
    def test_archive_parent_path_replacement_is_rejected(self):
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        utility = SimpleNamespace(stamp=stamp, open_directory=lambda name: os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent_path = root / 'payload'
            parent_path.mkdir()
            filename = parent_path / 'archive.zip'
            filename.write_bytes(b'fixed tiny archive fixture')
            parent = utility.open_directory(parent_path)
            before, parent_before = filename.stat(), os.fstat(parent)
            try:
                assert_archive_current(utility, filename, parent, before, parent_before)
                parent_path.rename(root / 'retained')
                parent_path.mkdir()
                filename.write_bytes(b'replacement')
                with self.assertRaisesRegex(ValueError, 'parent'):
                    assert_archive_current(utility, filename, parent, before, parent_before)
            finally:
                os.close(parent)


if __name__ == '__main__':
    unittest.main()
