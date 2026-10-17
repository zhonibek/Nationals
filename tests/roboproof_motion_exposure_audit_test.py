import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from roboproof.motion_learning import exposure_audit as audit


class IndependentExposureAuditTests(unittest.TestCase):
    def test_small_bound_is_independently_computed(self):
        config = {'updates': 64, 'warmStartUpdates': 8, 'rolloutSteps': 256, 'maxSteps': 50000,
                  'expertDatasetDefinition': {'nativeCurriculumOffset': 128}}
        self.assertEqual(audit.bound(config), {'nativeCollections': 56, 'rolloutSteps': 256,
            'maxSteps': 50000, 'nativeCurriculumOffset': 128, 'maximumGeneratedWorlds': 14336})
        for changes in ({'updates': True}, {'warmStartUpdates': 65}, {'rolloutSteps': 257}, {'maxSteps': 0}, {'resume': False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit.bound({**config, **changes})

    def test_duplicate_nonfinite_and_deep_json_fail_closed(self):
        for content in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'[' * 34 + b'0' + b']' * 34):
            with self.subTest(content=content), self.assertRaises(ValueError):
                audit.json_value(content, audit.Budget())

    def test_exact_metadata_preserves_types_and_rejects_missing_fields(self):
        expected = {'worlds': [1, 2], 'verified': False, 'value': 1.0}
        audit.exact(copy.deepcopy(expected), expected, 'fixture')
        for actual in ({'worlds': [True, 2], 'verified': False, 'value': 1.0},
                       {'worlds': [1, 2], 'verified': 0, 'value': 1.0},
                       {'worlds': [1, 2], 'verified': False, 'value': 1}, {'worlds': [1, 2]}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                audit.exact(actual, expected, 'fixture')

    def test_unsafe_evidence_paths_are_not_accepted(self):
        self.assertEqual(audit.relative_parts('seed-136001-part-00/exposure.json'),
                         ['seed-136001-part-00', 'exposure.json'])
        for relative in ('../outside', '/root/file', 'C:/outside', 'folder\\file', 'folder//file', './file', 'a\x00b'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                audit.relative_parts(relative)

    def test_catalog_requires_every_ordinal_for_all_three_reserved_seeds(self):
        catalog = audit.expected_shards(14336)
        self.assertEqual(len(catalog), 45)
        for seed in audit.SEEDS:
            rows = [row for row in catalog if row['trainingSeed'] == seed]
            self.assertEqual(sum(row['count'] for row in rows), 14336)
            self.assertEqual(rows[-1]['firstOrdinal'], 14000)
            self.assertEqual(rows[-1]['count'], 336)

    def test_full_audit_cannot_run_on_the_laptop(self):
        with patch.object(audit.platform, 'system', return_value='Windows'), self.assertRaises(ValueError):
            audit.audit(Path('unused.zip'), Path('unused-bundle'), Path('unused-envelope'))


if __name__ == '__main__':
    unittest.main()
