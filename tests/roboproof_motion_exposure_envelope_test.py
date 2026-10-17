import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from roboproof.motion_learning import exposure_envelope as envelope


ROOT = Path(__file__).resolve().parents[1]


PROTOCOL = json.loads((ROOT / 'tests/fixtures/motion-exposure-envelope-configs.json')
                      .read_text(encoding='utf-8-sig'))


def config(seed=136001):
    return {**copy.deepcopy(PROTOCOL['config']), 'seed': seed}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def member_tuple(content):
    return envelope.digest(content), len(content), f'{zlib.crc32(content):08x}'


def world(seed=10):
    return {'seed': seed, 'task': {'start': {'xIn': 0, 'yIn': 0, 'headingDeg': 0},
                                 'goal': {'xIn': 0, 'yIn': 12, 'headingDeg': 0}, 'deadlineSeconds': 10},
            'configuration': {}, 'stage': 1}


class ExposureEnvelopeTests(unittest.TestCase):
    def factory(self):
        return envelope.world_factory((ROOT / 'roboproof/motion_learning/train.py').read_bytes(),
                                      (ROOT / 'roboproof/motion_learning/td3_train.py').read_bytes())

    def synthetic_bundle(self, directory):
        runs = copy.deepcopy(envelope.ARCHIVED_RUNS)
        entries = []
        for run_id, expected in runs.items():
            config_name = ('controlConfig' if expected['role'] == 'control' else
                           'proofConfig' if expected['role'].startswith('proof-') else 'config')
            actual = copy.deepcopy(PROTOCOL[config_name])
            expected['nativeWorlds'] = 1
            folder = directory / expected['role']
            folder.mkdir()
            manifest = {'runId': run_id, 'status': 'completed', 'sources': PROTOCOL['sources'], 'config': actual,
                        'models': [{'seed': expected['seed'], 'updates': actual['updates'], 'episodes': 1}]}
            exposure = {'schemaVersion': 1, 'cases': [{**world(), 'kind': 'td3-collection'}]}
            manifest_content, exposure_content = encoded(manifest), encoded(exposure)
            (folder / 'run.json').write_bytes(manifest_content)
            (folder / 'exposure.json').write_bytes(exposure_content)
            expected['manifest'], expected['exposure'] = member_tuple(manifest_content), member_tuple(exposure_content)
            entries.append({'role': expected['role'], 'runId': run_id, 'seed': expected['seed'], 'config': actual,
                            'sources': {'train.py': envelope.TRAIN_SHA256, 'td3_train.py': envelope.TD3_SHA256},
                            'path': expected['role'] + '/exposure.json', 'sha256': expected['exposure'][0]})
        return runs, {'schemaVersion': 1, 'runs': entries}

    def test_frozen_budget_remains_14336_for_every_reserved_seed(self):
        for seed in envelope.FRESH_SEEDS:
            budget = envelope.envelope_budget(config(seed))
            self.assertEqual(budget['maximumGeneratedWorlds'], 14336)
            self.assertEqual(budget['nativeCollections'], 56)
            self.assertEqual(budget['nativeCurriculumOffset'], 128)
        self.assertEqual(envelope.envelope_budget(PROTOCOL['config'])['maximumGeneratedWorlds'], 14336)

    def test_unsupported_or_ambiguous_execution_is_not_admitted(self):
        for changes in [{'warmStartUpdates': True}, {'warmStartUpdates': 17}, {'warmStartUpdates': 64},
                        {'updates': 8}, {'updates': 63}, {'updates': 65}, {'seedCount': 2}, {'maxSteps': 13},
                        {'curriculum': 'other'}, {'expertDatasetDefinition': None}, {'onlineAnchorDefinition': {}},
                        {'onlineAnchorInterval': False}, {'onlineAnchorInterval': 1}, {'rolloutSteps': 1},
                        {'algorithm': 'ppo-beta-reference-v1'}, {'resume': True}, {'resume': None}, {'resume': False},
                        {'seed': True}, {'seed': 136004}, {'networkWidth': 256.0}, {'explorationStd': float('nan')}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                envelope.envelope_budget({**config(), **changes})

    def test_exact_corpus_definition_is_required(self):
        for changes in [{'worldCount': 127}, {'nativeCurriculumOffset': 129}, {'rowCount': 7857},
                        {'nativeCurriculumOffset': True}, {'method': 'other'}]:
            actual = config()
            actual['expertDatasetDefinition'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                envelope.envelope_budget(actual)
        actual = config()
        del actual['expertDatasetDefinition']['worldCount']
        with self.assertRaises(ValueError):
            envelope.envelope_budget(actual)

    def test_historical_control_and_proof_are_not_fresh_configurations(self):
        for name in ('controlConfig', 'proofConfig'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                envelope.envelope_budget(PROTOCOL[name])

    def test_prefix_rejects_wrong_seed_and_out_of_bound_counts_before_generation(self):
        def forbidden(*arguments):
            self.fail('Invalid prefix request reached the generator')
        for seed, count in [(136002, 0), (True, 0), (136004, 0), (201, 0),
                            (136001, True), (136001, -1), (136001, 14337)]:
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                list(envelope.generated_prefix(forbidden, config(), seed, count))

    def test_tiny_original_generator_prefix_is_deterministic_without_torch_or_physics(self):
        factory = self.factory()
        first = list(envelope.generated_prefix(factory, config(), 136001, 1))
        self.assertEqual(first, list(envelope.generated_prefix(factory, config(), 136001, 1)))
        self.assertNotEqual(first, list(envelope.generated_prefix(factory, config(136002), 136002, 1)))
        self.assertEqual(first[0]['stage'], 1)

    def test_changed_generator_or_control_source_is_not_executed(self):
        train = (ROOT / 'roboproof/motion_learning/train.py').read_bytes()
        td3 = (ROOT / 'roboproof/motion_learning/td3_train.py').read_bytes()
        for sources in [(train + b'\n', td3), (train, td3 + b'\n')]:
            with self.assertRaises(ValueError):
                envelope.world_factory(*sources)

    def test_prefix_check_retains_exact_task_physics_and_rejects_unknown_query_kinds(self):
        factory = lambda generator, ordinal, curriculum: world()
        rows = [{**world(), 'kind': 'td3-collection'}]
        result = envelope.verify_measured_prefix(factory, config(), 136001, rows)
        self.assertEqual(result['measuredNativeWorlds'], 1)
        changed = copy.deepcopy(rows)
        changed[0]['task']['goal']['xIn'] += 0.01
        with self.assertRaises(ValueError):
            envelope.verify_measured_prefix(factory, config(), 136001, changed)
        with self.assertRaises(ValueError):
            envelope.verify_measured_prefix(factory, config(), 136001, [{'kind': 'unreviewed'}])
        with self.assertRaises(ValueError):
            envelope.verify_measured_prefix(factory, config(), 136002, [])
        for name in ('controlConfig', 'proofConfig'):
            actual = PROTOCOL[name]
            self.assertTrue(envelope.verify_measured_prefix(factory, actual, actual['seed'], rows)['exactOriginalFieldsMatched'])

    def test_all_offline_worlds_and_original_order_are_required(self):
        corpus = [world(10), world(20)]
        cases = [{**entry, 'kind': 'td3-expert-corpus'} for entry in corpus]
        envelope.verify_offline_worlds(cases, corpus)
        for changed in (cases[:1], list(reversed(cases)), [{**cases[0], 'configuration': {'massKg': 10}}, cases[1]]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                envelope.verify_offline_worlds(changed, corpus)

    def test_archived_member_verification_checks_sha_size_and_crc(self):
        content = b'small synthetic archived evidence'
        binding = envelope.member_binding('pilot', 'synthetic.json', *member_tuple(content))
        self.assertEqual(envelope.verify_member(content, binding), content)
        for changes in ({'sha256': '0' * 64}, {'bytes': len(content) + 1}, {'crc32': '00000000'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                envelope.verify_member(content, {**binding, **changes})

    def test_self_consistent_replacement_declaration_and_caller_are_not_authenticated(self):
        caller = b'pass\n'
        replacement = encoded({'trialProtocolSha256': envelope.PROTOCOL_SHA256,
                               'seedOrder': list(envelope.FRESH_SEEDS), 'freshDriverSha256': envelope.digest(caller)})
        for name, content in [('final-stage-protocol.json', replacement), ('fresh-driver.py', caller)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                envelope.verify_member(content, envelope.ARCHIVED_MEMBERS[name])

    def test_four_distinct_authenticated_runs_can_have_identical_exposure_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            with patch.object(envelope, 'ARCHIVED_RUNS', runs):
                authenticated = envelope.authenticate_index(directory, index, PROTOCOL)
            self.assertEqual(len(authenticated), 4)
            self.assertEqual(len({record['entry']['runId'] for record in authenticated}), 4)
            self.assertEqual(len({record['entry']['sha256'] for record in authenticated}), 1)
            for record in authenticated:
                for binding in record['bindings'].values():
                    self.assertEqual(binding['archiveSha256'], envelope.ARCHIVES['pilot']['sha256'])

    def test_duplicate_one_world_audits_and_missing_roles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            duplicates = {'runs': [index['runs'][0], index['runs'][0], index['runs'][2], index['runs'][2]]}
            for changed in (duplicates, {'runs': index['runs'][:3]}):
                with patch.object(envelope, 'ARCHIVED_RUNS', runs), self.assertRaises(ValueError):
                    envelope.authenticate_index(directory, changed, PROTOCOL)

    def test_index_cannot_substitute_role_seed_source_configuration_or_member_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            for changes in ({'runId': 'unretained'}, {'role': 'control'}, {'seed': 1360052}, {'seed': True},
                            {'sources': {}}, {'sha256': '0' * 64}, {'config': config()},
                            {'path': '../exposure.json'}, {'path': index['runs'][1]['path']}):
                changed = copy.deepcopy(index)
                changed['runs'][0].update(changes)
                with self.subTest(changes=changes), patch.object(envelope, 'ARCHIVED_RUNS', runs), self.assertRaises(ValueError):
                    envelope.authenticate_index(directory, changed, PROTOCOL)

    def test_truncation_cannot_be_authenticated_by_rehashing_the_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            entry = index['runs'][0]
            content = encoded({'cases': []})
            (directory / entry['path']).write_bytes(content)
            with patch.object(envelope, 'ARCHIVED_RUNS', runs), self.assertRaises(ValueError):
                envelope.authenticate_index(directory, index, PROTOCOL)
            entry['sha256'] = envelope.digest(content)
            with patch.object(envelope, 'ARCHIVED_RUNS', runs), self.assertRaises(ValueError):
                envelope.authenticate_index(directory, index, PROTOCOL)

    def test_completed_status_and_all_declared_updates_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            entry = index['runs'][0]
            filename = (directory / entry['path']).with_name('run.json')
            original = json.loads(filename.read_bytes())
            for field, value in [('status', 'budget-stopped'), ('updates', 63), ('episodes', 0)]:
                manifest = copy.deepcopy(original)
                if field == 'status':
                    manifest[field] = value
                else:
                    manifest['models'][0][field] = value
                content = encoded(manifest)
                filename.write_bytes(content)
                changed_runs = copy.deepcopy(runs)
                changed_runs[entry['runId']]['manifest'] = member_tuple(content)
                with self.subTest(field=field), patch.object(envelope, 'ARCHIVED_RUNS', changed_runs), self.assertRaises(ValueError):
                    envelope.authenticate_index(directory, index, PROTOCOL)

    def test_nonempty_but_incomplete_native_exposure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            entry = index['runs'][0]
            filename = (directory / entry['path']).with_name('run.json')
            manifest = json.loads(filename.read_bytes())
            manifest['models'][0]['episodes'] = 2
            content = encoded(manifest)
            filename.write_bytes(content)
            runs[entry['runId']]['manifest'] = member_tuple(content)
            runs[entry['runId']]['nativeWorlds'] = 2
            with patch.object(envelope, 'ARCHIVED_RUNS', runs), self.assertRaises(ValueError):
                envelope.authenticate_index(directory, index, PROTOCOL)

    def test_each_index_entry_requires_its_authenticated_run_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs, index = self.synthetic_bundle(directory)
            (directory / index['runs'][0]['path']).with_name('run.json').unlink()
            with patch.object(envelope, 'ARCHIVED_RUNS', runs), self.assertRaises(ValueError):
                envelope.authenticate_index(directory, index, PROTOCOL)

    def test_full_build_remains_unavailable_on_the_local_windows_runtime(self):
        with patch.object(envelope.platform, 'system', return_value='Windows'), self.assertRaises(ValueError):
            envelope.build(Path('unused-bundle'), Path('unused-output'), envelope.PROTOCOL_SHA256)


if __name__ == '__main__':
    unittest.main()
