import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import types
import zipfile
import zlib

UTILITY_SHA = 'a86608be23a3f999e50bbc0f14321b6030c35b28f7ded7ed0dc2cc5b3804159b'
MANIFEST_SHA = '9e600ecea59aa2b9e42e0cf3ad9d0e65f038bb35ff52d9e219b016fc9e33b3a6'
AUDIT_SHA = '4513a338848ad79cbef65744053833ad69abcc25793ed7ed287edea95189413c'
PROOF_SHA = 'f1ce3e978c9624be35c8a24871859fb4e83062fac59065c9d36577359771715a'
ARCHIVE_SHA = 'd15aff07bb92d55e09f9da0924e182249fcd3fe43528da3fe869a81e45b76168'
GEOMETRY_MODULE_SHA = '0f52052f294a457ec4e27f40b2d51bd365cf85029f2ad90ca8e6f0c408e51dea'
BASE = 'roboproof/runs/motion/recovery/colab-expiry-20261016'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def round_javascript(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Finite rounding input required')
    lower = math.floor(value)
    return lower + int(value - lower >= 0.5)


def geometry_key(task):
    require(type(task) is dict and set(task) == {'start', 'goal', 'deadlineSeconds'}, 'Exact original task fields required')
    for pose in (task['start'], task['goal']):
        require(type(pose) is dict and set(pose) == {'xIn', 'yIn', 'headingDeg'}, 'Exact original pose required')
        for key, limit in [('xIn', 60), ('yIn', 60), ('headingDeg', 180)]:
            require(type(pose[key]) in (int, float) and math.isfinite(pose[key]) and -limit <= pose[key] <= limit, 'Bounded finite pose value required')
    deadline = task['deadlineSeconds']
    require(type(deadline) in (int, float) and math.isfinite(deadline) and 0.01 <= deadline <= 60 and
            abs(round_javascript(deadline / 0.01) * 0.01 - deadline) <= 1e-9, 'Original deadline contract required')
    start, goal = task['start'], task['goal']
    angle = start['headingDeg'] * math.pi / 180
    delta_x, delta_y = goal['xIn'] - start['xIn'], goal['yIn'] - start['yIn']
    heading = ((goal['headingDeg'] - start['headingDeg'] + 540) % 360) - 180
    coordinates = [round_javascript((delta_x * math.cos(angle) - delta_y * math.sin(angle)) * 10),
                   round_javascript((delta_x * math.sin(angle) + delta_y * math.cos(angle)) * 10), round_javascript(heading)]
    return sha(':'.join(str(value) for value in coordinates).encode())


def decode_pairs(vectors, count):
    require(type(count) is int and 0 <= count <= 200000 and type(vectors) is dict and set(vectors) == {
        'encoding', 'worldSeedCount', 'bodyGeometryCount', 'worldSeeds', 'bodyGeometries'}, 'Exact bounded vector fields required')
    require(vectors['encoding'] == 'uint32be-sha256-base64-v1' and type(vectors['worldSeedCount']) is int and
            type(vectors['bodyGeometryCount']) is int and vectors['worldSeedCount'] == vectors['bodyGeometryCount'] == count, 'Complete paired counts required')
    decoded = []
    for name, width in [('worldSeeds', 4), ('bodyGeometries', 32)]:
        encoded = vectors[name]
        require(type(encoded) is str and len(encoded) == 4 * math.ceil(count * width / 3), 'Exact encoded vector length required')
        content = base64.b64decode(encoded, validate=True)
        require(len(content) == count * width and base64.b64encode(content).decode() == encoded, 'Canonical complete vector bytes required')
        decoded.append(content)
    return decoded


def mapping_identity(geometry_source, motion_source):
    require(sha(geometry_source) == GEOMETRY_MODULE_SHA, 'Exact original mapping source required')
    source = geometry_source.decode().replace('\r\n', '\n')
    first = source.index('function geometryFingerprint(task) {')
    last = source.index('\n}', first) + 2
    geometry_function = source[first:last]
    source = motion_source.decode().replace('\r\n', '\n')
    require('const FIXED_DT = 0.01;' in source, 'Original fixed timestep required')
    first = source.index('  function keys(')
    last = source.index('  function configuration(', first)
    mapping = {'specification': 'original-body-relative-0.1inch-1degree-v1', 'fixedDt': 0.01,
               'geometryFunctionSha256': sha(geometry_function.encode()), 'normalizationHelpersSha256': sha(source[first:last].encode())}
    require(mapping['geometryFunctionSha256'] == '0f4f734ecb77fbe4ff27d8f25ead0ea45a0d7d30eb7d2fea30be3b62e6b97d56' and
            mapping['normalizationHelpersSha256'] == '788906db3e85a50a48dddec1bf2c62e5395232cbcf6843fffdf99a759f224cf1', 'Immutable original mapping identity required')
    return mapping


def assert_archive_current(utility, filename, parent, before, parent_before):
    require(utility.stamp(os.fstat(parent)) == utility.stamp(parent_before), 'Archive parent changed')
    current = utility.open_directory(filename.parent)
    try:
        require(utility.stamp(os.fstat(current)) == utility.stamp(parent_before), 'Archive parent pathname replaced')
        require(utility.stamp(os.stat(filename.name, dir_fd=current, follow_symlinks=False)) == utility.stamp(before), 'Archive pathname changed')
    finally:
        os.close(current)


def audit_capsule(project, capsule_file, archive_file, proof_file):
    require(platform.system() == 'Linux' and platform.python_implementation() == 'CPython' and platform.python_version() == '3.12.3',
            'Run the full capsule audit only on approved remote Linux CPython3.12.3')
    require(os.environ.get('ROBOPROOF_APPROVED_REMOTE') == '1', 'Explicit approved remote execution required')
    project = Path(project)
    utility_file = project / BASE / 'auditor.py'
    utility_bytes = utility_file.read_bytes()
    require(len(utility_bytes) == 37746 and sha(utility_bytes) == UTILITY_SHA, 'Pinned independent safe-reader source required')
    utility = types.ModuleType('pinned_exposure_safe_reader')
    utility.__file__ = str(utility_file)
    exec(compile(utility_bytes, str(utility_file), 'exec'), utility.__dict__)
    budget = utility.Budget()
    capsule_bytes = utility.read_single(capsule_file, budget)
    capsule = utility.json_value(capsule_bytes, budget)
    proof_bytes = utility.read_single(proof_file, budget)
    require(sha(proof_bytes) == PROOF_SHA, 'Exact prior full-data admission proof required')
    with utility.EvidenceTree(project, budget) as tree:
        audit_bytes = tree.read('roboproof/runs/motion/recovery/continuation-audit-20261017/exposure-audit.json')
        require(sha(audit_bytes) == AUDIT_SHA, 'Exact original independent envelope audit required')
        prior = utility.json_value(audit_bytes, budget)
        mapping = mapping_identity(tree.read('roboproof/motion-exposures.js'), tree.read('simulator/motion.js'))
        parents = {'envelopeArchiveSha256': ARCHIVE_SHA, 'envelopeArchiveBytes': 7583049,
                   'envelopeManifestSha256': MANIFEST_SHA, 'independentAuditSha256': AUDIT_SHA,
                   'sourcePackageSha256': 'c437521e2b95934baf6b2d8cf36f6089685428904844ef1f123e59b4e67949f4',
                   'fullAdmissionProofSha256': PROOF_SHA,
                   'fullAdmissionLedgerSha256': '6e27a99f17140c20c8ef91175c52b8760c1445f434f637fac5608724a9d3f2ab'}
        require(type(capsule) is dict and set(capsule) == {'schemaVersion', 'specification', 'recordCount', 'parents', 'mapping',
                'trainingSeeds', 'sources', 'potentialRecords', 'offlineRecords', 'actualQueriesInferred',
                'modelOrOutcomeRecoveryClaimed', 'qualificationClaimed', 'newPhysicsQueries', 'eventState', 'scope', 'vectors'}, 'Exact capsule fields required')
        utility.exact({key: capsule[key] for key in ['schemaVersion', 'specification', 'recordCount', 'parents', 'mapping', 'trainingSeeds',
                      'potentialRecords', 'offlineRecords', 'actualQueriesInferred', 'modelOrOutcomeRecoveryClaimed', 'qualificationClaimed', 'newPhysicsQueries', 'eventState']},
                      {'schemaVersion': 1, 'specification': 'audited-potential-exclusion-capsule-v1', 'recordCount': 43136,
                       'parents': parents, 'mapping': mapping, 'trainingSeeds': [136001, 136002, 136003], 'potentialRecords': 43008,
                       'offlineRecords': 128, 'actualQueriesInferred': False, 'modelOrOutcomeRecoveryClaimed': False,
                       'qualificationClaimed': False, 'newPhysicsQueries': 0, 'eventState': 'unresolved'}, 'Capsule metadata')
        require(capsule['scope'] == 'Lossless ordered exclusion-key projection only. Original full task/configuration bytes remain in the pinned archive. Not measured queries, recovered models/outcomes, complete historical geometry, an independent final or learned improvement.', 'Explicit projection-only scope required')
        seeds, geometries = decode_pairs(capsule['vectors'], 43136)
        archive_file = Path(archive_file)
        parent = utility.open_directory(archive_file.parent)
        parent_before = os.fstat(parent)
        descriptor = os.open(archive_file.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_size == 7583049, 'Exact bounded original binary archive required')
            budget.files += 1
            budget.bytes += before.st_size
            require(budget.bytes <= 128 * 1024 * 1024, 'Aggregate read budget exceeded')
            digest = hashlib.sha256()
            total = 0
            while block := os.read(descriptor, 1024 * 1024):
                total += len(block)
                require(total <= before.st_size, 'Archive expanded while reading')
                digest.update(block)
            require(total == before.st_size and digest.hexdigest() == ARCHIVE_SHA and utility.stamp(os.fstat(descriptor)) == utility.stamp(before), 'Original archive identity or stability mismatch')
            with os.fdopen(os.dup(descriptor), 'rb') as original, zipfile.ZipFile(original) as archive:
                require(archive.testzip() is None, 'Original retained archive CRC required')
                def member(name, metadata):
                    entry = archive.getinfo('envelope/' + name)
                    require(not entry.is_dir() and not entry.flag_bits & 1 and entry.file_size == metadata['bytes'], 'Exact original envelope member required')
                    budget.charge_file(entry.file_size)
                    with archive.open(entry) as stream:
                        content = stream.read(4 * 1024 * 1024 + 1)
                    require(len(content) == entry.file_size and sha(content) == metadata['sha256'] and zlib.crc32(content) == entry.CRC, 'Every original member byte must verify')
                    return utility.json_value(content, budget)
                manifest = member('envelope-manifest.json', prior['envelopeFiles']['envelope-manifest.json'])
                require(prior['envelopeManifest']['sha256'] == MANIFEST_SHA, 'Original manifest binding required')
                expected_sources = [dict(shard, role='potential') for shard in manifest['shards']] + [dict(manifest['offlineCorpus'], role='offline')]
                utility.exact(capsule['sources'], expected_sources, 'Complete original source order')
                index = 0
                potential = offline = 0
                for row in capsule['sources']:
                    value = member(row['path'], prior['envelopeFiles'][row['path']])
                    require(type(value['cases']) is list and len(value['cases']) == row['count'], 'Complete original records required')
                    for world in value['cases']:
                        budget.charge_world()
                        require(type(world['seed']) is int and 0 <= world['seed'] <= 0xffffffff, 'Original uint32 seed required')
                        require(struct.unpack_from('>I', seeds, index * 4)[0] == world['seed'], 'Ordered seed mismatch')
                        require(geometries[index * 32:(index + 1) * 32].hex() == geometry_key(world['task']), 'Independently computed ordered geometry mismatch')
                        index += 1
                        if row['role'] == 'potential':
                            potential += 1
                        else:
                            offline += 1
                require((index, potential, offline) == (43136, 43008, 128), 'Every potential and offline pair required')
            require(utility.stamp(os.fstat(descriptor)) == utility.stamp(before) and
                    utility.stamp(os.stat(archive_file.name, dir_fd=parent, follow_symlinks=False)) == utility.stamp(before), 'Archive changed after audit')
            assert_archive_current(utility, archive_file, parent, before, parent_before)
        finally:
            os.close(descriptor)
            os.close(parent)
        tree.assert_current()
    require(sha(utility.read_single(capsule_file, budget)) == sha(capsule_bytes), 'Capsule changed during audit')
    return {'schemaVersion': 1, 'specification': 'independent-exclusion-capsule-audit-v1', 'verified': True,
            'capsuleSha256': sha(capsule_bytes), 'capsuleBytes': len(capsule_bytes), 'archiveSha256': ARCHIVE_SHA,
            'archiveBytes': 7583049, 'originalManifestSha256': MANIFEST_SHA, 'originalAuditSha256': AUDIT_SHA,
            'parentAdmissionProofSha256': PROOF_SHA, 'auditorSha256': sha(Path(__file__).read_bytes()), 'safeReaderSha256': UTILITY_SHA,
            'mapping': mapping, 'orderedPairsIndependentlyCompared': 43136, 'potentialRecords': 43008, 'offlineRecords': 128,
            'orderAndDuplicatesPreserved': True, 'newPhysicsQueries': 0, 'modelOrOutcomeRecoveryClaimed': False,
            'mappingEquivalenceScope': 'Every exact archived record was compared; universal cross-runtime trigonometric rounding parity is not claimed',
            'qualificationClaimed': False, 'eventState': 'unresolved', 'runtime': {'system': platform.system(), 'python': platform.python_version(), 'implementation': platform.python_implementation()},
            'readBytes': budget.bytes, 'sourceRecordsCompared': budget.worlds,
            'scope': 'Independent Python projection audit against every original archived record, not a replacement for full source data, recovered models, training or an independent final'}


def main():
    parser = argparse.ArgumentParser()
    for name in ['project', 'capsule', 'archive', 'proof', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_capsule(arguments.project, arguments.capsule, arguments.archive, arguments.proof)
    require(arguments.output.resolve() not in [arguments.capsule.resolve(), arguments.archive.resolve(), arguments.proof.resolve()], 'New audit output required')
    with arguments.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({'verified': True, 'audit': arguments.output.name, 'capsuleSha256': result['capsuleSha256'],
                      'orderedPairsIndependentlyCompared': result['orderedPairsIndependentlyCompared'], 'modelTrainingPerformed': False}))


if __name__ == '__main__':
    main()
