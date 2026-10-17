import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import stat
import sys
import zipfile
import zlib


METHOD = 'td3-full-corpus-prefix-envelope-v1'
AUDIT_METHOD = 'independent-td3-prefix-envelope-audit-v1'
SEEDS = (136001, 136002, 136003)
WORLD_FIELDS = ('seed', 'task', 'configuration', 'stage')
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_AUDIT_BYTES = 256 * 1024
SOURCE_PACKAGE_SHA256 = 'c437521e2b95934baf6b2d8cf36f6089685428904844ef1f123e59b4e67949f4'
SOURCE_PACKAGE_BYTES = 205248
BUILDER_SHA256 = '6f6078fc1251bad75e2e9754c16e4197d2edde64e648ea771bbe29d096613ffb'
TRAIN_SHA256 = 'a82599abfba521c07a711649a0aa941e9a5c0367da33d85aff8ad103c247560e'
TD3_SHA256 = '741e11263b766e857a3a9540f17a25eefc08a6997406c750215eb5da1c295314'
PROTOCOL_SHA256 = 'a8b167033c3e22723756ea534c9399a431b33e5766bea1e89c4f72feef6eede8'
DECLARATION_SHA256 = '7b287c2726ecb28e7a50a7a3b6a6a67fe43fd1922592183561bd532867d44cc0'
CORPUS_SHA256 = 'e353cbc86f660fe1b8d0e250ce6bb0733317452f315108b4ef4dff42aabf3c5b'
CONFIG_HASHES = {
    'config': 'd8eb17caca550c51bdf77249c4e4e900750b7f9b605485c90bdc1d218ee645ae',
    'controlConfig': 'fd4a09dda3828b34d3bc6c6d5591af7aa9573d61031fdb9d9a9f4ee1f1f0498d',
    'proofConfig': '7d402e135595dbee497648db896b1d5720d326be402b58daef99ddb3227595f0',
}
ARCHIVES = {
    'pilot': {'name': 'episode-balanced-colab-pilot-20261016.zip',
              'sha256': 'c53261df5f2a049f36b7018973e801357c328b67009264f633eb08fcfa6cf922'},
    'fresh-launch': {'name': 'episode-balanced-colab-fresh-launch-20261016.zip',
                     'sha256': '2f361574b2eda463512bd3ecb091b54fd0d4b9aff0e1f255849830c58f23daee'},
}
STUDY_MEMBER = 'roboproof/runs/motion/research/episode-balanced-remote-study-20261016/'
RUNS = (
    ('67cbb420-fade-4208-91a3-1da837ea397b', 'pilot', 'candidate', 201, 'config', 130,
     ('f326e7474b5fae162f990f910c15bcd6609c6526a24f046972eb7f8018a614a8', 120638, '47312d8a'),
     ('00786cafd9bcdc5f96b1b26e90443ad17f5803d7ff46aa0f5265f1cc0cb86790', 108324, 'eb16385d')),
    ('484dbd88-2d67-4dd2-a9b7-0ccab8810337', 'control', 'control', 201, 'controlConfig', 125,
     ('069ef7eab242a5fded4bd87141864a2b6945d55438cb768e3a9e148099d2f04d', 116092, '0a34d216'),
     ('da28b62dd23969c0e0a81185393f3931fc17dcbf92bad2d07ef6674c428e7d6b', 106323, '16a3eacb')),
    ('edb7296a-9189-41ad-8864-19edcea0f813', 'proof-complete', 'proof-complete', 1360052, 'proofConfig', 3,
     ('a6b014d85513a31f0d749d0a5594be12713ff1d87d38bb4aa0129ffe6cfefbd9', 20961, 'd4e91ef6'),
     ('2b5a72d5a065734f278edb980da436ff75e29f2a2e2b601134997c1b2f851218', 59221, '8b6eaa97')),
    ('3bc26c9c-04ca-4be0-931f-3e3b2f2ef651', 'proof-resumed', 'proof-resumed', 1360052, 'proofConfig', 3,
     ('8fe2582e7ec4d687c55c8c296161f559fc45999bbacb07bbd53a5e278ce5f0c0', 20961, '0dcc31e8'),
     ('2b5a72d5a065734f278edb980da436ff75e29f2a2e2b601134997c1b2f851218', 59221, '8b6eaa97')),
)
BUILDER_SCOPE = ('Conservative exclusion metadata only. Original task-generator functions and Python task RNG '
                 'are replayed without training, simulator steps or actor/model recovery. Abort-only '
                 'frozen/consumed filters are omitted to include a superset, not to query forbidden worlds. '
                 'Old campaign outcomes remain unavailable.')
PREFIX_SCOPE = 'Matches actual retained task/physics metadata, not model states or newly simulated outcomes'
INDEX_SCOPE = ('Derived index checked against four distinct pinned archived completed run manifests and '
               'complete exposure members; not self-authenticated index assertions.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def exact(actual, expected, label):
    require(type(actual) is type(expected), f'{label}: JSON type mismatch')
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f'{label}: exact object fields required')
        for name in expected:
            exact(actual[name], expected[name], f'{label}.{name}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{label}: exact array length required')
        for ordinal, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, f'{label}[{ordinal}]')
    else:
        require(actual == expected, f'{label}: value mismatch')


def integer(value, minimum, maximum, label):
    require(type(value) is int and minimum <= value <= maximum, f'{label}: bounded integer required')
    return value


def relative_parts(value):
    require(type(value) is str and 0 < len(value) <= 512 and
            not any(character in value for character in ('\\', ':', '\x00')), 'Unsafe evidence path')
    parts = value.split('/')
    require(all(part and part not in ('.', '..') for part in parts) and len(parts) <= 7 and
            not PurePosixPath(value).is_absolute(), 'Canonical contained evidence path required')
    return parts


def file_record(content):
    return {'sha256': sha256(content), 'bytes': len(content), 'crc32': f'{zlib.crc32(content):08x}'}


def stamp(details):
    return (details.st_dev, details.st_ino, details.st_mode, details.st_size,
            details.st_mtime_ns, details.st_ctime_ns)


def open_directory(filename):
    filename = Path(os.path.abspath(filename))
    descriptor = os.open(filename.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in filename.parts[1:]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class Budget:
    def __init__(self):
        self.files = 0
        self.bytes = 0
        self.entries = 0
        self.nodes = 0
        self.worlds = 0

    def charge_file(self, size):
        integer(size, 0, MAX_FILE_BYTES, 'Evidence bytes')
        self.files += 1
        self.bytes += size
        require(self.files <= 2048 and self.bytes <= MAX_TOTAL_BYTES, 'Aggregate evidence read limit exceeded')

    def charge_entry(self):
        self.entries += 1
        require(self.entries <= 32768, 'Aggregate directory entry limit exceeded')

    def charge_world(self):
        self.worlds += 1
        require(self.worlds <= 200000, 'Aggregate world comparison limit exceeded')


class EvidenceTree:
    def __init__(self, directory, budget):
        self.path = Path(os.path.abspath(directory))
        self.descriptor = open_directory(self.path)
        self.identity = os.fstat(self.descriptor)
        self.budget = budget
        self.records = {}
        self.stamps = {}

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        os.close(self.descriptor)

    def parent(self, relative):
        parts = relative_parts(relative)
        descriptor = os.dup(self.descriptor)
        try:
            for part in parts[:-1]:
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = following
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def read(self, relative):
        require(relative not in self.records, 'Evidence must be read and charged once per path')
        parent, name = self.parent(relative)
        descriptor = None
        try:
            initial = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISREG(initial.st_mode), 'Regular non-symlink evidence file required')
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            before = os.fstat(descriptor)
            require(stamp(initial) == stamp(before) and stat.S_ISREG(before.st_mode), 'Evidence replaced before read')
            self.budget.charge_file(before.st_size)
            remaining = before.st_size + 1
            blocks = []
            while remaining:
                block = os.read(descriptor, min(remaining, 1024 * 1024))
                if not block:
                    break
                blocks.append(block)
                remaining -= len(block)
            content = b''.join(blocks)
            require(len(content) == before.st_size and stamp(os.fstat(descriptor)) == stamp(before) and
                    stamp(os.stat(name, dir_fd=parent, follow_symlinks=False)) == stamp(before),
                    'Evidence changed during read')
            self.records[relative] = file_record(content)
            self.stamps[relative] = stamp(before)
            return content
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def inventory(self, expected):
        expected = set(expected)
        directories = {''}
        for relative in expected:
            parts = relative_parts(relative)
            directories.update('/'.join(parts[:length]) for length in range(1, len(parts)))
        found = set()

        def walk(descriptor, prefix):
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    self.budget.charge_entry()
                    relative = f'{prefix}/{entry.name}' if prefix else entry.name
                    relative_parts(relative)
                    details = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(details.st_mode):
                        require(relative in directories, 'Unexpected evidence directory, including staging/.scan data')
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                        try:
                            require(stamp(os.fstat(child)) == stamp(details), 'Evidence directory replaced')
                            walk(child, relative)
                        finally:
                            os.close(child)
                    else:
                        require(stat.S_ISREG(details.st_mode) and relative in expected,
                                'Unexpected, nonregular or symbolic evidence entry')
                        found.add(relative)

        walk(self.descriptor, '')
        require(found == expected, 'Missing evidence files in exact inventory')

    def assert_current(self):
        current = open_directory(self.path)
        try:
            require(stamp(os.fstat(current)) == stamp(self.identity), 'Evidence root changed during audit')
        finally:
            os.close(current)
        for relative, expected in self.stamps.items():
            parent, name = self.parent(relative)
            try:
                require(stamp(os.stat(name, dir_fd=parent, follow_symlinks=False)) == expected,
                        'Evidence changed after verification')
            finally:
                os.close(parent)


def read_single(filename, budget):
    filename = Path(os.path.abspath(filename))
    with EvidenceTree(filename.parent, budget) as tree:
        content = tree.read(filename.name)
        tree.assert_current()
    return content


def json_value(content, budget):
    def pairs(entries):
        result = {}
        for name, value in entries:
            require(name not in result, 'Duplicate JSON object key')
            result[name] = value
        return result

    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')

    value = json.loads(content.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject_constant)
    pending = [(value, 0)]
    nodes = 0
    while pending:
        entry, depth = pending.pop()
        nodes += 1
        budget.nodes += 1
        require(nodes <= 100000 and budget.nodes <= 4000000 and depth <= 32, 'JSON structure budget exceeded')
        if type(entry) is dict:
            pending.extend((child, depth + 1) for child in entry.values())
        elif type(entry) is list:
            pending.extend((child, depth + 1) for child in entry)
        elif type(entry) is float:
            require(math.isfinite(entry), 'Finite JSON numbers required')
    return value


def member_binding(archive, member, metadata):
    checksum, size, crc32 = metadata
    return {'archiveSha256': ARCHIVES[archive]['sha256'], 'member': member,
            'sha256': checksum, 'bytes': size, 'crc32': crc32}


def archived_bindings():
    bindings = {
        'trial-protocol.json': member_binding('pilot', STUDY_MEMBER + 'trial-protocol.json',
            (PROTOCOL_SHA256, 128732, '03e7376a')),
        'final-stage-protocol.json': member_binding('fresh-launch', STUDY_MEMBER + 'final-stage-protocol.json',
            (DECLARATION_SHA256, 1582, '8a241cc8')),
        'study-driver.py': member_binding('pilot', '.cache/remote-episode-replay-study-20261016.py',
            ('d979c77ab90c060186a649e7ec07e32d72ae62b455582f930f75b5efda4eb418', 14976, '9f10d3d3')),
        'fresh-driver.py': member_binding('fresh-launch', '.cache/episode-balanced-colab-fresh-20261016.py',
            ('44b49c8cf25a130f64ec62714520db4f6af490beb0906a6cc05fef0c9683fc03', 4763, '2426bc09')),
    }
    for run_id, role, folder, seed, config_name, count, manifest, exposure in RUNS:
        base = STUDY_MEMBER + folder + '/' + run_id + '/'
        bindings[role + '/run.json'] = member_binding('pilot', base + 'run.json', manifest)
        bindings[role + '/exposure.json'] = member_binding('pilot', base + f'seed-{seed}/exposure.json', exposure)
    return bindings


def verify_source_package(filename, budget, bindings):
    content = read_single(filename, budget)
    require(len(content) == SOURCE_PACKAGE_BYTES and sha256(content) == SOURCE_PACKAGE_SHA256,
            'Exact bound 20-file 205248-byte source ZIP required')
    bundle_names = set(bindings) | {'train.py', 'td3_train.py', 'measured-index.json', 'corpus-worlds.json'}
    expected = {'bundle/' + name for name in bundle_names} | {
        'envelope.py', 'capacity.js', 'motion-exposure-codec.js', 'motion-holdout.js'}
    members = {}
    records = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist()
        names = [entry.filename for entry in infos]
        require(len(names) == 21 and len(set(names)) == 21 and set(names) == expected | {'source-manifest.json'},
                'Exact unique source ZIP inventory required')
        for entry in infos:
            relative_parts(entry.filename)
            require(not entry.is_dir() and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG) and
                    not entry.flag_bits & 1, 'Regular unencrypted source ZIP members required')
            budget.charge_file(entry.file_size)
            with archive.open(entry) as source:
                data = source.read(MAX_FILE_BYTES + 1)
            require(len(data) == entry.file_size and zlib.crc32(data) == entry.CRC, 'Source ZIP size/CRC mismatch')
            members[entry.filename] = data
            records[entry.filename] = file_record(data)
    manifest = json_value(members['source-manifest.json'], budget)
    require(type(manifest) is dict and manifest.get('schemaVersion') == 1 and
            type(manifest.get('files')) is list and len(manifest['files']) == 20, 'Complete source manifest required')
    for name, row in zip(sorted(expected), manifest['files']):
        exact(row, {'path': name, 'sha256': records[name]['sha256'], 'bytes': records[name]['bytes']},
              'Source package member')
    require(records['envelope.py']['sha256'] == BUILDER_SHA256, 'Pinned remote builder source identity required')
    for name, binding in bindings.items():
        exact(records['bundle/' + name], {key: binding[key] for key in ('sha256', 'bytes', 'crc32')},
              f'Original archive member {name}')
    return members, records, bundle_names


def config_hash(config):
    require(type(config) is dict and type(config.get('seed')) is int, 'Exact integer-seeded configuration required')
    return sha256(canonical({name: value for name, value in config.items() if name != 'seed'}))


def bound(config):
    require(type(config) is dict and 'resume' not in config, 'Fresh invocation without resume required')
    updates = integer(config.get('updates'), 1, 64, 'Updates')
    warm = integer(config.get('warmStartUpdates'), 1, updates, 'Warm updates')
    rollout = integer(config.get('rolloutSteps'), 1, 256, 'Rollout steps')
    max_steps = integer(config.get('maxSteps'), 1, 50000, 'Maximum actual steps')
    expert = config.get('expertDatasetDefinition')
    require(type(expert) is dict, 'Pinned offline corpus definition required')
    offset = integer(expert.get('nativeCurriculumOffset'), 0, 128, 'Native curriculum offset')
    collections = updates - warm
    return {'nativeCollections': collections, 'rolloutSteps': rollout, 'maxSteps': max_steps,
            'nativeCurriculumOffset': offset, 'maximumGeneratedWorlds': min(collections * rollout, max_steps)}


def original_factory(train_content, td3_content):
    train_source = train_content.decode('utf-8-sig').replace('\r\n', '\n')
    td3_source = td3_content.decode('utf-8-sig').replace('\r\n', '\n')
    require(sha256(train_source.encode()) == TRAIN_SHA256 and sha256(td3_source.encode()) == TD3_SHA256,
            'Pinned original task-generator and collection-control sources required')
    declarations = {}
    functions = {}
    for node in ast.parse(train_source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('CURRICULUM', 'CURRICULA'):
                require(name not in declarations, 'Duplicate original curriculum declaration')
                declarations[name] = node
        elif isinstance(node, ast.FunctionDef) and node.name in ('training_world', 'full_reach_world'):
            require(node.name not in functions, 'Duplicate original task-generator function')
            functions[node.name] = node
    require(set(declarations) == {'CURRICULUM', 'CURRICULA'} and
            set(functions) == {'training_world', 'full_reach_world'}, 'Exact original AST entry points required')
    nodes = [declarations['CURRICULUM'], declarations['CURRICULA'],
             functions['full_reach_world'], functions['training_world']]
    namespace = {'__builtins__': {'ValueError': ValueError}, 'math': math,
                 'exclude_frozen_world': lambda world: world}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned-original-training-world', 'exec'), namespace)
    return namespace['training_world']


def compare_world(actual, expected, label, budget):
    require(type(actual) is dict and type(expected) is dict and
            all(name in actual and name in expected for name in WORLD_FIELDS), f'{label}: complete world required')
    integer(actual['seed'], 0, 0xffffffff, f'{label}.seed')
    integer(actual['stage'], 0, 2, f'{label}.stage')
    for name in WORLD_FIELDS:
        exact(actual[name], expected[name], f'{label}.{name}')
    budget.charge_world()


def audit_measured(values, protocol, corpus, factory, budget):
    index = values['measured-index.json']
    require(type(index) is dict, 'Measured index object required')
    exact(sorted(index), ['runs', 'schemaVersion'], 'Measured index fields')
    exact(index['schemaVersion'], 1, 'Measured index schema')
    require(type(index['runs']) is list and len(index['runs']) == 4, 'All four distinct retained runs required')
    results = []
    for entry, record in zip(index['runs'], RUNS):
        run_id, role, folder, seed, config_name, count, manifest_binding, exposure_binding = record
        config = protocol[config_name]
        exact(entry, {'role': role, 'runId': run_id, 'seed': seed, 'config': config,
                      'sources': {'train.py': TRAIN_SHA256, 'td3_train.py': TD3_SHA256},
                      'path': role + '/exposure.json', 'sha256': exposure_binding[0]}, f'{role} index')
        manifest = values[role + '/run.json']
        require(type(manifest) is dict, 'Archived run manifest object required')
        exact(manifest.get('runId'), run_id, f'{role} run ID')
        exact(manifest.get('status'), 'completed', f'{role} terminal state')
        exact(manifest.get('sources'), protocol['sources'], f'{role} sources')
        exact(manifest.get('config'), config, f'{role} configuration')
        models = manifest.get('models')
        require(type(models) is list and len(models) == 1 and type(models[0]) is dict, 'One archived seed model required')
        for name, expected in [('seed', seed), ('updates', config['updates']), ('episodes', count)]:
            exact(models[0].get(name), expected, f'{role} {name}')
        measured = values[role + '/exposure.json']
        require(type(measured) is dict and measured.get('schemaVersion') == 1, 'Archived exposure schema required')
        cases = measured.get('cases')
        require(type(cases) is list and len(cases) == 128 + count, 'Complete retained offline/native exposure required')
        generator = random.Random(seed)
        native_bound = bound(config)
        require(count <= native_bound['maximumGeneratedWorlds'], 'Measured prefix exceeds independent bound')
        for ordinal, actual in enumerate(cases):
            if ordinal < 128:
                require(actual.get('kind') == 'td3-expert-corpus', 'Offline exposure order/kind mismatch')
                compare_world(actual, corpus[ordinal], f'{role} offline {ordinal}', budget)
            else:
                require(actual.get('kind') == 'td3-collection', 'Unknown or reordered measured query kind')
                native_ordinal = ordinal - 128
                expected = factory(generator, native_bound['nativeCurriculumOffset'] + native_ordinal, config['curriculum'])
                compare_world(actual, expected, f'{role} native {native_ordinal}', budget)
        results.append({'seed': seed, 'measuredNativeWorlds': count, 'exactOriginalFieldsMatched': True,
                        'scope': PREFIX_SCOPE, 'role': role, 'runId': run_id, 'exposureSha256': exposure_binding[0]})
    return results


def expected_shards(maximum):
    return [{'path': f'seed-{seed}-part-{part:02}/exposure.json', 'trainingSeed': seed,
             'firstOrdinal': first, 'count': min(1000, maximum - first)}
            for seed in SEEDS for part, first in enumerate(range(0, maximum, 1000))]


def checked_payload(tree, metadata, expected_metadata, budget):
    require(type(metadata) is dict and set(metadata) == set(expected_metadata) | {'sha256', 'bytes'},
            'Exact shard metadata fields required')
    for name, value in expected_metadata.items():
        exact(metadata[name], value, f'Shard metadata {name}')
    require(type(metadata['sha256']) is str and re.fullmatch('[a-f0-9]{64}', metadata['sha256']) is not None,
            'Complete lowercase shard SHA256 required')
    integer(metadata['bytes'], 1, MAX_FILE_BYTES, 'Shard byte count')
    content = tree.read(expected_metadata['path'])
    require(len(content) == metadata['bytes'] and sha256(content) == metadata['sha256'], 'Shard SHA256/bytes mismatch')
    return json_value(content, budget)


def audit(source_archive, bundle_directory, envelope_directory):
    require(platform.system() == 'Linux' and platform.python_implementation() == 'CPython' and
            platform.python_version() == '3.12.3', 'Full audit requires approved remote Linux CPython3.12.3')
    budget = Budget()
    auditor_content = read_single(Path(__file__), budget)
    bindings = archived_bindings()
    source_members, source_records, bundle_names = verify_source_package(source_archive, budget, bindings)
    with EvidenceTree(bundle_directory, budget) as bundle, EvidenceTree(envelope_directory, budget) as envelope:
        require(bundle.path != envelope.path and bundle.path not in envelope.path.parents and
                envelope.path not in bundle.path.parents, 'Separate non-nested source and envelope roots required')
        bundle.inventory(bundle_names)
        contents = {}
        for name in sorted(bundle_names):
            contents[name] = bundle.read(name)
            require(contents[name] == source_members['bundle/' + name], 'Extracted bundle differs from pinned source ZIP')
        values = {name: json_value(content, budget) for name, content in contents.items() if name.endswith('.json')}
        protocol = values['trial-protocol.json']
        declaration = values['final-stage-protocol.json']
        require(type(protocol) is dict and type(declaration) is dict, 'Original protocol/declaration objects required')
        exact(protocol.get('freshSeeds'), list(SEEDS), 'Reserved seed declaration')
        exact(declaration.get('seedOrder'), list(SEEDS), 'Pre-fresh declared seed order')
        exact(declaration.get('trialProtocolSha256'), PROTOCOL_SHA256, 'Pre-fresh protocol binding')
        exact(protocol.get('driverSha256'), sha256(contents['study-driver.py']), 'Original study caller')
        exact(declaration.get('freshDriverSha256'), sha256(contents['fresh-driver.py']), 'Original fresh caller')
        exact(protocol['sources'].get('roboproof/motion_learning/train.py'), TRAIN_SHA256, 'Declared generator source')
        exact(protocol['sources'].get('roboproof/motion_learning/td3_train.py'), TD3_SHA256, 'Declared collection control source')
        for name, expected_hash in CONFIG_HASHES.items():
            require(config_hash(protocol[name]) == expected_hash, 'Originally archived configuration identity required')
        exact(protocol['config']['seed'], 201, 'Pilot configuration seed')
        exact(protocol['controlConfig']['seed'], 201, 'Control configuration seed')
        exact(protocol['proofConfig']['seed'], 1360052, 'Proof configuration seed')
        calculated = bound(protocol['config'])
        exact(calculated, {'nativeCollections': 56, 'rolloutSteps': 256, 'maxSteps': 50000,
                           'nativeCurriculumOffset': 128, 'maximumGeneratedWorlds': 14336}, 'Independent bound')
        corpus_value = values['corpus-worlds.json']
        require(sha256(contents['corpus-worlds.json']) == CORPUS_SHA256 and
                protocol['config']['expertDatasetDefinition']['worldsSha256'] == CORPUS_SHA256 and
                type(corpus_value) is dict and type(corpus_value.get('cases')) is list and
                len(corpus_value['cases']) == 128, 'All original128 offline worlds required')
        corpus = corpus_value['cases']
        factory = original_factory(contents['train.py'], contents['td3_train.py'])
        measured_audits = audit_measured(values, protocol, corpus, factory, budget)
        descriptors = expected_shards(calculated['maximumGeneratedWorlds'])
        require(len(descriptors) == 45, 'Exactly45 generated shards required')
        expected_inventory = {'envelope-manifest.json', 'offline-corpus/exposure.json'} | {
            row['path'] for row in descriptors}
        envelope.inventory(expected_inventory)
        manifest_content = envelope.read('envelope-manifest.json')
        manifest = json_value(manifest_content, budget)
        expected_fields = {'schemaVersion', 'specification', 'python', 'platform', 'trialProtocolSha256',
            'preFreshDeclarationSha256', 'trainingSeeds', 'budget', 'inputSha256', 'measuredIndexSha256',
            'measuredPrefixAudits', 'shards', 'retainedArchives', 'authenticatedArchiveMembers',
            'indexAuthentication', 'offlineCorpus', 'totalPotentialWorlds', 'actualQueriesInferred',
            'newPhysicsQueries', 'modelOrResultRecoveryClaimed', 'scope'}
        require(type(manifest) is dict and set(manifest) == expected_fields, 'Exact conservative envelope manifest required')
        for name, expected in [('schemaVersion', 1), ('specification', METHOD), ('python', '3.12.3'),
                ('platform', 'Linux'), ('trialProtocolSha256', PROTOCOL_SHA256),
                ('preFreshDeclarationSha256', DECLARATION_SHA256), ('trainingSeeds', list(SEEDS)),
                ('inputSha256', {name: sha256(content) for name, content in contents.items()}),
                ('measuredIndexSha256', sha256(contents['measured-index.json'])),
                ('measuredPrefixAudits', measured_audits), ('retainedArchives', ARCHIVES),
                ('authenticatedArchiveMembers', bindings), ('indexAuthentication', INDEX_SCOPE),
                ('totalPotentialWorlds', 43008), ('actualQueriesInferred', False), ('newPhysicsQueries', 0),
                ('modelOrResultRecoveryClaimed', False), ('scope', BUILDER_SCOPE)]:
            exact(manifest[name], expected, f'Envelope {name}')
        declared_budget = manifest['budget']
        require(type(declared_budget) is dict and set(declared_budget) == {
            'nativeCollections', 'maximumGeneratedWorlds', 'nativeCurriculumOffset', 'reason'}, 'Exact declared budget fields required')
        for name in ('nativeCollections', 'maximumGeneratedWorlds', 'nativeCurriculumOffset'):
            exact(declared_budget[name], calculated[name], f'Envelope budget {name}')
        require(type(declared_budget['reason']) is str and 0 < len(declared_budget['reason']) <= 2048,
                'Bounded budget explanation required')
        offline = checked_payload(envelope, manifest['offlineCorpus'],
                                  {'path': 'offline-corpus/exposure.json', 'count': 128}, budget)
        exact(offline, {'schemaVersion': 1, 'specification': METHOD, 'declaredOriginalCorpusExposure': True,
                        'cases': [{**world, 'kind': 'td3-expert-corpus'} for world in corpus]}, 'Offline payload')
        for ordinal, actual in enumerate(offline['cases']):
            compare_world(actual, corpus[ordinal], f'Envelope offline {ordinal}', budget)
        require(type(manifest['shards']) is list and len(manifest['shards']) == len(descriptors),
                'Complete ordered45-shard catalog required')
        generated_count = 0
        generators = {seed: random.Random(seed) for seed in SEEDS}
        for metadata, expected_metadata in zip(manifest['shards'], descriptors):
            payload = checked_payload(envelope, metadata, expected_metadata, budget)
            seed = expected_metadata['trainingSeed']
            require(type(payload) is dict and set(payload) == {
                'schemaVersion', 'specification', 'trainingSeed', 'conservativeSupersetNotMeasuredQueries', 'cases'},
                'Exact potential-world shard fields required')
            for name, expected in [('schemaVersion', 1), ('specification', METHOD), ('trainingSeed', seed),
                                   ('conservativeSupersetNotMeasuredQueries', True)]:
                exact(payload[name], expected, f'Potential shard {name}')
            rows = payload['cases']
            require(type(rows) is list and len(rows) == expected_metadata['count'], 'Exact shard row count required')
            for local_ordinal, actual in enumerate(rows):
                ordinal = expected_metadata['firstOrdinal'] + local_ordinal
                expected = factory(generators[seed], calculated['nativeCurriculumOffset'] + ordinal,
                                   protocol['config']['curriculum'])
                expected = {'id': f'possible-{seed}-{ordinal}', **expected, 'kind': 'conservative-potential-training',
                            'trainingSeed': seed, 'ordinal': ordinal}
                exact(actual, expected, f'Potential seed{seed} ordinal{ordinal}')
                compare_world(actual, expected, f'Potential seed{seed} ordinal{ordinal}', budget)
                generated_count += 1
        require(generated_count == 43008 and budget.worlds == 43909,
                'Complete aggregate generated/offline/measured row accounting required')
        bundle.inventory(bundle_names)
        envelope.inventory(expected_inventory)
        bundle.assert_current()
        envelope.assert_current()
        result = {'schemaVersion': 1, 'specification': AUDIT_METHOD, 'verified': True,
            'auditor': file_record(auditor_content),
            'runtime': {'implementation': platform.python_implementation(), 'python': platform.python_version(),
                        'platform': platform.system(), 'version': sys.version},
            'sourcePackage': {'sha256': SOURCE_PACKAGE_SHA256, 'bytes': SOURCE_PACKAGE_BYTES,
                              'payloadMembers': 20, 'zipEntries': 21, 'members': source_records},
            'builderSha256': BUILDER_SHA256, 'trialProtocolSha256': PROTOCOL_SHA256,
            'preFreshDeclarationSha256': DECLARATION_SHA256, 'configurationSha256': CONFIG_HASHES,
            'normalizedGeneratorSha256': TRAIN_SHA256, 'normalizedCollectionControlSha256': TD3_SHA256,
            'trainingSeeds': list(SEEDS), 'independentlyComputedBound': calculated,
            'boundReason': '64 minus8 offline-only updates leaves56 native collections. Each can start at most256 worlds, including a final interrupted zero-step world. The observed<50000 pre-generation guard is a second bound. Interruption restores task RNG and exits; the pinned fresh caller never resumes. Abort-only frozen/consumed rejection is omitted only for this exclusion superset.',
            'measuredPrefixAudits': measured_audits, 'authenticatedArchiveMembers': bindings,
            'envelopeManifest': {'path': 'envelope-manifest.json', **file_record(manifest_content)},
            'bundleFiles': bundle.records, 'envelopeFiles': envelope.records,
            'verifiedPotentialWorlds': generated_count, 'verifiedOfflineWorlds': 128,
            'verifiedMeasuredNativeWorldRecords': 261, 'verifiedMeasuredOfflineWorldRecords': 512,
            'verifiedExclusionWorldRecords': 43136,
            'counters': {'filesRead': budget.files, 'bytesRead': budget.bytes, 'directoryEntries': budget.entries,
                         'jsonValueNodes': budget.nodes, 'worldRowsCompared': budget.worlds},
            'actualQueriesInferred': False, 'newPhysicsQueries': 0, 'modelOrResultRecoveryClaimed': False,
            'lossRegistryModified': False, 'lossStateRemains': 'unresolved', 'qualificationClaimed': False,
            'scope': 'Read-only source-bound conservative exclusion proof, not recovered model/outcome evidence, a qualification verdict or external cryptographic custody. The registry event remains unresolved and its seeds remain reserved. Admission must pin this audit and every original manifest/input/shard byte; a verified flag alone is insufficient. Historical development, measured exposures and consumed-final exclusions remain separately required.'}
        require(len(canonical(result)) <= MAX_AUDIT_BYTES, 'Bounded audit JSON required')
        return result


def publish(filename, result, input_roots, source_archive):
    filename = Path(os.path.abspath(filename))
    for directory in input_roots:
        root = Path(os.path.abspath(directory))
        require(filename != root and root not in filename.parents, 'Audit output cannot modify either evidence tree')
    require(filename != Path(os.path.abspath(source_archive)) and filename != Path(os.path.abspath(__file__)),
            'Audit output cannot replace an input or its auditor')
    content = canonical(result) + b'\n'
    require(len(content) <= MAX_AUDIT_BYTES, 'Audit output exceeds byte limit')
    parent = open_directory(filename.parent)
    descriptor = None
    try:
        descriptor = os.open(filename.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        with os.fdopen(descriptor, 'wb') as output:
            descriptor = None
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.fsync(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
    return {'path': filename.name, 'sha256': sha256(content), 'bytes': len(content)}


def main():
    parser = argparse.ArgumentParser(description='Independent remote-only exhaustive metadata audit; no training or physics')
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--envelope', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = audit(arguments.source_archive, arguments.bundle, arguments.envelope)
        publication = publish(arguments.output, result, (arguments.bundle, arguments.envelope), arguments.source_archive)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, zipfile.BadZipFile) as error:
        print(json.dumps({'specification': AUDIT_METHOD, 'verified': False,
                          'error': str(error)[:2048], 'lossStateRemains': 'unresolved'}), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({'specification': AUDIT_METHOD, 'audit': publication,
                      'verifiedPotentialWorlds': result['verifiedPotentialWorlds'], 'verifiedOfflineWorlds': 128,
                      'newPhysicsQueries': 0, 'modelOrResultRecoveryClaimed': False, 'lossStateRemains': 'unresolved'}))


if __name__ == '__main__':
    main()
