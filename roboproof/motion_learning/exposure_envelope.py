import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import zlib


METHOD = 'td3-full-corpus-prefix-envelope-v1'
TRAIN_SHA256 = 'a82599abfba521c07a711649a0aa941e9a5c0367da33d85aff8ad103c247560e'
TD3_SHA256 = '741e11263b766e857a3a9540f17a25eefc08a6997406c750215eb5da1c295314'
MAX_JSON_BYTES = 4 * 1024 * 1024
WORLD_FIELDS = ('seed', 'task', 'configuration', 'stage')
FRESH_SEEDS = (136001, 136002, 136003)
PROTOCOL_SHA256 = 'a8b167033c3e22723756ea534c9399a431b33e5766bea1e89c4f72feef6eede8'
FRESH_CONFIG_SHA256 = 'd8eb17caca550c51bdf77249c4e4e900750b7f9b605485c90bdc1d218ee645ae'
CONTROL_CONFIG_SHA256 = 'fd4a09dda3828b34d3bc6c6d5591af7aa9573d61031fdb9d9a9f4ee1f1f0498d'
PROOF_CONFIG_SHA256 = '7d402e135595dbee497648db896b1d5720d326be402b58daef99ddb3227595f0'
STUDY_MEMBER = 'roboproof/runs/motion/research/episode-balanced-remote-study-20261016/'
ARCHIVES = {
    'pilot': {'name': 'episode-balanced-colab-pilot-20261016.zip',
              'sha256': 'c53261df5f2a049f36b7018973e801357c328b67009264f633eb08fcfa6cf922'},
    'fresh-launch': {'name': 'episode-balanced-colab-fresh-launch-20261016.zip',
                     'sha256': '2f361574b2eda463512bd3ecb091b54fd0d4b9aff0e1f255849830c58f23daee'},
}


def member_binding(archive, member, sha256, size, crc32):
    return {'archiveSha256': ARCHIVES[archive]['sha256'], 'member': member,
            'sha256': sha256, 'bytes': size, 'crc32': crc32}


ARCHIVED_MEMBERS = {
    'trial-protocol.json': member_binding('pilot', STUDY_MEMBER + 'trial-protocol.json',
        PROTOCOL_SHA256, 128732, '03e7376a'),
    'final-stage-protocol.json': member_binding('fresh-launch', STUDY_MEMBER + 'final-stage-protocol.json',
        '7b287c2726ecb28e7a50a7a3b6a6a67fe43fd1922592183561bd532867d44cc0', 1582, '8a241cc8'),
    'study-driver.py': member_binding('pilot', '.cache/remote-episode-replay-study-20261016.py',
        'd979c77ab90c060186a649e7ec07e32d72ae62b455582f930f75b5efda4eb418', 14976, '9f10d3d3'),
    'fresh-driver.py': member_binding('fresh-launch', '.cache/episode-balanced-colab-fresh-20261016.py',
        '44b49c8cf25a130f64ec62714520db4f6af490beb0906a6cc05fef0c9683fc03', 4763, '2426bc09'),
}
ARCHIVED_RUNS = {
    '67cbb420-fade-4208-91a3-1da837ea397b': {
        'role': 'pilot', 'folder': 'candidate', 'seed': 201, 'configSha256': FRESH_CONFIG_SHA256, 'nativeWorlds': 130,
        'manifest': ('f326e7474b5fae162f990f910c15bcd6609c6526a24f046972eb7f8018a614a8', 120638, '47312d8a'),
        'exposure': ('00786cafd9bcdc5f96b1b26e90443ad17f5803d7ff46aa0f5265f1cc0cb86790', 108324, 'eb16385d')},
    '484dbd88-2d67-4dd2-a9b7-0ccab8810337': {
        'role': 'control', 'folder': 'control', 'seed': 201, 'configSha256': CONTROL_CONFIG_SHA256, 'nativeWorlds': 125,
        'manifest': ('069ef7eab242a5fded4bd87141864a2b6945d55438cb768e3a9e148099d2f04d', 116092, '0a34d216'),
        'exposure': ('da28b62dd23969c0e0a81185393f3931fc17dcbf92bad2d07ef6674c428e7d6b', 106323, '16a3eacb')},
    'edb7296a-9189-41ad-8864-19edcea0f813': {
        'role': 'proof-complete', 'folder': 'proof-complete', 'seed': 1360052,
        'configSha256': PROOF_CONFIG_SHA256, 'nativeWorlds': 3,
        'manifest': ('a6b014d85513a31f0d749d0a5594be12713ff1d87d38bb4aa0129ffe6cfefbd9', 20961, 'd4e91ef6'),
        'exposure': ('2b5a72d5a065734f278edb980da436ff75e29f2a2e2b601134997c1b2f851218', 59221, '8b6eaa97')},
    '3bc26c9c-04ca-4be0-931f-3e3b2f2ef651': {
        'role': 'proof-resumed', 'folder': 'proof-resumed', 'seed': 1360052,
        'configSha256': PROOF_CONFIG_SHA256, 'nativeWorlds': 3,
        'manifest': ('8fe2582e7ec4d687c55c8c296161f559fc45999bbacb07bbd53a5e278ce5f0c0', 20961, '0dcc31e8'),
        'exposure': ('2b5a72d5a065734f278edb980da436ff75e29f2a2e2b601134997c1b2f851218', 59221, '8b6eaa97')},
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def normalized_source(content):
    return content.decode('utf-8-sig').replace('\r\n', '\n')


def bounded_bytes(filename):
    filename = Path(filename)
    if filename.is_symlink() or not filename.is_file() or filename.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Bounded regular original evidence file required')
    before = filename.stat()
    content = filename.read_bytes()
    after = filename.stat()
    if len(content) != before.st_size or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError('Evidence changed during read')
    return content


def bounded_json(filename):
    content = bounded_bytes(filename)
    return json.loads(content.decode('utf-8-sig')), digest(content)


def owned_path(bundle, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Owned original evidence path required')
    filename = Path(bundle)
    for part in (None, *relative.parts):
        if part is not None:
            filename = filename / part
        if filename.is_symlink():
            raise ValueError('Evidence paths cannot traverse symlinks')
    return filename


def verify_member(content, binding):
    if (digest(content) != binding['sha256'] or len(content) != binding['bytes'] or
            f'{zlib.crc32(content):08x}' != binding['crc32']):
        raise ValueError('Exact member bytes from the retained SHA/CRC-verified archive required')
    return content


def read_member(bundle, relative, binding):
    return verify_member(bounded_bytes(owned_path(bundle, relative)), binding)


def write_json(filename, value):
    content = json.dumps(value, indent=2, allow_nan=False).encode()
    if len(content) > MAX_JSON_BYTES:
        raise ValueError('Envelope shard exceeds original evidence byte limit')
    with Path(filename).open('xb') as output:
        output.write(content)
    return {'sha256': digest(content), 'bytes': len(content)}


def configuration_sha256(config):
    if not isinstance(config, dict) or isinstance(config.get('seed'), bool) or not isinstance(config.get('seed'), int):
        raise ValueError('Exact declared configuration and integer seed required')
    try:
        content = json.dumps({name: value for name, value in config.items() if name != 'seed'},
                             sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError('Finite original JSON configuration required') from error
    return digest(content)


def checked_configuration(config, expected_sha256, seeds):
    if configuration_sha256(config) != expected_sha256 or config['seed'] not in seeds:
        raise ValueError('Exact originally declared configuration required; overrides and fresh resume are not admitted')


def configuration_budget(config):
    expert = config['expertDatasetDefinition']
    native_collections = config['updates'] - config['warmStartUpdates']
    bound = min(config['maxSteps'], native_collections * config['rolloutSteps'])
    return {'nativeCollections': native_collections, 'maximumGeneratedWorlds': bound,
            'nativeCurriculumOffset': expert['nativeCurriculumOffset'],
            'reason': 'Each native attempt starts at most rolloutSteps candidates, including its last interrupted zero-step candidate. The pre-generation observed<maxSteps guard also bounds candidates by maxSteps. No fresh resume is admitted; an interrupted collection restores RNG then exits the fit.'}


def envelope_budget(config):
    checked_configuration(config, FRESH_CONFIG_SHA256, (201, *FRESH_SEEDS))
    return configuration_budget(config)


def world_factory(train_content, td3_content):
    train_source, td3_source = normalized_source(train_content), normalized_source(td3_content)
    if digest(train_source.encode()) != TRAIN_SHA256 or digest(td3_source.encode()) != TD3_SHA256:
        raise ValueError('Exact audited task-generator and collection-control sources required')
    tree = ast.parse(train_source)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) and
             len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('CURRICULUM', 'CURRICULA') or
             isinstance(node, ast.FunctionDef) and node.name in ('full_reach_world', 'training_world')]
    if len(nodes) != 4:
        raise ValueError('Exact original two declarations and two task-generator functions required')
    namespace = {'math': math, 'exclude_frozen_world': lambda world: world}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'audited-original-task-generator', 'exec'), namespace)
    return namespace['training_world']


def generated_prefix(factory, config, seed, count):
    budget = envelope_budget(config)
    if seed not in FRESH_SEEDS:
        raise ValueError('Only the three originally reserved fresh training seeds are supported')
    yield from prefix_rows(factory, config, seed, count, budget)


def prefix_rows(factory, config, seed, count, budget):
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed != config['seed'] or
            isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= budget['maximumGeneratedWorlds']):
        raise ValueError('Bounded declared seed and conservative prefix length required')
    generator = random.Random(seed)
    for ordinal in range(count):
        yield factory(generator, budget['nativeCurriculumOffset'] + ordinal, config['curriculum'])


def verify_measured_prefix(factory, config, seed, cases):
    config_sha = configuration_sha256(config)
    if config_sha == FRESH_CONFIG_SHA256:
        checked_configuration(config, FRESH_CONFIG_SHA256, (201, *FRESH_SEEDS))
    elif config_sha == CONTROL_CONFIG_SHA256:
        checked_configuration(config, CONTROL_CONFIG_SHA256, (201,))
    else:
        checked_configuration(config, PROOF_CONFIG_SHA256, (1360052,))
    if type(seed) is not int or seed != config['seed']:
        raise ValueError('Measured exposure seed must match its exact declared configuration')
    if not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases):
        raise ValueError('Complete measured source exposure required')
    if any(row.get('kind') not in ('td3-expert-corpus', 'td3-collection') for row in cases):
        raise ValueError('Unreviewed query kind cannot be silently omitted')
    native = [row for row in cases if row['kind'] == 'td3-collection']
    generated = prefix_rows(factory, config, seed, len(native), configuration_budget(config))
    for ordinal, (actual, expected) in enumerate(zip(native, generated)):
        if any(actual.get(name) != expected[name] for name in WORLD_FIELDS):
            raise ValueError(f'Original measured task prefix differs at ordinal{ordinal}')
    return {'seed': seed, 'measuredNativeWorlds': len(native), 'exactOriginalFieldsMatched': True,
            'scope': 'Matches actual retained task/physics metadata, not model states or newly simulated outcomes'}


def authenticate_index(bundle, index, protocol):
    if not isinstance(index, dict) or not isinstance(index.get('runs'), list) or len(index['runs']) != len(ARCHIVED_RUNS):
        raise ValueError('Exactly the four retained archived audit runs are required')
    seen_runs, seen_paths, authenticated = set(), set(), []
    for entry in index['runs']:
        if not isinstance(entry, dict) or not isinstance(entry.get('runId'), str) or entry['runId'] not in ARCHIVED_RUNS:
            raise ValueError('Unknown archived audit run')
        run_id = entry['runId']
        expected = ARCHIVED_RUNS[run_id]
        if not isinstance(entry.get('path'), str):
            raise ValueError('Owned original measured exposure path required')
        relative = Path(entry['path'])
        if (len(relative.parts) != 2 or relative.name != 'exposure.json' or
                relative.is_absolute() or '..' in relative.parts or '\\' in entry['path']):
            raise ValueError('Owned original measured exposure path required')
        if run_id in seen_runs or relative.as_posix() in seen_paths:
            raise ValueError('Distinct archived audit run IDs and exposure paths required')
        seen_runs.add(run_id)
        seen_paths.add(relative.as_posix())
        if (entry.get('role', expected['role']) != expected['role'] or
                type(entry.get('seed')) is not int or entry['seed'] != expected['seed'] or
                entry.get('sources') != {'train.py': TRAIN_SHA256, 'td3_train.py': TD3_SHA256} or
                entry.get('sha256') != expected['exposure'][0]):
            raise ValueError('Index entries must identify the exact retained archive run members')
        checked_configuration(entry.get('config'), expected['configSha256'], (expected['seed'],))
        base = STUDY_MEMBER + expected['folder'] + '/' + run_id + '/'
        bindings = {
            relative.with_name('run.json').as_posix(): member_binding('pilot', base + 'run.json', *expected['manifest']),
            relative.as_posix(): member_binding('pilot', base + f'seed-{expected["seed"]}/exposure.json', *expected['exposure']),
        }
        contents = {name: read_member(bundle, name, binding) for name, binding in bindings.items()}
        manifest = json.loads(contents[relative.with_name('run.json').as_posix()].decode('utf-8-sig'))
        measured = json.loads(contents[relative.as_posix()].decode('utf-8-sig'))
        models = manifest.get('models')
        if (manifest.get('status') != 'completed' or manifest.get('runId') != run_id or
                manifest.get('sources') != protocol['sources'] or manifest.get('config') != entry['config'] or
                not isinstance(models, list) or len(models) != 1 or models[0].get('seed') != expected['seed'] or
                models[0].get('updates') != entry['config']['updates'] or
                models[0].get('episodes') != expected['nativeWorlds']):
            raise ValueError('Actual archived completed run manifest/configuration/source evidence required')
        cases = measured.get('cases')
        if (not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases) or
                sum(row.get('kind') == 'td3-collection' for row in cases) != expected['nativeWorlds']):
            raise ValueError('Complete original archived native exposure required, not a shortened prefix')
        authenticated.append({'entry': entry, 'measured': measured, 'bindings': bindings})
    if seen_runs != set(ARCHIVED_RUNS):
        raise ValueError('Every distinct retained pilot/control/complete/resumed proof run is required')
    return authenticated


def verify_offline_worlds(cases, corpus):
    offline = [row for row in cases if row.get('kind') == 'td3-expert-corpus']
    if len(offline) != len(corpus) or any(any(left.get(name) != right.get(name) for name in WORLD_FIELDS)
                                       for left, right in zip(offline, corpus)):
        raise ValueError('Every actual preserved run must retain all original corpus worlds in original order')


def build(bundle, directory, expected_protocol_sha256):
    if platform.system() != 'Linux' or platform.python_version() != '3.12.3':
        raise ValueError('Full envelope generation requires the approved remote pinned Python3.12.3 Linux runtime')
    bundle, directory = Path(bundle), Path(directory)
    contents = {name: read_member(bundle, name, binding) for name, binding in ARCHIVED_MEMBERS.items()}
    protocol = json.loads(contents['trial-protocol.json'].decode('utf-8-sig'))
    protocol_sha = digest(contents['trial-protocol.json'])
    if (expected_protocol_sha256 != PROTOCOL_SHA256 or protocol_sha != expected_protocol_sha256 or
            protocol['freshSeeds'] != list(FRESH_SEEDS)):
        raise ValueError('Exact original lost trial and reserved independent seed declaration required')
    if (protocol['sources'].get('roboproof/motion_learning/train.py') != TRAIN_SHA256 or
            protocol['sources'].get('roboproof/motion_learning/td3_train.py') != TD3_SHA256):
        raise ValueError('Original trial must bind both audited generator/control sources')
    declaration = json.loads(contents['final-stage-protocol.json'].decode('utf-8-sig'))
    declaration_sha = digest(contents['final-stage-protocol.json'])
    if (declaration['trialProtocolSha256'] != protocol_sha or declaration['seedOrder'] != protocol['freshSeeds'] or
            digest(contents['study-driver.py']) != protocol['driverSha256'] or
            digest(contents['fresh-driver.py']) != declaration['freshDriverSha256']):
        raise ValueError('Actual archived Colab-adapted driver and pre-fresh caller/declaration bindings required')
    budget = envelope_budget(protocol['config'])
    train_content, td3_content = (bounded_bytes(owned_path(bundle, name)) for name in ('train.py', 'td3_train.py'))
    contents.update({'train.py': train_content, 'td3_train.py': td3_content})
    factory = world_factory(train_content, td3_content)
    contents['measured-index.json'] = bounded_bytes(owned_path(bundle, 'measured-index.json'))
    index = json.loads(contents['measured-index.json'].decode('utf-8-sig'))
    index_sha = digest(contents['measured-index.json'])
    contents['corpus-worlds.json'] = bounded_bytes(owned_path(bundle, 'corpus-worlds.json'))
    corpus = json.loads(contents['corpus-worlds.json'].decode('utf-8-sig'))
    corpus_sha = digest(contents['corpus-worlds.json'])
    if (corpus_sha != protocol['config']['expertDatasetDefinition']['worldsSha256'] or
            not isinstance(corpus.get('cases'), list) or len(corpus['cases']) != 128):
        raise ValueError('Every one of the original128 corpus worlds, including failed-label worlds, must remain excluded')
    authenticated = authenticate_index(bundle, index, protocol)
    audits = []
    inputs = {name: digest(content) for name, content in contents.items()}
    bindings = dict(ARCHIVED_MEMBERS)
    for record in authenticated:
        entry, measured = record['entry'], record['measured']
        result = verify_measured_prefix(factory, entry['config'], entry['seed'], measured['cases'])
        verify_offline_worlds(measured['cases'], corpus['cases'])
        audits.append({**result, 'role': ARCHIVED_RUNS[entry['runId']]['role'], 'runId': entry['runId'],
                       'exposureSha256': entry['sha256']})
        bindings.update(record['bindings'])
        inputs.update({name: binding['sha256'] for name, binding in record['bindings'].items()})
    directory.mkdir(exist_ok=False)
    shards = []
    offline_directory = directory / 'offline-corpus'
    offline_directory.mkdir()
    offline_payload = {'schemaVersion': 1, 'specification': METHOD, 'declaredOriginalCorpusExposure': True,
                       'cases': [{**world, 'kind': 'td3-expert-corpus'} for world in corpus['cases']]}
    offline_metadata = write_json(offline_directory / 'exposure.json', offline_payload)
    for seed in protocol['freshSeeds']:
        rows = []
        seed_config = {**protocol['config'], 'seed': seed}
        for ordinal, world in enumerate(generated_prefix(factory, seed_config, seed, budget['maximumGeneratedWorlds'])):
            rows.append({'id': f'possible-{seed}-{ordinal}', **world, 'kind': 'conservative-potential-training',
                         'trainingSeed': seed, 'ordinal': ordinal})
            if len(rows) == 1000 or ordinal + 1 == budget['maximumGeneratedWorlds']:
                name = f'seed-{seed}-part-{ordinal // 1000:02}'
                folder = directory / name
                folder.mkdir()
                payload = {'schemaVersion': 1, 'specification': METHOD, 'trainingSeed': seed,
                           'conservativeSupersetNotMeasuredQueries': True, 'cases': rows}
                metadata = write_json(folder / 'exposure.json', payload)
                shards.append({'path': name + '/exposure.json', 'trainingSeed': seed,
                               'firstOrdinal': rows[0]['ordinal'], 'count': len(rows), **metadata})
                rows = []
    manifest = {'schemaVersion': 1, 'specification': METHOD, 'python': platform.python_version(), 'platform': platform.system(),
                'trialProtocolSha256': protocol_sha, 'preFreshDeclarationSha256': declaration_sha,
                'trainingSeeds': protocol['freshSeeds'], 'budget': budget,
                'inputSha256': inputs, 'measuredIndexSha256': index_sha, 'measuredPrefixAudits': audits, 'shards': shards,
                'retainedArchives': ARCHIVES, 'authenticatedArchiveMembers': bindings,
                'indexAuthentication': 'Derived index checked against four distinct pinned archived completed run manifests and complete exposure members; not self-authenticated index assertions.',
                'offlineCorpus': {'path': 'offline-corpus/exposure.json', 'count': 128, **offline_metadata},
                'totalPotentialWorlds': sum(row['count'] for row in shards), 'actualQueriesInferred': False,
                'newPhysicsQueries': 0, 'modelOrResultRecoveryClaimed': False,
                'scope': 'Conservative exclusion metadata only. Original task-generator functions and Python task RNG are replayed without training, simulator steps or actor/model recovery. Abort-only frozen/consumed filters are omitted to include a superset, not to query forbidden worlds. Old campaign outcomes remain unavailable.'}
    write_json(directory / 'envelope-manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Remote source-bound conservative task-exposure envelope; never a trainer or simulator')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    arguments = parser.parse_args()
    result = build(arguments.bundle, arguments.directory, arguments.protocol_sha256)
    print(json.dumps({name: result[name] for name in ['specification', 'budget', 'totalPotentialWorlds', 'measuredPrefixAudits', 'newPhysicsQueries', 'modelOrResultRecoveryClaimed']}))


if __name__ == '__main__':
    main()
