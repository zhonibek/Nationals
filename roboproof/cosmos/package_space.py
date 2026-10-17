"""Build an allowlisted deployment artifact; never authenticate, publish or allocate compute."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
MODULES = ('__init__.py', 'app.py', 'config.py', 'errors.py', 'media.py', 'request_guard.py', 'runtime.py', 'space_app.py')


def package_space(destination=None):
    output = Path(destination) if destination is not None else ROOT / '.cache/cosmos-space-20261016.zip'
    output = output.resolve()
    allowed = (ROOT / '.cache').resolve()
    if output.parent != allowed or output.exists() or any(parent.is_symlink() for parent in [output, *output.parents]):
        raise ValueError('A new archive directly inside the project ignored cache is required')
    package = ROOT / 'roboproof/cosmos'
    files = {}
    for name in (*MODULES, 'SPACE_README.md', 'requirements-space.txt'):
        source = package / name
        if (source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(ROOT) or
                any(parent.is_symlink() for parent in source.parents) or source.stat().st_size > 128 * 1024):
            raise ValueError('Only bounded reviewed Cosmos source files may be packaged')
        destination_name = {'SPACE_README.md': 'README.md', 'requirements-space.txt': 'requirements.txt'}.get(name, f'roboproof/cosmos/{name}')
        files[destination_name] = source.read_bytes()
    files['app.py'] = b'from roboproof.cosmos.space_app import launch\n\nlaunch(host="0.0.0.0")\n'
    files['NOTICE'] = b'Built on NVIDIA Cosmos. Licensed by NVIDIA Corporation under the NVIDIA Open Model License.\nModel license: https://huggingface.co/nvidia/Cosmos-Reason2-2B/blob/main/LICENSE\n'
    manifest = {'schemaVersion': 1, 'files': [{'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                                           for name, content in sorted(files.items())],
                'scope': 'Existing project Cosmos adapter only; no credentials, simulator data, robot source, weights or personal files. Packaging is not deployment or inference.',
                'publicationPerformed': False, 'hardwareAllocated': False, 'inferenceVerified': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr('deployment-manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files) | {'deployment-manifest.json'}:
            raise ValueError('Complete unique deployment inventory required')
        if any(hashlib.sha256(archive.read(row['path'])).hexdigest() != row['sha256'] for row in manifest['files']):
            raise ValueError('Every packaged source byte must verify')
    return {'path': str(output), 'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'files': len(files), 'publicationPerformed': False, 'hardwareAllocated': False, 'inferenceVerified': False}


if __name__ == '__main__':
    print(json.dumps(package_space()))
