"""Export bounded visualizations of saved original-Simulator poses, without inference."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def export_session(session_id, directory=None, frame_count=32):
    if str(uuid.UUID(session_id)) != session_id or type(frame_count) is not int or not 2 <= frame_count <= 32:
        raise ValueError('Canonical session ID and2..32 visualization frames required')
    sessions = ROOT / 'runs/motion/sessions'
    source = sessions / f'{session_id}.json'
    if (source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(sessions.resolve()) or
            not 1 <= source.stat().st_size <= 64 * 1024 * 1024):
        raise ValueError('Bounded original saved session required')
    data = source.read_bytes()
    record = json.loads(data)
    report = record['report']
    if record['id'] != session_id or record.get('exactReplayVerified') is not True or report.get('fixedDt') != 0.01:
        raise ValueError('Exact-replay-verified original100Hz Simulator session required')
    rows = report['transitions']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6000:
        raise ValueError('Bounded complete recorded transitions required')
    start, goal = report['task']['start'], report['task']['goal']
    samples = [(0.0, [start['xIn'], start['yIn'], start['headingDeg']])]
    previous_tick = -1
    for row in rows:
        pose = row['evaluator']['poseInchesDegrees']
        tick = row['tick']
        if (type(tick) is not int or tick != previous_tick + 1 or not isinstance(pose, list) or len(pose) != 3 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in pose)):
            raise ValueError('Every original contiguous tick and finite pose must be retained')
        samples.append(((tick + 1) * report['fixedDt'], pose))
        previous_tick = tick
    duration = samples[-1][0]
    if not 0 < duration <= 29:
        raise ValueError('Use an original session shorter than29 seconds; never silently crop motion')
    target = (Path(directory) if directory is not None else ROOT.parent / '.cache/cosmos-motion-media').resolve()
    allowed = (ROOT.parent / '.cache/cosmos-motion-media').resolve()
    if target != allowed or any(parent.is_symlink() for parent in [target, *target.parents]):
        raise ValueError('Only the dedicated project visualization directory is allowed')
    target.mkdir(parents=True, exist_ok=True)
    destination = target / f'{session_id}.zip'
    if destination.exists():
        raise ValueError('Never overwrite a saved visualization')
    from PIL import Image, ImageDraw, ImageFont

    positions = [pose[:2] for _, pose in samples] + [[goal['xIn'], goal['yIn']]]
    minimum_x = min(position[0] for position in positions) - 10
    minimum_y = min(position[1] for position in positions) - 10
    span = max(max(position[0] for position in positions) - minimum_x,
               max(position[1] for position in positions) - minimum_y) + 10

    def screen(position):
        return (32 + (position[0] - minimum_x) / span * 320, 360 - (position[1] - minimum_y) / span * 320)

    font = ImageFont.load_default(size=14)
    frames = []
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(frame_count):
            requested_time = duration * index / (frame_count - 1)
            sample_index = min(len(samples) - 1, math.floor(requested_time / report['fixedDt'] + 1e-9))
            timestamp, pose = samples[sample_index]
            image = Image.new('RGB', (384, 384), '#f2f5f8')
            drawing = ImageDraw.Draw(image)
            drawing.text((10, 5), 'SIMULATOR POSE REPLAY - not camera', fill='#111827', font=font)
            drawing.text((10, 23), f't = {timestamp:.2f} s', fill='#111827', font=font)
            drawing.rectangle((32, 40, 352, 360), outline='#b3becb', width=1)
            path_points = [screen(sample[1]) for sample in samples[:sample_index + 1]]
            if len(path_points) > 1:
                drawing.line(path_points, fill='#7fb3d5', width=2)
            goal_x, goal_y = screen((goal['xIn'], goal['yIn']))
            drawing.line((goal_x - 6, goal_y, goal_x + 6, goal_y), fill='#c81e1e', width=3)
            drawing.line((goal_x, goal_y - 6, goal_x, goal_y + 6), fill='#c81e1e', width=3)
            robot_x, robot_y = screen(pose)
            angle = math.radians(pose[2])
            forward = (math.sin(angle), -math.cos(angle))
            right = (math.cos(angle), math.sin(angle))
            drawing.polygon([(robot_x + forward[0] * 10, robot_y + forward[1] * 10),
                             (robot_x - forward[0] * 7 + right[0] * 7, robot_y - forward[1] * 7 + right[1] * 7),
                             (robot_x - forward[0] * 7 - right[0] * 7, robot_y - forward[1] * 7 - right[1] * 7)],
                            fill='#155eef')
            content = io.BytesIO()
            image.save(content, format='PNG')
            filename = f'frames/{index:03d}.png'
            frame_bytes = content.getvalue()
            archive.writestr(filename, frame_bytes)
            frames.append({'path': filename, 'sha256': hashlib.sha256(frame_bytes).hexdigest(),
                           'recordedTick': sample_index - 1 if sample_index else None, 'recordedTimeSeconds': timestamp})
        manifest = {'schemaVersion': 1, 'sessionId': session_id, 'sessionSha256': hashlib.sha256(data).hexdigest(),
                    'kind': 'original-simulator-pose-visualization', 'frames': frames,
                    'fps': (frame_count - 1) / duration, 'recordedDurationSeconds': duration,
                    'finalFrameHoldSeconds': duration / (frame_count - 1), 'sourceFixedDt': report['fixedDt'],
                    'newSimulatorQueries': 0, 'modelInferencePerformed': False,
                    'scope': 'Nearest original recorded pose at or before each display time; schematic blue robot and red target, not camera or3D footage. The final encoding interval holds the last recorded pose. No new physics, invented motion or training.'}
        archive.writestr('manifest.json', json.dumps(manifest, indent=2, allow_nan=False))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or any(hashlib.sha256(archive.read(frame['path'])).hexdigest() != frame['sha256'] for frame in frames):
            raise ValueError('Every original-pose visualization frame must verify')
    return {'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'bytes': destination.stat().st_size, 'sessionId': session_id, 'frames': frame_count,
            'modelInferencePerformed': False, 'newSimulatorQueries': 0}


def main():
    parser = argparse.ArgumentParser(description='Package saved original Simulator motion for a separately approved Cosmos GPU analysis.')
    parser.add_argument('--session', required=True)
    arguments = parser.parse_args()
    print(json.dumps(export_session(arguments.session)))


if __name__ == '__main__':
    main()
