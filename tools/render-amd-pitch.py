import argparse
import hashlib
import json
import math
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.cache' / 'pitch-python'))
import imageio_ffmpeg

WIDTH, HEIGHT, FPS, SCALE = 1920, 1080, 24, 1.5
INK = '#eef5ff'
MUTED = '#95aac4'
CYAN = '#49e7dc'
AMBER = '#ffb45b'
RED = '#ff688b'
BLUE = '#71a7ff'


@lru_cache(maxsize=50)
def font(size, bold=False):
    filename = 'segoeuib.ttf' if bold else 'segoeui.ttf'
    return ImageFont.truetype(str(Path('C:/Windows/Fonts') / filename), round(size * SCALE))


def coordinates(values):
    return tuple(round(value * SCALE) for value in values)


def text(draw, location, value, size=24, fill=INK, bold=False):
    draw.text(coordinates(location), value, font=font(size, bold), fill=fill, spacing=round(5 * SCALE))


def line(draw, points, fill=CYAN, width=2):
    draw.line([coordinates(point) for point in points], fill=fill, width=max(1, round(width * SCALE)), joint='curve')


def box(draw, bounds, fill='#101d32', outline='#263d56', radius=16, width=1):
    draw.rounded_rectangle(coordinates(bounds), radius=round(radius * SCALE), fill=fill, outline=outline, width=round(width * SCALE))


def circle(draw, center, radius, fill=None, outline=CYAN, width=2):
    center_x, center_y = center
    draw.ellipse(coordinates((center_x-radius, center_y-radius, center_x+radius, center_y+radius)), fill=fill, outline=outline, width=round(width*SCALE))


def wrap(draw, value, size, max_width, bold=False):
    rows = []
    current = ''
    for word in value.split():
        candidate = (current + ' ' + word).strip()
        if draw.textlength(candidate, font=font(size, bold)) > max_width*SCALE and current:
            rows.append(current)
            current = word
        else:
            current = candidate
    if current:
        rows.append(current)
    return rows


def paragraph(draw, location, value, size=23, max_width=480, fill=MUTED):
    start_x, start_y = location
    for index, row in enumerate(wrap(draw, value, size, max_width)):
        text(draw, (start_x, start_y+index*(size+9)), row, size, fill)


def background():
    grid_y, grid_x = np.mgrid[0:HEIGHT, 0:WIDTH]
    glow = np.exp(-(((grid_x-WIDTH*.88)/WIDTH*.85)**2+((grid_y-HEIGHT*.28)/HEIGHT)**2)*4)
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for channel, (base, strength) in enumerate([(7, 8), (13, 23), (24, 30)]):
        pixels[:, :, channel] = np.clip(base+strength*glow, 0, 255)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for offset in range(0, 1280, 48):
        line(draw, [(offset, 0), (offset, 720)], '#162236', .5)
    for offset in range(0, 720, 48):
        line(draw, [(0, offset), (1280, offset)], '#162236', .5)
    return image


BASE = background()


def robot(draw, center, heading=0, radius=16, fill=RED):
    center_x, center_y = center
    angle = math.radians(heading)
    corners = []
    for horizontal, vertical in [(-1,-1),(-1,1),(1,1),(1,-1)]:
        corners.append((center_x+radius*(horizontal*math.cos(angle)+vertical*math.sin(angle)),
                        center_y+radius*(horizontal*math.sin(angle)-vertical*math.cos(angle))))
    draw.polygon([coordinates(point) for point in corners], fill='#162439', outline=fill, width=round(2*SCALE))
    line(draw, [corners[0], corners[2]], '#7388a4', 2)
    line(draw, [corners[1], corners[3]], '#7388a4', 2)
    line(draw, [(center_x, center_y), (center_x+radius*1.7*math.sin(angle), center_y-radius*1.7*math.cos(angle))], AMBER, 3)


def field(draw, state, bounds, focus=None, trail=None):
    left, top, size = bounds
    span = 140.4 if focus is None else focus[2]
    center_x, center_y = (0, 0) if focus is None else focus[:2]
    convert = lambda world_x, world_y: (left+size/2+(world_x-center_x)*size/span, top+size/2-(world_y-center_y)*size/span)
    box(draw, (left, top, left+size, top+size), '#0b1626', '#3a526b', 12)
    for offset in range(-70, 71, 24):
        point_a, point_b = convert(offset, -70.2), convert(offset, 70.2)
        if left <= point_a[0] <= left+size:
            line(draw, [(point_a[0], top), (point_a[0], top+size)], '#25354a', 1)
        point_a = convert(-70.2, offset)
        if top <= point_a[1] <= top+size:
            line(draw, [(left, point_a[1]), (left+size, point_a[1])], '#25354a', 1)
    for goal in state['goals']:
        point = convert(goal['x'], goal['y'])
        if left+12 < point[0] < left+size-12 and top+12 < point[1] < top+size-12:
            circle(draw, point, max(7, 4.25*size/span), '#18273a', RED if goal['alliance']=='red' else BLUE if goal['alliance']=='blue' else '#6c839c', 2)
    if trail:
        line(draw, [convert(item['robots'][0]['x'], item['robots'][0]['y']) for item in trail], '#e68fa1', 2)
    for item in state['objects']:
        point = convert(item['pose']['x'], item['pose']['y'])
        if not (left+5 < point[0] < left+size-5 and top+5 < point[1] < top+size-5):
            continue
        if item['kind']=='cup':
            circle(draw, point, max(3, 1.58*size/span), '#102332', '#b2d7e7', 1)
        else:
            color = {'red': RED, 'blue': BLUE, 'yellow': '#ffd967'}.get(item['halves'][item.get('upIndex', 0)], AMBER)
            circle(draw, point, max(2, 1.1*size/span), color, color, 1)
    for item in state['robots']:
        point = convert(item['x'], item['y'])
        if left+25 < point[0] < left+size-25 and top+25 < point[1] < top+size-25:
            robot(draw, point, item['theta'], 9*size/span, RED if item['id'].startswith('red') else BLUE)


def chart(draw, transitions, bounds, progress, color=CYAN):
    left, top, width, height = bounds
    poses = [item['evaluator']['poseInchesDegrees'] for item in transitions]
    minimum_x = min([0]+[pose[0] for pose in poses])-4
    maximum_x = max([24]+[pose[0] for pose in poses])+4
    minimum_y = min([0]+[pose[1] for pose in poses])-4
    maximum_y = max([24]+[pose[1] for pose in poses])+4
    convert = lambda pose: (left+(pose[0]-minimum_x)/(maximum_x-minimum_x)*width, top+height-(pose[1]-minimum_y)/(maximum_y-minimum_y)*height)
    box(draw, (left-15, top-15, left+width+15, top+height+15), '#0b1729', '#29415a', 12)
    count = max(2, min(len(poses), int(len(poses)*progress)))
    line(draw, [convert(pose) for pose in poses[:count]], color, 3)
    robot(draw, convert(poses[count-1]), poses[count-1][2], 11, color)


def render_frame(data, timestamp):
    image = BASE.copy()
    draw = ImageDraw.Draw(image)
    index = next((number for number, scene in enumerate(data['scenes']) if scene['start'] <= timestamp < scene['end']), 6)
    scene = data['scenes'][index]
    progress = min(1, (timestamp-scene['start'])/(scene['end']-scene['start']))
    text(draw, (54, 28), 'ROBO', 20, INK, True)
    text(draw, (119, 28), 'PROOF', 20, CYAN, True)
    text(draw, (930, 32), 'AMD ACT III  /  PROTOTYPE PITCH', 13, MUTED, True)
    line(draw, [(54, 65), (1226, 65)], '#2b425b', 1)
    text(draw, (54, 88), scene['eyebrow'], 14, CYAN if index != 5 else AMBER, True)
    if index == 0:
        text(draw, (54, 157), scene['title'], 64, INK, True)
        paragraph(draw, (59, 345), 'An evidence-first platform for improving robot movement.', 26, 500)
        box(draw, (60, 465, 516, 512), '#112c32', '#285c62', 8)
        text(draw, (77, 475), 'Simulation  +  Control  +  AI research', 21, CYAN, True)
        chart(draw, data['nominal']['transitions'], (760, 179, 398, 324), .2+.8*progress)
        text(draw, (755, 530), 'Actual controller trace • approximate simulation', 15, MUTED)
    elif index == 1:
        text(draw, (54, 135), scene['title'], 49, INK, True)
        paragraph(draw, (57, 283), 'One control core.\nNot a disconnected replacement robot.', 23, 490)
        cards = [(610, 158, 'SIMULATOR', '2D / 3D + game dynamics', CYAN), (610, 292, 'iraLIB / C++ WASM', 'LTV-LQR → wheel PI + feedforward', AMBER), (610, 426, 'ROBOPROOF', 'Stress tests • replay • experiment data', BLUE)]
        for number, (left, top, title, detail, color) in enumerate(cards):
            box(draw, (left, top, 1190, top+100))
            text(draw, (left+25, top+14), title, 24, color, True)
            text(draw, (left+25, top+53), detail, 21)
            if number < 2:
                line(draw, [(900, top+105), (900, top+128)], color, 2)
        text(draw, (57, 453), 'Controller binary source-checked', 18, CYAN)
        text(draw, (57, 486), data['identity']['controllerWasmSha256'][:28]+'…', 15, MUTED)
    elif index == 2:
        frames = data['scoring']['frames']
        frame_index = min(len(frames)-1, int(progress*(len(frames)-1)))
        state = frames[frame_index]
        field(draw, state, (56, 145, 416), (-47.09, -39.0, 58), frames[:frame_index+1])
        text(draw, (57, 574), 'Prepared scoring fixture • actual four-controller engine', 14, MUTED)
        text(draw, (518, 132), scene['title'], 47, INK, True)
        box(draw, (519, 267, 1192, 437))
        row = state['score']['red']
        text(draw, (546, 282), 'RULE-DERIVED LIVE SCORE', 16, CYAN, True)
        text(draw, (548, 320), f"RED  {row['total']}", 48, RED, True)
        text(draw, (858, 326), f"BLUE  {state['score']['blue']['total']}", 36, BLUE, True)
        text(draw, (548, 389), f"{row['coloredHalves']} × 5  +  {row['yellowHalves']} × 10  +  {row['midfieldRobots']} × 8  +  {row['autonomousBonus']}", 21)
        paragraph(draw, (528, 466), 'Pin / Cup / Cup + Pin capture. Finite lift. Points only after landing.', 24, 610)
        text(draw, (528, 543), f"Observed lift: {state['robots'][0]['lift']:.1f} in   |   Cup adds 0 direct points", 18, AMBER)
    elif index == 3:
        text(draw, (54, 131), scene['title'], 47, INK, True)
        success, count = data['evaluation']['success'], data['evaluation']['count']
        text(draw, (57, 286), f'{success} / {count}', 65, AMBER, True)
        paragraph(draw, (58, 378), 'Controller baseline on frozen stress worlds. Hard failures stay visible.', 22, 450)
        text(draw, (58, 488), 'Not a learned-policy improvement.', 19, MUTED)
        chart(draw, data['stress']['transitions'], (667, 189, 482, 251), .15+.85*progress, AMBER)
        text(draw, (655, 470), 'Exact failure replay • source-bound identity', 19, CYAN)
        text(draw, (655, 509), 'Record → reproduce → test a proposal → reject regressions', 17, MUTED)
        for tile in range(count):
            left = 59+(tile%12)*31
            top = 544+(tile//12)*24
            box(draw, (left, top, left+23, top+15), '#246556' if tile < success else '#633f29', None, 3)
    elif index == 4:
        text(draw, (54, 132), scene['title'], 47, INK, True)
        paragraph(draw, (58, 291), 'AI coordination is not a trained driving policy. We label the difference.', 23, 452)
        rows = [('Local Nemotron', 'Reviewed motion-task coordination', 'CONNECTED', CYAN),
                ('Neural failure predictor', 'CPU MLP • separate legacy plant', 'EXPERIMENTAL', AMBER),
                ('Movement learning', 'Original-engine transitions + Python bridge', 'TRAINING NEXT', BLUE),
                ('LocateAnything vision', 'Calibration / task contract preparation', 'FUTURE', MUTED)]
        for number, (title, detail, status, color) in enumerate(rows):
            top = 132+number*114
            box(draw, (569, top, 1221, top+100))
            text(draw, (590, top+11), title, 24, INK, True)
            text(draw, (590, top+49), detail, 18, MUTED)
            text(draw, (590, top+74), status, 12, color, True)
        text(draw, (57, 486), 'Human review before motion.', 23, CYAN, True)
    elif index == 5:
        text(draw, (54, 132), scene['title'], 47, INK, True)
        paragraph(draw, (59, 301), 'A measurable GPU roadmap, not an invented speedup.', 25, 440)
        text(draw, (60, 445), 'AMD execution + speedup: pending', 20, AMBER, True)
        steps = [('01', 'PACKAGE', 'Versioned datasets + bounded jobs'), ('02', 'TRAIN', 'ROCm / PyTorch on supported AMD'), ('03', 'EVALUATE', 'Frozen seeds + independent holdouts')]
        for number, (label, title, detail) in enumerate(steps):
            top = 163+number*128
            box(draw, (584, top, 1205, top+105), '#211d21', '#65503c')
            text(draw, (603, top+17), label, 33, AMBER, True)
            text(draw, (676, top+11), title, 24, INK, True)
            text(draw, (676, top+56), detail, 19, MUTED)
        text(draw, (588, 557), 'Verify real AMD workload before final submission.', 17, AMBER)
    else:
        text(draw, (54, 160), 'RoboProof', 94, INK, True)
        text(draw, (60, 296), 'Evidence before motion.', 37, CYAN, True)
        paragraph(draw, (62, 388), 'Learn better movement. Prove the improvement. Then test on hardware.', 27, 685)
        box(draw, (61, 522, 689, 567), '#112b32', '#386874', 10)
        text(draw, (79, 531), 'github.com/zhonibek/Nationals', 24, INK, True)
        frames = data['bundle']['frames']
        state = frames[min(len(frames)-1, int(progress*(len(frames)-1)))]
        field(draw, state, (848, 159, 335), (0, -29, 53))
        text(draw, (846, 510), 'Actual Cup + Pin carry', 18, AMBER)
    box(draw, (40, 610, 1240, 688), '#08121f', '#29415b', 13)
    caption_rows = wrap(draw, scene['narration'], 21, 1158)
    for number, row in enumerate(caption_rows[:2]):
        text(draw, (60, 622+number*28), row, 21, INK)
    line(draw, [(54, 707), (1226, 707)], '#24354b', 3)
    line(draw, [(54, 707), (54+1172*timestamp/60, 707)], CYAN if index != 5 else AMBER, 3)
    return image


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    return result


def audio(directory, scenes, executable):
    normalized = []
    for index, scene in enumerate(scenes):
        source = directory / f'narration-{index}.wav'
        with wave.open(str(source)) as waveform:
            duration = waveform.getnframes()/waveform.getframerate()
        length = scene['end']-scene['start']
        speed = max(.92, duration/(length-.5))
        if not .5 <= speed <= 2:
            raise ValueError(f'Narration {index} would need an unnatural tempo: {speed}')
        destination = directory / f'voice-{index}.wav'
        run([executable, '-y', '-v', 'error', '-i', str(source), '-af', f'atempo={speed:.7f},adelay=160,apad', '-t', str(length), '-ar', '48000', '-ac', '1', str(destination)])
        normalized.append(destination)
    concat = directory / 'voice-concat.txt'
    concat.write_text(''.join(f"file '{filename.as_posix()}'\n" for filename in normalized), encoding='utf8')
    run([executable, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat), '-c:a', 'pcm_s16le', str(directory/'voiceover.wav')])
    sample_rate = 48000
    times = np.arange(sample_rate*60)/sample_rate
    sound = .013*np.sin(2*np.pi*110*times)+.009*np.sin(2*np.pi*164.8138*times)
    sound *= np.minimum(1, times/2)*np.minimum(1, (60-times)/3)
    with wave.open(str(directory/'music-bed.wav'), 'wb') as waveform:
        waveform.setnchannels(1)
        waveform.setsampwidth(2)
        waveform.setframerate(sample_rate)
        waveform.writeframes((sound*32767).astype('<i2').tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--preview-only', action='store_true')
    arguments = parser.parse_args()
    directory = arguments.directory.resolve()
    data = json.loads((directory/'pitch-data.json').read_text(encoding='utf8'))
    previews = []
    for index, timestamp in enumerate([4, 11, 21, 29, 38, 47, 56]):
        preview = render_frame(data, timestamp)
        preview.save(directory/f'scene-{index+1}.png')
        previews.append(preview.resize((640, 360), Image.Resampling.LANCZOS))
    sheet = Image.new('RGB', (1280, 1440), '#071018')
    for index, preview in enumerate(previews):
        sheet.paste(preview, ((index%2)*640, (index//2)*360))
    sheet.save(directory/'contact-sheet.png')
    render_frame(data, 4).save(directory/'poster.png')
    def timestamp(seconds):
        return f'00:{int(seconds)//60:02d}:{int(seconds)%60:02d},000'
    subtitles = ''.join(f"{index+1}\n{timestamp(scene['start'])} --> {timestamp(scene['end'])}\n{scene['narration']}\n\n" for index, scene in enumerate(data['scenes']))
    (directory/'captions.srt').write_text(subtitles, encoding='utf8')
    if arguments.preview_only:
        print(directory/'contact-sheet.png', flush=True)
        return
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    audio(directory, data['scenes'], executable)
    destination = directory/'RoboProof-AMD-60s.mp4'
    command = [executable, '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS), '-i', 'pipe:0',
               '-i', str(directory/'voiceover.wav'), '-i', str(directory/'music-bed.wav'), '-filter_complex',
               '[1:a]loudnorm=I=-16:TP=-1.5:LRA=11[voice];[2:a]volume=0.2[bed];[voice][bed]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.94[audio]',
               '-map', '0:v', '-map', '[audio]', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '2', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-t', '60', '-movflags', '+faststart', str(destination)]
    with (directory/'encode.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in range(60*FPS):
                process.stdin.write(render_frame(data, frame/FPS).tobytes())
                if frame%240 == 0:
                    print(f'Rendered {frame/FPS:.0f}/60 seconds', flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError((directory/'encode.log').read_text()[-3000:])
        except BaseException:
            process.kill()
            process.wait()
            raise
    frame_count, duration = imageio_ffmpeg.count_frames_and_secs(str(destination))
    if frame_count != 60*FPS or abs(duration-60) > .05:
        raise ValueError(f'Incorrect video length: {frame_count} frames / {duration} seconds')
    reader = imageio_ffmpeg.read_frames(str(destination))
    metadata = next(reader)
    reader.close()
    verification = {'file':str(destination), 'durationSeconds':duration, 'frames':frame_count, 'resolution':metadata['size'], 'fps':metadata['fps'],
                    'bytes':destination.stat().st_size, 'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(), 'voice':'Local Windows Microsoft Zira; synthetic voice',
                    'visuals':'Animated motion graphics based on actual source-checked simulation data; not screen-recorded UI', 'motionPolicyTrained':False, 'amdGpuExecutionVerified':False}
    (directory/'video-verification.json').write_text(json.dumps(verification, indent=2), encoding='utf8')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
