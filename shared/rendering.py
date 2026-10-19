"""Shared renderer, delivery verification, and scoped finalization."""
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np

from .captions import CaptionLayout


class Renderer:
    def __init__(self, lesson):
        self.lesson = lesson

    def render(self, preview=False, captions_only=False, resolution=None):
        lesson, settings = self.lesson, self.lesson.render
        narration = lesson.generated()
        duration = sum(beat['duration'] for beat in narration['beats'])
        name = 'preview' if preview else lesson.slug
        raw, final = lesson.output/f'{name}_raw.mp4', lesson.output/f'{name}.mp4'
        width, height = resolution or ((settings.preview_width, settings.preview_height)
                                      if preview else (settings.width, settings.height))
        env = os.environ.copy()
        if settings.local_fonts:
            env['FONTCONFIG_FILE'] = str(lesson.root/'assets/fonts/fonts.conf')
        env['PYTHONPATH'] = str(lesson.root)+os.pathsep+env.get('PYTHONPATH', '')
        env['ML_ANIMATIONS_MEDIA_ROOT'] = str(lesson.media_root)
        if not captions_only:
            subprocess.run([
                str(Path(sys.executable).with_name('manimgx')), 'render',
                str(lesson.directory/'scene.py'), lesson.scene, '-r', f'{width}x{height}',
                '--fps', str(settings.fps), '--preset', settings.preset,
                '--crf', str(23 if preview else settings.raw_crf), '-o', str(raw),
            ], check=True, cwd=lesson.root, env=env)
        if not raw.is_file():
            raise ValueError(f'Missing raw render: {raw}')
        (lesson.output/'captions.ass').write_text(CaptionLayout(lesson).ass(narration))
        # A local filename keeps user-selected media paths out of the ASS filter syntax.
        filters = f'tpad=stop_mode=clone:stop_duration={duration},fps={settings.fps},ass=captions.ass'
        if settings.local_fonts:
            filters += ':fontsdir=fonts'
        # Use a temporary relative font link to avoid filter escaping arbitrary root paths.
        fonts = lesson.output/'fonts'
        if fonts.exists() or fonts.is_symlink():
            raise ValueError(f'Unexpected temporary font path: {fonts}')
        try:
            if settings.local_fonts:
                fonts.symlink_to(lesson.root/'assets/fonts/ttf', target_is_directory=True)
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', str(raw),
                '-vf', filters, '-t', str(duration), '-c:v', 'libx264', '-preset', settings.preset,
                '-crf', str(settings.final_crf), '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                '-b:a', '192k', '-movflags', '+faststart', str(final),
            ], check=True, cwd=lesson.output, env=env)
        finally:
            if settings.local_fonts and fonts.is_symlink():
                fonts.unlink()
        print(final)
        return final


def verify(lesson, final=None):
    final = Path(final or lesson.final)
    if not final.is_file():
        raise ValueError(f'Missing delivery: {final}')
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(final)]))
    video = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    audio = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
    if not video or not audio:
        raise ValueError('Delivery must contain video and audio')
    settings = lesson.render
    if (video['width'], video['height']) != (settings.width, settings.height):
        raise ValueError('Delivery resolution differs from lesson configuration')
    if Fraction(video['avg_frame_rate']) != settings.fps:
        raise ValueError('Delivery frame rate differs from lesson configuration')
    if not 0 < float(info['format']['duration']) < settings.max_duration:
        raise ValueError('Delivery duration exceeds configured limit')
    if abs(float(video['duration'])-float(audio['duration'])) > .1:
        raise ValueError('Audio and video durations differ by more than 0.1 seconds')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(final), '-f', 'null', '-'], check=True)
    samples = np.frombuffer(subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(final), '-vn', '-ac', '1', '-ar', '16000',
        '-f', 'f32le', '-']), dtype=np.float32)
    if not len(samples) or np.sqrt(np.mean(samples*samples)) <= .001:
        raise ValueError('Delivery audio is silent or too quiet')
    return info


def finalize(lesson, reviewed=False):
    if not reviewed:
        raise ValueError('Finish visual/audio review before using finalize --reviewed')
    if lesson.output.is_symlink() or lesson.output.resolve() != lesson.output or lesson.final.is_symlink():
        raise ValueError('Unexpected symlink in delivery path; inspect before cleanup')
    info = verify(lesson)
    final = lesson.final
    digest = hashlib.sha256(final.read_bytes()).digest()
    patterns = ('voice_*', 'fixture_*', '*_raw.mp4', '*_raw.srt', '*.storyboard*.png',
                'preview.mp4', 'preview.srt', 'captions.ass', 'narration.json', 'inspection*.png')
    allowed = {p for pattern in patterns for p in lesson.output.glob(pattern)}
    others = set(lesson.output.iterdir())-{final}
    if others-allowed or any(not p.is_file() or p.is_symlink() for p in others):
        raise ValueError('Unexpected files in lesson media directory; inspect before cleanup')
    for path in others:
        path.unlink()
    if hashlib.sha256(final.read_bytes()).digest() != digest:
        raise RuntimeError('Final delivery changed during cleanup')
    return info, len(others)
