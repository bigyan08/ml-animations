"""Run after visual review to verify the delivery and remove lesson build media."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[3] / 'media/machine_learning/neural_networks/neuron_to_network'


def main():
    final = OUTPUT / 'neuron_to_network.mp4'
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(final)]))
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in info['streams'] if s['codec_type'] == 'audio')
    assert (video['width'], video['height']) == (1080, 1920)
    assert video['avg_frame_rate'] == '30/1'
    assert 0 < float(info['format']['duration']) < 180
    assert abs(float(video['duration']) - float(audio['duration'])) < .05
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(final),
                    '-f', 'null', '-'], check=True)
    samples = np.frombuffer(subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(final), '-vn', '-ac', '1', '-ar', '16000',
        '-f', 'f32le', '-']), dtype=np.float32)
    assert np.sqrt(np.mean(samples * samples)) > .001, 'Audio is silent or too quiet'
    patterns = ['voice_*.mp3', 'voice_*.json', 'voice_*.partial', '*.storyboard*.png',
                'inspection*.png', 'preview*.mp4', 'preview*.srt', 'neuron_to_network_raw.mp4',
                'neuron_to_network_raw.srt', 'captions.ass', 'narration.json']
    files = {p for pattern in patterns for p in OUTPUT.glob(pattern)
             if p.is_file() and not p.is_symlink()}
    assert final not in files
    size = sum(p.stat().st_size for p in files)
    for path in files:
        path.unlink()
    assert hashlib.sha256(final.read_bytes()).hexdigest() == digest
    print(f"Verified {info['format']['duration']}s, 1080x1920, 30 fps, non-silent audio.")
    print(f'Removed {len(files)} intermediates ({size / 1024**2:.1f} MiB). Final MP4 unchanged.')


if __name__ == '__main__':
    main()
