"""One lesson manifest replaces copied production scripts and path constants."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VoiceSettings:
    name: str = 'en-US-AndrewNeural'
    rate: str = '-5%'
    pause: float = 0.65


@dataclass(frozen=True)
class CaptionStyle:
    font: str = 'Inter SemiBold'
    font_file: str = 'Inter-SemiBold.ttf'
    size: int = 56
    layout: str = 'measured'
    width: int = 770
    margin_left: int = 96
    margin_right: int = 180
    margin_bottom: int = 380
    grouping: str = 'phrases'
    max_words: int = 7
    clause_min_words: int = 3
    merge_short_tail: bool = False
    provider_tokens: bool = False
    break_long_words: bool = False
    break_on_hyphens: bool = False
    protected_phrases: tuple[str, ...] = (
        'gradient descent', 'machine learning', 'deep learning', 'neural network',
        'neural networks', 'next token', 'step size', 'weighted sums', 'does not',
    )


@dataclass(frozen=True)
class RenderSettings:
    width: int = 1080
    height: int = 1920
    preview_width: int = 540
    preview_height: int = 960
    fps: int = 30
    raw_crf: int = 18
    final_crf: int = 19
    preset: str = 'fast'
    max_duration: float = 180
    duration_override_reason: str = ''
    local_fonts: bool = True


@dataclass(frozen=True)
class LessonConfig:
    directory: Path
    root: Path
    title: str
    scene: str
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    render: RenderSettings = field(default_factory=RenderSettings)

    @classmethod
    def load(cls, path):
        directory = Path(path)
        if not directory.is_absolute():
            directory = ROOT / directory
        directory = directory.resolve()
        if directory.is_file():
            directory = directory.parent
        root = next((p for p in directory.parents if (p / 'pyproject.toml').is_file()), None)
        if root is None or not directory.is_relative_to(root / 'animations'):
            raise ValueError('A lesson must live inside the project animations directory')
        with (directory / 'lesson.toml').open('rb') as stream:
            data = tomllib.load(stream)
        allowed = {'title', 'scene', 'voice', 'captions', 'render'}
        if data.keys() - allowed:
            raise ValueError(f'Unknown lesson settings: {data.keys() - allowed}')
        caption_data = data.get('captions', {})
        if 'protected_phrases' in caption_data:
            caption_data['protected_phrases'] = tuple(caption_data['protected_phrases'])
        result = cls(directory, root, data['title'], data['scene'],
                     VoiceSettings(**data.get('voice', {})),
                     CaptionStyle(**caption_data), RenderSettings(**data.get('render', {})))
        result.validate()
        return result

    def validate(self):
        if self.voice.pause < 0 or not math.isfinite(self.voice.pause):
            raise ValueError('Voice pause must be finite and nonnegative')
        if self.captions.layout not in {'measured', 'characters'}:
            raise ValueError('Caption layout must be measured or characters')
        if self.captions.grouping not in {'phrases', 'fixed'}:
            raise ValueError('Caption grouping must be phrases or fixed')
        if min(self.captions.size, self.captions.width, self.captions.max_words,
               self.render.width, self.render.height, self.render.fps) <= 0:
            raise ValueError('Font, layout, and export dimensions must be positive')
        if not 0 < self.render.max_duration <= 3600:
            raise ValueError('Invalid duration limit')
        if self.render.max_duration > 180 and not self.render.duration_override_reason.strip():
            raise ValueError('A longer video needs an explicit duration override reason')
        if not self.scene.isidentifier():
            raise ValueError('Scene must name a Python scene class')

    @property
    def slug(self):
        return self.directory.name

    @property
    def media_root(self):
        return Path(os.environ.get('ML_ANIMATIONS_MEDIA_ROOT', self.root / 'media')).resolve()

    @property
    def output(self):
        return self.media_root / self.directory.relative_to(self.root / 'animations')

    @property
    def final(self):
        return self.output / f'{self.slug}.mp4'

    def source(self):
        data = json.loads((self.directory / 'narration.json').read_text())
        beats = data.get('beats', [])
        if not beats:
            raise ValueError('Narration must contain at least one beat')
        ids = set()
        for beat in beats:
            if not all(isinstance(beat.get(key), str) and beat[key].strip()
                       for key in ('id', 'title', 'text')):
                raise ValueError('Every narration beat needs id, title, and text')
            if beat['id'] in ids:
                raise ValueError(f'Duplicate narration id: {beat["id"]}')
            ids.add(beat['id'])
        return beats

    def fingerprint(self):
        value = {'beats': self.source(), 'voice': asdict(self.voice),
                 'captions': asdict(self.captions)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def generated(self):
        path = self.output / 'narration.json'
        if not path.is_file():
            raise ValueError('Generate narration first with: python -m shared narrate LESSON')
        data = json.loads(path.read_text())
        if data.get('build_fingerprint') != self.fingerprint():
            raise ValueError('Narration is stale; rerun the shared narrate command')
        source = self.source()
        beats = data.get('beats', [])
        if len(beats) != len(source):
            raise ValueError('Generated beat count differs from source')
        for current, authored in zip(beats, source):
            if any(current.get(key) != authored[key] for key in ('id', 'title', 'text')):
                raise ValueError('Generated narration differs from source')
            duration = current['duration']
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('Beat durations must be finite and positive')
            if Path(current['audio']).name != current['audio']:
                raise ValueError('Audio paths must be local filenames')
            if not (self.output / current['audio']).is_file():
                raise ValueError(f'Missing narration audio: {current["audio"]}')
            for caption in current['captions']:
                if not 0 <= caption['start'] < caption['end'] <= duration:
                    raise ValueError('Invalid caption timing')
        total = sum(b['duration'] for b in beats)
        if not total < self.render.max_duration:
            raise ValueError(f'{total:.2f}s exceeds lesson duration limit {self.render.max_duration}s')
        return data
