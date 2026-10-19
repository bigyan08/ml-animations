"""Offline contracts for shared production, timing, and safe cleanup."""
import asyncio
from dataclasses import replace
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from shared.captions import CaptionLayout, ass_time
from shared.config import CaptionStyle, LessonConfig, ROOT, RenderSettings
from shared.narration import NarrationPipeline
from shared.rendering import finalize, verify
from shared.scenes import NarratedScene


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        self.directory = self.root/'animations/math/optimization/example'
        self.directory.mkdir(parents=True)
        (self.directory/'lesson.toml').write_text('title="Example"\nscene="Example"\n')
        self.source = {'beats': [
            {'id':'predict', 'title':'Prediction', 'text':'Try gradient descent now.'},
            {'id':'check', 'title':'Check', 'text':'Measure the new loss.'},
        ]}
        self.write_source()
        font_parent = self.root/'assets/fonts'
        font_parent.mkdir(parents=True)
        (font_parent/'ttf').symlink_to(ROOT/'assets/fonts/ttf', target_is_directory=True)
        self.env = patch.dict(os.environ, {'ML_ANIMATIONS_MEDIA_ROOT':str(self.root/'media')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)
        self.lesson = LessonConfig.load(self.directory)
        self.calls = []

    def write_source(self):
        (self.directory/'narration.json').write_text(json.dumps(self.source))

    async def speech(self, text, voice, path):
        self.calls.append((text, voice.name, voice.rate))
        duration = len(text.split())*.2+.3
        samples = (np.sin(2*np.pi*440*np.arange(int(duration*16000))/16000)*1500).astype('<i2')
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(samples.tobytes())
        return [{'text':t.strip('.'), 'start':i*.2, 'duration':.15}
                for i,t in enumerate(text.split())]

    def build(self, lesson=None, offline=False, synthesize=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(NarrationPipeline(lesson or self.lesson, synthesize or self.speech).build(offline))

    def test_cache_reuses_speech_but_invalidates_changed_text_and_voice(self):
        self.build()
        self.build(offline=True)
        self.assertEqual(len(self.calls), 2)
        self.source['beats'][0]['text'] = 'Try a different prediction.'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.lesson.generated()
        self.build()
        self.assertEqual(len(self.calls), 3)
        changed = replace(self.lesson, voice=replace(self.lesson.voice, rate='+5%'))
        self.build(changed)
        self.assertEqual(len(self.calls), 5)

    def test_caption_change_rebuilds_metadata_without_regenerating_audio(self):
        self.build()
        changed = replace(self.lesson, captions=replace(self.lesson.captions, margin_bottom=400))
        with self.assertRaisesRegex(ValueError, 'stale'):
            changed.generated()
        self.build(changed, offline=True)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(changed.generated()['beats']), 2)

    def test_offline_cache_miss_never_calls_provider(self):
        with self.assertRaisesRegex(ValueError, 'offline mode'):
            self.build(offline=True)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.lesson.output/'narration.json').exists())

    def test_provider_failure_preserves_previous_manifest_and_removes_partial(self):
        self.build()
        manifest = self.lesson.output/'narration.json'
        old = manifest.read_bytes()
        self.source['beats'][0]['text'] = 'Changed narration.'
        self.write_source()
        async def fail(text, voice, path):
            path.write_bytes(b'incomplete')
            raise RuntimeError('service unavailable')
        with self.assertRaisesRegex(RuntimeError, 'service unavailable'):
            self.build(synthesize=fail)
        self.assertEqual(manifest.read_bytes(), old)
        self.assertEqual(list(self.lesson.output.glob('*.partial')), [])

    def test_equal_word_count_still_rejects_wrong_alignment(self):
        async def incorrect(text, voice, path):
            words = await self.speech(text, voice, path)
            words[0]['text'] = 'Wrong'
            return words
        with self.assertRaisesRegex(ValueError, 'tokenization differs'):
            self.build(synthesize=incorrect)
        self.assertFalse((self.lesson.output/'narration.json').exists())

    def test_phrase_groups_preserve_words_and_fit_two_measured_lines(self):
        style = replace(self.lesson.captions, protected_phrases=('negative three squared',))
        layout = CaptionLayout(replace(self.lesson, captions=style))
        tokens = 'Recommendation systems use negative three squared as an illustrative calculation here.'.split()
        groups = layout.groups(tokens)
        rendered = [' '.join(tokens[a:b]) for a,b in groups]
        self.assertEqual(' '.join(rendered), ' '.join(tokens))
        self.assertTrue(any('negative three squared' in line for line in rendered))
        for text in rendered:
            lines = layout.lines(text).split(r'\N')
            self.assertLessEqual(len(lines), 2)
            self.assertTrue(all(layout.font.getlength(line) <= style.width for line in lines))
        tiny = replace(style, max_words=2)
        with self.assertRaisesRegex(ValueError, 'protected phrase exceeds'):
            CaptionLayout(replace(self.lesson, captions=tiny)).groups('negative three squared'.split())

    def test_ass_uses_cumulative_beat_offsets_and_removes_override_syntax(self):
        layout = CaptionLayout(self.lesson)
        content = layout.ass({'beats':[
            {'duration':2,'captions':[{'text':'First.','start':0,'end':1}]},
            {'duration':2,'captions':[{'text':r'{\pos(1,1)}Second.','start':.5,'end':1.5}]},
        ]})
        self.assertIn('0:00:02.50,0:00:03.50', content)
        self.assertNotIn(r'\pos', content)
        self.assertEqual(ass_time(3599.999), '1:00:00.00')

    def test_source_rejects_duplicate_ids_and_config_requires_long_video_reason(self):
        self.source['beats'][1]['id'] = 'predict'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.lesson.source()
        long = replace(self.lesson, render=replace(self.lesson.render, max_duration=300))
        with self.assertRaisesRegex(ValueError, 'override reason'):
            long.validate()
        replace(long, render=replace(long.render, duration_override_reason='User requested complete version')).validate()

    def test_scene_timing_uses_word_cues_and_detects_overruns(self):
        self.build()
        class Harness(NarratedScene):
            def __init__(self):
                self.clock = 0
                self.sounds = []
                self.subcaptions = []
            @property
            def time(self):
                return self.clock
            def wait(self, duration):
                self.clock += duration
            def add_sound(self, path):
                self.sounds.append(path)
            def add_subcaption(self, text, **kwargs):
                self.subcaptions.append(text)
        scene = Harness()
        # load_narration accepts an actual scene path, as production does.
        (self.directory/'scene.py').touch()
        scene.load_narration(self.directory/'scene.py')
        duration = scene.begin_beat('predict')
        scene.cue('gradient descent')
        self.assertAlmostEqual(scene.time, .2)
        with self.assertRaisesRegex(ValueError, 'not found'):
            scene.cue('nonexistent cue')
        scene.finish()
        self.assertAlmostEqual(scene.time, duration)
        scene.begin_beat('check')
        scene.wait(5)
        with self.assertRaisesRegex(ValueError, 'exceeds narration'):
            scene.finish()

    def make_delivery(self):
        settings = RenderSettings(width=160, height=288, fps=15)
        lesson = replace(self.lesson, render=settings)
        lesson.output.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=navy:s=160x288:r=15:d=1',
                        '-f','lavfi','-i','sine=frequency=440:duration=1','-shortest','-c:v','libx264',
                        '-pix_fmt','yuv420p','-c:a','aac',str(lesson.final)],check=True)
        return lesson

    def test_real_ffmpeg_delivery_verification_and_scoped_cleanup(self):
        lesson = self.make_delivery()
        self.assertEqual(verify(lesson)['streams'][0]['width'],160)
        digest = hashlib.sha256(lesson.final.read_bytes()).digest()
        cache = lesson.output/'voice_01.mp3'
        cache.write_bytes(b'cache')
        unknown = lesson.output/'keep.txt'
        unknown.write_text('user data')
        with self.assertRaisesRegex(ValueError,'Unexpected files'):
            finalize(lesson,reviewed=True)
        self.assertTrue(cache.exists())
        self.assertTrue(unknown.exists())
        unknown.unlink()
        target = self.root/'outside.mp3'
        target.write_bytes(b'outside')
        link = lesson.output/'voice_link.mp3'
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'Unexpected files'):
            finalize(lesson,reviewed=True)
        self.assertEqual(target.read_bytes(),b'outside')
        link.unlink()
        with self.assertRaisesRegex(ValueError,'review'):
            finalize(lesson)
        _, count = finalize(lesson,reviewed=True)
        self.assertEqual(count,1)
        self.assertEqual(list(lesson.output.iterdir()),[lesson.final])
        self.assertEqual(hashlib.sha256(lesson.final.read_bytes()).digest(),digest)


if __name__ == '__main__':
    unittest.main()
