"""Cached speech production driven by versionable narration JSON."""
import hashlib
import json
import math
import subprocess

from .captions import CaptionLayout, normalize
from .elevenlabs import elevenlabs_speech


async def edge_speech(text, voice, path):
    import edge_tts
    words = []
    talk = edge_tts.Communicate(text, voice.name, rate=voice.rate, boundary='WordBoundary')
    with path.open('wb') as stream:
        async for chunk in talk.stream():
            if chunk['type'] == 'audio':
                stream.write(chunk['data'])
            elif chunk['type'] == 'WordBoundary':
                words.append({'text': chunk['text'], 'start': chunk['offset']/1e7,
                              'duration': chunk['duration']/1e7})
    return words


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    try:
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class NarrationPipeline:
    def __init__(self, lesson, synthesize=None):
        self.lesson = lesson
        self.synthesize = synthesize
        self.layout = CaptionLayout(lesson)

    async def build(self, offline=False):
        lesson = self.lesson
        lesson.output.mkdir(parents=True, exist_ok=True)
        beats = []
        sources = lesson.source()
        for i, source in enumerate(sources):
            text = source['text']
            context = {}
            if lesson.voice.provider == 'edge':
                # Retain historical Edge cache filenames.
                cache_input = f'{lesson.voice.name}|{lesson.voice.rate}|{text}'
            else:
                if lesson.voice.model_id == 'eleven_multilingual_v2':
                    context = {'previous_text': sources[i-1]['text'] if i else '',
                               'next_text': sources[i+1]['text'] if i+1 < len(sources) else ''}
                cache_input = json.dumps({'version': 1, 'text': text, **context,
                                          **lesson.voice.synthesis_settings()}, sort_keys=True)
            key = hashlib.sha256(cache_input.encode()).hexdigest()[:12]
            audio = lesson.output / f'voice_{i+1:02d}_{key}.mp3'
            word_file = audio.with_suffix('.json')
            if not audio.exists() or not word_file.exists():
                if offline:
                    raise ValueError(f'Missing cached speech for {source["id"]}; offline mode will not contact TTS')
                temporary = audio.with_suffix('.partial')
                try:
                    if self.synthesize is not None:
                        words = await self.synthesize(text, lesson.voice, temporary)
                    elif lesson.voice.provider == 'edge':
                        words = await edge_speech(text, lesson.voice, temporary)
                    else:
                        words = await elevenlabs_speech(text, lesson.voice, temporary, **context)
                    temporary.replace(audio)
                    atomic_json(word_file, words)
                finally:
                    temporary.unlink(missing_ok=True)
            words = json.loads(word_file.read_text())
            tokens = text.split()
            if len(tokens) != len(words) or any(normalize(t) != normalize(w['text'])
                                                for t, w in zip(tokens, words)):
                raise ValueError(f'Speech tokenization differs for {source["id"]}; align script before captioning')
            duration = float(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(audio)]))
            last = -1
            for word in words:
                start = word['start']
                if not math.isfinite(start) or not 0 <= start < duration or start < last:
                    raise ValueError('Speech word timestamps are invalid')
                last = start
            caption_tokens = [w['text'] for w in words] if lesson.captions.provider_tokens else tokens
            captions = [{'text': ' '.join(caption_tokens[first:stop]),
                         'start': words[first]['start'],
                         'end': words[stop]['start'] if stop < len(words) else duration}
                        for first, stop in self.layout.groups(caption_tokens)]
            beats.append(dict(source, audio=audio.name, audio_duration=duration,
                              duration=duration+lesson.voice.pause, captions=captions))
            print(f'{source["title"]}: {duration:.2f}s', flush=True)
        result = {'provider': 'Microsoft Edge TTS' if lesson.voice.provider == 'edge' else 'ElevenLabs',
                  'voice': lesson.voice.name if lesson.voice.provider == 'edge' else lesson.voice.voice_id,
                  'voice_settings': lesson.voice.synthesis_settings(),
                  'build_fingerprint': lesson.fingerprint(), 'beats': beats}
        if lesson.voice.provider == 'edge':
            result['rate'] = lesson.voice.rate
        else:
            result['model'] = lesson.voice.model_id
        total = sum(b['duration'] for b in beats)
        if total >= lesson.render.max_duration:
            raise ValueError(f'{total:.2f}s exceeds configured duration limit')
        atomic_json(lesson.output/'narration.json', result)
        print(f'Total: {total:.2f}s')
        return result
