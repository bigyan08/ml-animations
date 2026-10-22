"""ElevenLabs speech and original-text timestamps, without an SDK dependency."""
import asyncio
import base64
import binascii
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class SpeechError(ValueError):
    """An actionable provider failure that the CLI can report without a traceback."""


class CreditsUnavailable(SpeechError):
    """The account or API key has insufficient generation credits."""


CREDIT_MESSAGE = ('ElevenLabs credits are exhausted or insufficient for this request. '
                  'Narration stopped; existing caches are preserved. Refill credits or '
                  'wait for renewal, or explicitly set [voice] provider = "edge" '
                  'in lesson.toml to use Microsoft Edge TTS.')


def request_json(path, payload=None):
    key = os.environ.get('ELEVENLABS_API_KEY', '').strip()
    if not key:
        raise SpeechError('Set ELEVENLABS_API_KEY in your environment to generate narration. '
                          'Alternatively, explicitly set [voice] provider = "edge" in lesson.toml.')
    request = Request('https://api.elevenlabs.io/v1/' + path,
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={'xi-api-key': key, 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as error:
        try:
            detail = json.loads(error.read()).get('detail', {})
            status = detail.get('status', '') if isinstance(detail, dict) else ''
        except (ValueError, AttributeError):
            status = ''
        # Never echo the response body: it may contain submitted text or credentials.
        if status in {'quota_exceeded', 'insufficient_credits', 'credit_limit_exceeded'}:
            raise CreditsUnavailable(CREDIT_MESSAGE) from None
        if error.code == 402:
            raise SpeechError('ElevenLabs requires a paid plan for this voice or model. '
                              'Available credits do not grant access to library voices '
                              'on the free API plan. Select an accessible voice or upgrade.') from None
        if error.code in {401, 403}:
            raise SpeechError('ElevenLabs rejected access. Check ELEVENLABS_API_KEY and '
                              'its Text to Speech and User read permissions.') from None
        if error.code == 429:
            raise SpeechError('ElevenLabs rate/concurrency limit reached. Retry later; '
                              'this does not necessarily mean credits are exhausted.') from None
        raise SpeechError(f'ElevenLabs request failed (HTTP {error.code}). '
                          'Check the selected voice/model and account access.') from None
    except (URLError, TimeoutError, OSError):
        raise SpeechError('Could not reach ElevenLabs. Check the network and retry. '
                          'No automatic generation retry was attempted.') from None
    except ValueError:
        raise SpeechError('ElevenLabs returned invalid JSON.') from None
    if not isinstance(result, dict):
        raise SpeechError('ElevenLabs returned an invalid response.')
    return result


def check_credits():
    subscription = request_json('user/subscription')
    used = subscription.get('character_count')
    limit = subscription.get('character_limit')
    if type(used) is not int or type(limit) is not int or min(used, limit) < 0:
        raise SpeechError('Could not determine ElevenLabs remaining credits; narration stopped.')
    remaining = max(0, limit - used)
    if remaining == 0:
        raise CreditsUnavailable(CREDIT_MESSAGE)
    return remaining


def alignment_words(text, alignment):
    """Keep authored tokens so captions and scene cues retain the same indices."""
    try:
        characters = alignment['characters']
        starts = alignment['character_start_times_seconds']
        ends = alignment['character_end_times_seconds']
        if (not all(isinstance(c, str) and len(c) == 1 for c in characters)
                or len(characters) != len(starts) or len(starts) != len(ends)):
            raise ValueError
        aligned = ''.join(characters)
        matches = list(re.finditer(r'\S+', aligned))
        if [m.group() for m in matches] != text.split():
            raise ValueError
        last = -1
        for start, end in zip(starts, ends):
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end or start < last:
                raise ValueError
            last = start
        words = []
        for token, match in zip(text.split(), matches):
            start, end = starts[match.start()], ends[match.end()-1]
            words.append({'text': token, 'start': start, 'duration': end-start})
        return words
    except (KeyError, TypeError, ValueError):
        raise SpeechError('ElevenLabs original-text alignment is missing or invalid. '
                          'Use plain spoken text and a model supporting timestamps.') from None


def synthesize(text, voice, path, previous_text='', next_text=''):
    remaining = check_credits()
    print(f'ElevenLabs: {remaining:,} credits remaining before this beat.', flush=True)
    payload = {
        'text': text, 'model_id': voice.model_id,
        'voice_settings': {key: getattr(voice, key) for key in
                           ('stability', 'similarity_boost', 'style', 'speed', 'use_speaker_boost')},
    }
    # Request stitching is supported by v2; v3 does not accept these context fields.
    if voice.model_id == 'eleven_multilingual_v2':
        payload.update(previous_text=previous_text, next_text=next_text)
    result = request_json('text-to-speech/' + quote(voice.voice_id, safe='')
                          + '/with-timestamps?output_format=mp3_44100_128', payload)
    words = alignment_words(text, result.get('alignment'))
    try:
        audio = base64.b64decode(result['audio_base64'], validate=True)
        if not audio:
            raise ValueError
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise SpeechError('ElevenLabs returned invalid audio.') from None
    path.write_bytes(audio)
    return words


async def elevenlabs_speech(text, voice, path, previous_text='', next_text=''):
    return await asyncio.to_thread(synthesize, text, voice, path, previous_text, next_text)
