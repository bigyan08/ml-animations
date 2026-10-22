"""Offline API contracts: credits, failures, and caption alignment."""
import base64
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from shared.config import VoiceSettings
from shared.elevenlabs import (CreditsUnavailable, SpeechError, alignment_words,
                              check_credits, request_json, synthesize)


def alignment(text):
    return {'characters': list(text),
            'character_start_times_seconds': [i*.05 for i in range(len(text))],
            'character_end_times_seconds': [(i+1)*.05 for i in range(len(text))]}


class ElevenLabsTests(unittest.TestCase):
    def test_original_tokens_and_punctuation_keep_word_cue_indices(self):
        words = alignment_words('Loss: 3.5\nthen zero.', alignment('Loss: 3.5 then zero.'))
        self.assertEqual([w['text'] for w in words], ['Loss:', '3.5', 'then', 'zero.'])
        self.assertAlmostEqual(words[2]['start'], .5)
        for bad in (None, alignment('Different words'),
                    {**alignment('Hello'), 'character_start_times_seconds': [0]},
                    {**alignment('Hello'), 'character_end_times_seconds': [float('nan')]*5}):
            with self.subTest(bad=bad), self.assertRaises(SpeechError):
                alignment_words('Hello', bad)

    def test_missing_key_does_not_contact_service(self):
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': ''}), patch('shared.elevenlabs.urlopen') as call:
            with self.assertRaisesRegex(SpeechError, 'Set ELEVENLABS_API_KEY'):
                request_json('user/subscription')
            call.assert_not_called()

    def test_request_authentication_and_json_contract(self):
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-key'}), \
                patch('shared.elevenlabs.urlopen', return_value=io.BytesIO(b'{"ok": true}')) as call:
            self.assertEqual(request_json('test', {'text': 'Hello'}), {'ok': True})
            request = call.call_args.args[0]
            self.assertEqual(request.get_method(), 'POST')
            self.assertEqual(request.get_header('Xi-api-key'), 'test-key')
            self.assertEqual(json.loads(request.data), {'text': 'Hello'})
            self.assertEqual(call.call_args.kwargs['timeout'], 120)

    def test_exhausted_balance_stops_even_when_overages_are_enabled(self):
        with patch('shared.elevenlabs.request_json', return_value={
                'character_count': 100, 'character_limit': 100,
                'can_extend_character_limit': True}):
            with self.assertRaisesRegex(CreditsUnavailable, 'credits are exhausted'):
                check_credits()
        with patch('shared.elevenlabs.request_json', return_value={'character_count': 1}):
            with self.assertRaisesRegex(SpeechError, 'determine'):
                check_credits()

    def test_http_quota_auth_rate_limit_and_network_are_distinct(self):
        cases = [(401, 'quota_exceeded', CreditsUnavailable, 'credits'),
                 (403, 'insufficient_credits', CreditsUnavailable, 'credits'),
                 (401, 'invalid_api_key', SpeechError, 'rejected access'),
                 (403, 'missing_permissions', SpeechError, 'permissions'),
                 (402, 'payment_required', SpeechError, 'requires a paid plan'),
                 (429, 'too_many_concurrent_requests', SpeechError, 'concurrency'),
                 (500, '', SpeechError, 'HTTP 500')]
        for code, status, expected, message in cases:
            body = io.BytesIO(json.dumps({'detail': {'status': status, 'message': 'SECRET'}}).encode())
            error = HTTPError('https://api.elevenlabs.io', code, 'failure', {}, body)
            with self.subTest(code=code, status=status), patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'SECRET'}), \
                    patch('shared.elevenlabs.urlopen', side_effect=error) as call:
                with self.assertRaisesRegex(expected, message) as caught:
                    request_json('test', {'text': 'Hello'})
                self.assertNotIn('SECRET', str(caught.exception))
                self.assertEqual(call.call_count, 1)
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'SECRET'}), \
                patch('shared.elevenlabs.urlopen', side_effect=URLError('SECRET')):
            with self.assertRaisesRegex(SpeechError, 'network'):
                request_json('test')

    def test_timestamp_request_and_audio_decoding(self):
        response = {'audio_base64': base64.b64encode(b'audio').decode(),
                    'alignment': alignment('Hello world.')}
        with tempfile.TemporaryDirectory() as directory, \
                patch('shared.elevenlabs.check_credits', return_value=100), \
                patch('shared.elevenlabs.request_json', return_value=response) as request:
            path = Path(directory)/'speech.mp3'
            words = synthesize('Hello world.', VoiceSettings(), path, 'Before.', 'After.')
            self.assertEqual(path.read_bytes(), b'audio')
            self.assertEqual([w['text'] for w in words], ['Hello', 'world.'])
            endpoint, payload = request.call_args.args
            self.assertIn('/with-timestamps?output_format=mp3_44100_128', endpoint)
            self.assertEqual(payload['previous_text'], 'Before.')
            self.assertEqual(payload['next_text'], 'After.')
            self.assertEqual(payload['voice_settings']['speed'], .95)
            synthesize('Hello world.', replace(VoiceSettings(), model_id='eleven_v3'), path)
            self.assertNotIn('previous_text', request.call_args.args[1])

    def test_no_synthesis_after_empty_balance(self):
        with patch('shared.elevenlabs.request_json', return_value={
                'character_count': 100, 'character_limit': 100}) as request:
            with self.assertRaises(CreditsUnavailable):
                synthesize('Hello', VoiceSettings(), Path('/unused'))
            request.assert_called_once_with('user/subscription')


if __name__ == '__main__':
    unittest.main()
