"""Generate narration and word-aligned captions for this lesson."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import subprocess

import edge_tts

from render import caption_lines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = ROOT / "media/machine_learning/foundations/simple_math_to_chatgpt"


def caption_groups(tokens):
    """Segment phrases while preserving important multiword technical concepts."""
    protected = [
        'recommendation systems', 'image AI', 'gradient descent', 'machine learning', 'deep learning', 'neural network',
        'neural networks', 'seven point six', 'negative three', 'weighted sums',
        'nonlinear transformations', 'step size', 'slope and intercept',
        'next token', 'training text', 'can still', 'does not', 'not guarantee',
        'real world', 'actual answer', 'further training', 'the other way',
        'negative three squared', 'linear function', 'linear operations',
        'in image models', 'small step', 'useful responses',
    ]
    normalized = [re.sub(r'[^a-z0-9]', '', t.lower()) for t in tokens]
    blocked = set()
    for phrase in protected:
        terms = phrase.split()
        for i in range(len(tokens)-len(terms)+1):
            if normalized[i:i+len(terms)] == terms:
                blocked.update(range(i+1,i+len(terms)))
    groups = []
    start = 0
    while start < len(tokens):
        stop = min(start+7,len(tokens))
        while stop in blocked: stop -= 1
        for i in range(start, stop):
            end = i+1
            if end not in blocked and (tokens[i].endswith(('.', '!', '?')) or
                (end-start >= 3 and tokens[i].endswith((',', ';', ':')))):
                stop=end
                break
        while True:
            try:
                caption_lines(' '.join(tokens[start:stop]))
                break
            except ValueError:
                stop -= 1
                while stop in blocked: stop -= 1
                if stop <= start: raise ValueError('Protected caption phrase cannot fit')
        groups.append((start,stop))
        start=stop
    return groups


async def generate(voice, rate):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script = (HERE / "script.md").read_text()
    sections = re.findall(
        r"^## \d{2} — ([^\n]+)\n(.+?)(?=\n## |\Z)",
        script, re.M | re.S,
    )
    if len(sections) != 11:
        raise ValueError(f"Expected eleven narration beats; found {len(sections)}")

    beats = []
    for i, (title, paragraph) in enumerate(sections):
        paragraph = " ".join(paragraph.split())
        key = hashlib.sha256(f"{voice}|{rate}|{paragraph}".encode()).hexdigest()[:12]
        audio = OUTPUT / f"voice_{i + 1:02d}_{key}.mp3"
        words_file = audio.with_suffix(".json")
        if not audio.exists() or not words_file.exists():
            words = []
            temporary = audio.with_suffix(".partial")
            talk = edge_tts.Communicate(paragraph, voice, rate=rate,
                                        boundary="WordBoundary")
            with temporary.open("wb") as stream:
                async for chunk in talk.stream():
                    if chunk["type"] == "audio":
                        stream.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        words.append({
                            "text": chunk["text"],
                            "start": chunk["offset"] / 1e7,
                            "duration": chunk["duration"] / 1e7,
                        })
            temporary.replace(audio)
            words_file.write_text(json.dumps(words, indent=2))

        words = json.loads(words_file.read_text())
        source_tokens = paragraph.split()
        if len(source_tokens) != len(words):
            raise ValueError(
                f"Word timing mismatch for {title}: {len(source_tokens)} script words, "
                f"{len(words)} audio words"
            )
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
        ]))
        captions = []
        for first, stop in caption_groups(source_tokens):
            captions.append({
                "text": " ".join(source_tokens[first:stop]),
                "start": words[first]["start"],
                "end": words[stop]["start"] if stop < len(words) else duration,
            })
        beats.append({
            "title": title, "text": paragraph, "audio": audio.name,
            "audio_duration": duration, "duration": duration + 0.65,
            "captions": captions,
        })
        print(f"{title}: {duration:.2f}s", flush=True)

    result = {"provider": "Microsoft Edge TTS", "voice": voice, "rate": rate,
              "beats": beats}
    (OUTPUT / "narration.json").write_text(json.dumps(result, indent=2))
    print(f"Total: {sum(beat['duration'] for beat in beats):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="en-US-AndrewNeural")
    parser.add_argument("--rate", default="-5%")
    args = parser.parse_args()
    asyncio.run(generate(args.voice, args.rate))
