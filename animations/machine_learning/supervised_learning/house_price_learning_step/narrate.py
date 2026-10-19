"""Generate narration and word-aligned captions for this lesson."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import subprocess

import edge_tts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = ROOT / "media/machine_learning/supervised_learning/house_price_learning_step"


def caption_groups(tokens):
    groups = []
    start = 0
    for index, token in enumerate(tokens):
        count = index - start + 1
        if (count == 7 or token.endswith((".", "!", "?"))
                or (count >= 3 and token.endswith((",", ";", ":")))):
            groups.append((start, index + 1))
            start = index + 1
    if start < len(tokens):
        groups.append((start, len(tokens)))
    if len(groups) > 1 and groups[-1][1] - groups[-1][0] < 3:
        groups[-2:] = [(groups[-2][0], groups[-1][1])]
    return groups


async def generate(voice, rate):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script = (HERE / "script.md").read_text()
    sections = re.findall(
        r"^## \d{2} — ([^\n]+)\n(.+?)(?=\n## |\Z)",
        script, re.M | re.S,
    )
    if len(sections) != 6:
        raise ValueError(f"Expected six narration beats; found {len(sections)}")

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
