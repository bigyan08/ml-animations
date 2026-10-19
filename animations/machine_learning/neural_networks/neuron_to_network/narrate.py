"""Generate cached speech and word-aligned captions before rendering."""

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
OUTPUT = ROOT / "media/machine_learning/neural_networks/neuron_to_network"


async def generate(voice, rate):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script = (HERE / "script.md").read_text()
    sections = re.findall(r"^## (\d{2}[^\n]*)\n(.+?)(?=\n## |\Z)", script, re.M | re.S)
    if len(sections) != 8:
        raise ValueError(f"Expected eight narration beats; found {len(sections)}")
    beats = []
    for i, (title, paragraph) in enumerate(sections):
        paragraph = " ".join(paragraph.split())
        key = hashlib.sha256(f"{voice}|{rate}|{paragraph}".encode()).hexdigest()[:12]
        audio = OUTPUT / f"voice_{i + 1:02d}_{key}.mp3"
        words_file = audio.with_suffix(".json")
        if not audio.exists() or not words_file.exists():
            words = []
            temporary = audio.with_suffix(".partial")
            talk = edge_tts.Communicate(paragraph, voice, rate=rate, boundary="WordBoundary")
            with temporary.open("wb") as stream:
                async for chunk in talk.stream():
                    if chunk["type"] == "audio":
                        stream.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        words.append({"text": chunk["text"], "start": chunk["offset"] / 1e7,
                                      "duration": chunk["duration"] / 1e7})
            temporary.replace(audio)
            words_file.write_text(json.dumps(words, indent=2))
        words = json.loads(words_file.read_text())
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio)
        ]))
        # Preserve punctuation when provider tokenization matches the script.
        tokens = paragraph.split()
        normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
        if len(tokens) != len(words) or any(normalize(t) != normalize(w["text"])
                                           for t, w in zip(tokens, words)):
            raise ValueError("Speech tokenization differs: align script before captioning")
        captions = []
        start_index = 0
        for j, token in enumerate(tokens):
            count = j - start_index + 1
            if count >= 6 or token.endswith((".", "?", "!", ",", ":")) or j == len(tokens) - 1:
                captions.append({"text": " ".join(tokens[start_index:j + 1]),
                                 "start": words[start_index]["start"],
                                 "end": words[j + 1]["start"] if j + 1 < len(words) else duration})
                start_index = j + 1
        beats.append({"title": title, "text": paragraph, "audio": audio.name,
                      "audio_duration": duration, "duration": duration + 0.8,
                      "captions": captions})
        print(f"{title}: {duration:.2f}s", flush=True)
    result = {"provider": "Microsoft Edge TTS", "voice": voice, "rate": rate, "beats": beats}
    (OUTPUT / "narration.json").write_text(json.dumps(result, indent=2))
    print(f"Total: {sum(b['duration'] for b in beats):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="en-US-AndrewNeural")
    parser.add_argument("--rate", default="-5%")
    args = parser.parse_args()
    asyncio.run(generate(args.voice, args.rate))
