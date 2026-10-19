"""Render the scene and burn word-aligned English captions into its MP4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = ROOT / "media/mathematics/optimization/gradient_descent_overshoot"
FONT_DIR = ROOT / "assets/fonts/ttf"
os.environ["FONTCONFIG_FILE"] = str(ROOT / "assets/fonts/fonts.conf")


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, centiseconds = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def main(preview=False, captions_only=False):
    narration = json.loads((OUTPUT / "narration.json").read_text())
    duration = sum(beat["duration"] for beat in narration["beats"])
    if not 0 < duration < 180:
        raise ValueError(f"Invalid short-form duration: {duration}")

    name = "preview" if preview else "gradient_descent_overshoot"
    raw = OUTPUT / f"{name}_raw.mp4"
    final = OUTPUT / f"{name}.mp4"
    resolution = "540x960" if preview else "1080x1920"
    renderer = Path(sys.executable).with_name("manimgx")
    if not captions_only:
        subprocess.run([
            str(renderer), "render", str(HERE / "scene.py"),
            "-r", resolution, "--fps", "30", "--preset", "fast",
            "-o", str(raw),
        ], check=True, cwd=ROOT, env=os.environ.copy())

    lines = ["""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Inter,70,&H00FAF2ED,&H00FAF2ED,&H0020100B,&H0020100B,0,0,0,0,100,100,0,0,1,2,0,2,115,145,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""]
    elapsed = 0
    for beat in narration["beats"]:
        for caption in beat["captions"]:
            words = caption["text"].replace("{", "").replace("}", "")
            words = r"\N".join(textwrap.wrap(
                words, width=26, break_long_words=False, break_on_hyphens=False
            ))
            start = ass_time(elapsed + caption["start"])
            end = ass_time(elapsed + caption["end"])
            lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{words}\n"
            )
        elapsed += beat["duration"]
    (OUTPUT / "captions.ass").write_text("".join(lines))

    filter_graph = (
        f"tpad=stop_mode=clone:stop_duration={duration},fps=30,"
        f"ass=captions.ass:fontsdir={FONT_DIR}"
    )
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(raw), "-vf", filter_graph, "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(final),
    ], check=True, cwd=OUTPUT)
    print(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--captions-only", action="store_true")
    args = parser.parse_args()
    main(args.preview, args.captions_only)
