# Mathematical animations

Short, English-narrated ManimGX videos for university learners and early-career
researchers. Default delivery: vertical 1080 × 1920, 30 fps, about one minute.

## Mathematics

### Optimization

- [Why gradient descent overshoots](animations/mathematics/optimization/gradient_descent_overshoot)
  — Compare two step sizes on one quadratic loss and see why crossing the minimum
  only raises loss when the step is too large for the curve.

## Machine learning

### Foundations

- [How does simple math turn into ChatGPT?](animations/machine_learning/foundations/simple_math_to_chatgpt)
  — Follow the complete bridge from a numerical prediction and correction to
  regression, deep networks, and language-model training.

### Supervised learning

- [How does AI actually learn?](animations/machine_learning/supervised_learning/house_price_learning_step)
  — Follow one house-price prediction through its loss, gradient, and a single
  parameter update.
- [Logistic regression: why gradient descent works](animations/machine_learning/supervised_learning/logistic_gradient_descent)
  — Follow two weights across the actual cross-entropy loss contours and see why
  the negative gradient decreases loss locally.

### Neural networks

- [One neuron → layer → network → application](animations/machine_learning/neural_networks/neuron_to_network)
  — Build a network from a worked neuron calculation, explain activation and
  training, and connect it to a medical-imaging workflow.

## Setup

Use Python 3.13+ and `uv sync` to install the declared dependencies. Narration
generation needs network access; renders reuse cached speech. Production uses
system `ffmpeg`/`ffprobe` for audio duration checks and captioned delivery.

The shared workflow below supports a fresh checkout. Local `AGENTS.md` adds
authoring guidelines.

## Shared production workflow

Each lesson contains `scene.py`, `lesson.toml`, and an authored `narration.json`.
Add `model.py` only for calculations worth keeping separate from animation.
Narration JSON is versioned source text; the `media/` file with the same name is
regenerable timing metadata. Markdown scripts are local planning/review documents.

The shared modules own narration, speech caching, caption measurement, export,
verification, cleanup, and beat timing. Keep lesson-specific diagrams and
explanations in the scene; use `shared.visuals` for common diagram primitives.

Run from the repository root, substituting the lesson directory:

```sh
lesson_path=animations/machine_learning/foundations/simple_math_to_chatgpt
.venv/bin/python -m shared narrate "$lesson_path"
.venv/bin/python -m shared render "$lesson_path" --preview
.venv/bin/python -m shared render "$lesson_path"
.venv/bin/python -m shared verify "$lesson_path"
# After the visual/audio review is complete:
.venv/bin/python -m shared finalize "$lesson_path" --reviewed
```

`narrate --offline` requires existing speech caches and never contacts the speech
service. `--media-root /tmp/video-build` isolates previews or experiments from
finished deliveries; use the same option on both narration and render commands.
Finalization verifies the configured resolution, frame rate, duration, audio,
and full-file decoding before removing known intermediates from that lesson.
It preserves the final MP4 and refuses unexpected files or symlinks.

For a new lesson, begin with a minimal `lesson.toml`:

```toml
title = "A useful prediction"
scene = "UsefulPrediction"
```

Defaults provide the project voice, measured Inter SemiBold captions, vertical
1080×1920 export, 30 fps, and a duration below 180 seconds. Override only settings
needed by the lesson; historical manifests retain previous styles intentionally.
A user-approved longer video sets `[render].max_duration` and an explicit
`duration_override_reason`. Voice settings live in `[voice]`; caption settings
live in `[captions]`. See `shared/config.py` for the supported configuration.

Author the spoken beats in `narration.json`, giving each a stable ID:

```json
{
  "beats": [
    {
      "id": "prediction",
      "title": "Make a prediction",
      "text": "The model predicts seven, but the target is ten."
    }
  ]
}
```

Build the scene on `shared.scenes.NarratedScene`. Start `construct()` with
`self.load_narration(__file__)`. `self.begin_beat("prediction")` starts that beat's
sound and captions; `self.cue("the target")` waits for its measured spoken cue;
`self.finish()` holds through the rest of the beat and detects animation overruns.
Scene composition and explanatory pacing remain customizable. Changing source
text, voice, or caption settings invalidates old timing metadata; run narration
again. Unchanged speech is reused from cache even when captions change.

The network TTS voice can change over time. Reusing exact recordings and timings
is necessary for exact audiovisual reproduction; cleanup deliberately removes
those generated caches unless an additional deliverable was explicitly requested.

Offline production checks:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Inspect the opening, densest equation, and ending with captions at phone size,
then review playback and sound. Automated checks complement visual review.
Only the root README is included among Markdown files when committing under the
current publishing convention; spoken JSON and lesson manifests remain tracked.
