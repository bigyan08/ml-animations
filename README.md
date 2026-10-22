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

- [How does AI predict using its nearest neighbors?](animations/machine_learning/supervised_learning/knn_nearest_neighbors)
  — Follow one student through distances, K=1/3/5 voting, decision maps, feature scaling and a checked Python prediction in five minutes.
- [How does a machine find the best boundary?](animations/machine_learning/supervised_learning/svm_maximum_margin)
  — Compare separating lines, maximize the margin, identify support vectors, and extend to soft margins and RBF kernels in a five-minute SVM lesson.
- [How does a straight line learn?](animations/machine_learning/supervised_learning/straight_line_learns)
  — Follow study hours and exam scores through slope, intercept, squared error, gradient descent and Python in a five-minute beginner lesson.
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

Defaults provide ElevenLabs narration, measured Inter SemiBold captions, vertical
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

## Voice configuration

ElevenLabs is the default speech provider, including for existing manifests that
omit `voice.provider`. Set `ELEVENLABS_API_KEY` in the environment used to run the
pipeline. The key needs Text to Speech access and User read access for the credit
check. Credentials are never stored in lesson files. No additional SDK is needed.

The defaults use George (`JBFqnCBsd6RMkjVDRZzb`) with `eleven_multilingual_v2`.
This model provides original-text timestamps and neighboring-text context for
continuity between beats. Choose a voice available to your account:

```toml
[voice]
provider = "elevenlabs"
voice_id = "JBFqnCBsd6RMkjVDRZzb"
model_id = "eleven_multilingual_v2"
stability = 0.5
similarity_boost = 0.75
style = 0.0
speed = 0.95
use_speaker_boost = true
pause = 0.65
```

`speed` controls ElevenLabs delivery; `pause` adds silence after each beat.
Use plain spoken text in `narration.json`; audio tags and SSML are not supported
by this pipeline's caption/cue mapping. Other models must support the timestamp
endpoint and selected voice settings; neighboring-text context is sent only for
Multilingual v2. Audition any changed voice/model before final production.

Before each uncached ElevenLabs beat, the pipeline checks the remaining included
credits and reports the balance. Zero credits or an API insufficient-credit error
stops the command with a clear message and a nonzero exit status. It preserves
completed caches and the previous timing manifest. It never silently switches
providers or automatically retries a generation request. The balance check is
not a reservation or a guarantee against overages: concurrent usage and model or
voice costs can change the balance. Disable usage-based billing in your account
if you need a hard spending limit.

To explicitly use the existing Microsoft Edge provider, set:

```toml
[voice]
provider = "edge"
name = "en-US-AndrewNeural"
rate = "-5%"
pause = 0.65
```

`name` and `rate` apply only to Edge. Switching provider invalidates generated
timings; run `narrate` again before rendering. Historical Edge caches and timing
fingerprints remain usable when `provider = "edge"` is explicitly selected.
ElevenLabs caches include the provider, voice, model, synthesis settings, and
neighboring text when used. Changing only captions or the post-beat pause reuses
speech. Fully cached builds and `--offline` never contact either provider or
require a key. Narration timing metadata records the selected provider/settings.

API references: [speech with timestamps](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps)
and [subscription balance](https://elevenlabs.io/docs/api-reference/user/subscription/get).

Inspect the opening, densest equation, and ending with captions at phone size,
then review playback and sound. Automated checks complement visual review.
Only the root README is included among Markdown files when committing under the
current publishing convention; spoken JSON and lesson manifests remain tracked.
