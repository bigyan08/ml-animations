# Mathematical animations

Short, English-narrated ManimGX videos for university learners and early-career
researchers. Default delivery: vertical 1080 × 1920, 30 fps, about one minute.

## Mathematics

### Optimization

- [Why gradient descent overshoots](animations/mathematics/optimization/gradient_descent_overshoot/README.md)
  — Compare two step sizes on one quadratic loss and see why crossing the minimum
  only raises loss when the step is too large for the curve.

## Machine learning

### Foundations

- [How does simple math turn into ChatGPT?](animations/machine_learning/foundations/simple_math_to_chatgpt/README.md)
  — Follow the complete bridge from a numerical prediction and correction to
  regression, deep networks, and language-model training.

### Supervised learning

- [How does AI actually learn?](animations/machine_learning/supervised_learning/house_price_learning_step/README.md)
  — Follow one house-price prediction through its loss, gradient, and a single
  parameter update.
- [Logistic regression: why gradient descent works](animations/machine_learning/supervised_learning/logistic_gradient_descent/README.md)
  — Follow two weights across the actual cross-entropy loss contours and see why
  the negative gradient decreases loss locally.

### Neural networks

- [One neuron → layer → network → application](animations/machine_learning/neural_networks/neuron_to_network/README.md)
  — Build a network from a worked neuron calculation, explain activation and
  training, and connect it to a medical-imaging workflow.

## Setup

Use Python 3.13+ and `uv sync` to install the declared dependencies. Narration
generation needs network access; renders reuse cached speech. This lesson uses
system `ffmpeg`/`ffprobe` for audio duration checks and captioned delivery.

See [AGENTS.md](AGENTS.md) for organization and production conventions.
