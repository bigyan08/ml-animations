"""Follow one house-price example through one supervised-learning update."""


import manimgx as m

from shared.scenes import NarratedScene

from model import ACTUAL_PRICE, INITIAL_W, LEARNING_RATE, SIZE, gradient, loss, prediction, update


m.config.pixel_width = 1080
m.config.pixel_height = 1920
m.config.frame_rate = 30
m.config.background_color = "#0B1020"

WHITE = "#F3F2ED"
MUTED = "#9BAAC3"
TEAL = "#4DE0C1"
YELLOW = "#FFD166"
PINK = "#FF819F"


def headline(value, y=4.85, size=42, color=WHITE):
    result = m.Text(value, font="Space Grotesk", weight="BOLD",
                    font_size=size, color=color)
    result.move_to([0, y, 0])
    return result


def text(value, y, size=38, color=WHITE):
    result = m.Text(value, font="Inter", weight="MEDIUM", font_size=size,
                    color=color)
    result.move_to([0, y, 0])
    return result


def equation(value, y, size=48, color=WHITE):
    result = m.MathTex(value, font_size=size, color=color)
    result.move_to([0, y, 0])
    return result


class HousePriceLearningStep(NarratedScene):
    def construct(self):
        self.load_narration(__file__)
        self.header = headline("One mistake → one update", size=40)
        self.house = text("HOUSE  •  1,000 sq ft", 3.45, 38, MUTED)
        self.predicted = text("Prediction     $100,000", 2.15, 40, YELLOW)
        self.actual = text("Labeled price   $200,000", 1.25, 40, TEAL)
        self.model = equation(r"\hat y=wx", 0.0, 59)
        self.model_note = text("x = 1    •    starting w = 1", -0.85, 34, MUTED)
        self.error = equation(r"\hat y-y=1-2=-1", 0.35, 47, PINK)
        self.loss_eq = equation(r"L=\frac{1}{2}(\hat y-y)^2=0.5", -0.9, 45, WHITE)
        self.gradient_eq = equation(r"\frac{\partial L}{\partial w}=(\hat y-y)x=-1", 0.55, 41, YELLOW)
        self.update_eq = equation(r"w'=w-\eta\frac{\partial L}{\partial w}=1.5", -0.8, 42, WHITE)
        self.eta_note = text("learning rate  η = 0.5", -1.8, 32, MUTED)
        self.new_prediction = text("New prediction     $150,000", 1.6, 40, YELLOW)
        self.new_loss = equation(r"L=\frac{1}{2}(1.5-2)^2=0.125", 0.15, 48, TEAL)
        self.loop = text("PREDICT  →  MEASURE", -2.15, 31, TEAL)
        self.loop2 = text("ADJUST  →  REPEAT", -2.9, 31, TEAL)
        self.new_weight = equation(r"w: 1\longrightarrow1.5", -1.0, 47, YELLOW)

        # Hook: put the model's visible miss next to its training label.
        duration = self.begin_beat(0)
        self.add(self.header, self.house, self.predicted, self.actual)
        self.play(m.Indicate(self.predicted, color=PINK, scale_factor=1.08),
                  m.Indicate(self.actual, color=TEAL, scale_factor=1.08),
                  run_time=0.9)
        self.wait(max(0, duration - 1.0))

        # Define the units and model that produced the displayed prediction.
        duration = self.begin_beat(1)
        next_header = headline("The model makes a guess", size=42)
        self.play(m.Transform(self.header, next_header), m.Write(self.model),
                  m.FadeIn(self.model_note), run_time=0.8)
        self.wait(max(0, duration - 0.8))

        # Turn the mismatch into one explicit objective value.
        duration = self.begin_beat(2)
        next_header = headline("Turn error into loss", size=42)
        self.play(m.Transform(self.header, next_header), m.FadeOut(self.model),
                  m.FadeOut(self.model_note), m.Write(self.error),
                  m.Write(self.loss_eq), run_time=0.9)
        self.wait(max(0, duration - 0.9))

        # The negative slope means a positive change to w lowers loss here.
        duration = self.begin_beat(3)
        next_header = headline("Use the loss slope", size=42)
        direction = m.Arrow([-1.2, -2.9, 0], [1.2, -2.9, 0], buff=0,
                            color=YELLOW, stroke_width=7)
        self.play(m.Transform(self.header, next_header), m.FadeOut(self.error),
                  m.Transform(self.loss_eq, equation(r"L=0.5\quad\text{at }w=1", -0.9, 43)),
                  m.Write(self.gradient_eq),
                  m.Transform(self.update_eq, equation(
                      r"w'=1-0.5(-1)=1.5", -2.0, 48)),
                  m.FadeIn(self.eta_note.move_to([0, -3.55, 0])),
                  m.GrowArrow(direction),
                  run_time=1.1)
        self.wait(max(0, duration - 1.1))

        # Re-evaluate the same house to check the direction of the update.
        duration = self.begin_beat(4)
        next_header = headline("The mistake gets smaller", size=40)
        self.play(m.Transform(self.header, next_header), m.FadeOut(self.gradient_eq),
                  m.FadeOut(self.eta_note), m.FadeOut(direction),
                  m.FadeOut(self.predicted), m.FadeOut(self.actual),
                  m.FadeOut(self.loss_eq), m.FadeOut(self.update_eq),
                  m.Write(self.new_weight),
                  m.Write(self.new_loss), m.FadeIn(self.new_prediction),
                  run_time=1.0)
        self.wait(max(0, duration - 1.0))

        # Summarize the loop and qualify what this example demonstrates.
        duration = self.begin_beat(5)
        next_header = headline("Training repeats the loop", size=40)
        scope = text("One training example\nNew houses may differ", -3.7, 24, MUTED)
        self.play(m.Transform(self.header, next_header), m.FadeOut(self.house),
                  m.FadeOut(self.new_prediction), m.FadeOut(self.new_loss),
                  m.FadeOut(self.new_weight), m.FadeOut(self.loss_eq),
                  m.FadeIn(self.loop), m.FadeIn(self.loop2), m.FadeIn(scope),
                  run_time=0.9)
        self.wait(max(0, duration - 0.9))
