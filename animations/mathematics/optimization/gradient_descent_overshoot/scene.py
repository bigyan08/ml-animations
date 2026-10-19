"""A single quadratic bowl shows why an oversized learning rate can fail."""


import manimgx as m

from shared.scenes import NarratedScene

from model import loss


m.config.pixel_width = 1080
m.config.pixel_height = 1920
m.config.frame_rate = 30
m.config.background_color = "#0B1020"

WHITE = "#F3F2ED"
MUTED = "#9BAAC3"
TEAL = "#4DE0C1"
YELLOW = "#FFD166"
PINK = "#FF819F"


def label(value, y, size=32, color=WHITE):
    result = m.Text(value, font="Inter", weight="MEDIUM", font_size=size,
                    color=color)
    result.move_to([0, y, 0])
    return result


def equation(value, y, size=40, color=WHITE):
    result = m.MathTex(value, font_size=size, color=color)
    result.move_to([0, y, 0])
    return result


class GradientDescentOvershoot(NarratedScene):
    def construct(self):
        self.load_narration(__file__)
        self.axes = m.Axes(
            x_range=[-2, 2, 1], y_range=[0, 2.2, 0.5],
            x_length=6.6, y_length=4.0, tips=False,
            axis_config={"color": "#52617C", "stroke_width": 2,
                         "include_ticks": True},
        ).move_to([0, 0.25, 0])
        self.bowl = self.axes.plot(lambda w: loss(w), x_range=[-2, 2],
                                   color=TEAL, stroke_width=6)
        self.start = m.Dot(self.axes.c2p(1, loss(1)), radius=0.11, color=YELLOW)
        self.large_end = m.Dot(self.axes.c2p(-1.4, loss(-1.4)), radius=0.12,
                               color=PINK)
        self.hook_step = m.Arrow(self.axes.c2p(1, 0), self.axes.c2p(-1.4, 0),
                                 buff=0, color=PINK, stroke_width=6)
        self.headline = label("Downhill. Higher loss.", 5.9, 48, WHITE)
        self.axes_labels = m.VGroup(
            label("w", -2.15, 48, MUTED).shift(2.95 * m.RIGHT),
            label("Loss", 2.75, 45, MUTED).shift(2.9 * m.LEFT),
        )
        self.start_note = label("start", 1.1, 42, YELLOW).shift(1.35 * m.RIGHT)
        self.end_note = label("land", 1.72, 42, PINK).shift(1.8 * m.LEFT)
        self.param_readout = equation(r"w: 1\ \longrightarrow\ -1.4", -3.25, 39)
        self.values = equation(r"L(1)=0.50\qquad L(-1.4)=0.98", -3.25, 36)

        # Hook: the same bowl contains a start point and a worse destination.
        duration = self.begin_beat(0)
        self.add(self.axes, self.bowl, self.axes_labels, self.start,
                 self.large_end, self.headline)
        self.play(m.GrowArrow(self.hook_step),
                  m.Indicate(self.large_end, color=PINK, scale_factor=1.7),
                  run_time=0.9)
        self.play(m.FadeIn(self.start_note), m.FadeIn(self.end_note),
                  m.Write(self.param_readout), run_time=0.9)
        self.wait(max(0, duration - 1.8))

        # The negative gradient gives a direction on the parameter axis.
        duration = self.begin_beat(1)
        next_headline = label("Direction is not distance", 5.9, 45, WHITE)
        grad_right = m.Arrow(self.axes.c2p(1, 0), self.axes.c2p(1.75, 0),
                             buff=0, color=PINK, stroke_width=5)
        grad_left = m.Arrow(self.axes.c2p(1, 0), self.axes.c2p(0.25, 0),
                            buff=0, color=YELLOW, stroke_width=5)
        derivative = equation(r"L(w)=\frac{1}{2}w^2\qquad L'(1)=1", -3.25,
                              36, TEAL)
        self.play(m.Transform(self.headline, next_headline),
                  m.FadeOut(self.hook_step),
                  m.GrowArrow(grad_right), m.GrowArrow(grad_left),
                  m.FadeOut(self.param_readout), m.Write(derivative),
                  run_time=1.2)
        self.wait(max(0, duration - 1.2))

        # Gradient descent also chooses a distance through its learning rate.
        duration = self.begin_beat(2)
        next_headline = label("The step is too long", 5.9, 48, WHITE)
        step_calculation = equation(
            r"\eta=2.4\qquad w'=1-2.4(1)=-1.4", -3.25, 36, PINK)
        overshoot = m.Arrow(self.axes.c2p(1, 0), self.axes.c2p(-1.4, 0),
                            buff=0, color=PINK, stroke_width=6)
        self.play(m.Transform(self.headline, next_headline),
                  m.FadeOut(grad_right), m.FadeOut(grad_left),
                  m.FadeOut(derivative), m.Write(step_calculation),
                  m.GrowArrow(overshoot),
                  run_time=1.2)
        self.wait(max(0, duration - 1.2))

        # Compare actual objective values, rather than calling every crossing bad.
        duration = self.begin_beat(3)
        next_headline = label("Check the destination", 5.9, 48, WHITE)
        self.play(m.Transform(self.headline, next_headline),
                  m.FadeOut(step_calculation),
                  m.Indicate(self.start, color=YELLOW, scale_factor=1.5),
                  m.Indicate(self.large_end, color=PINK, scale_factor=1.5),
                  m.FadeIn(self.values), run_time=1.1)
        self.wait(max(0, duration - 1.1))

        # A tangent shows what the derivative knows: the local slope at w=1.
        duration = self.begin_beat(4)
        next_headline = label("Local slope. Long jump.", 5.9, 46, WHITE)
        tangent = self.axes.plot(lambda w: 0.5 + (w - 1), x_range=[0.2, 1.8],
                                 color=YELLOW, stroke_width=3)
        tangent_note = label("slope at the start", 3.35, 27, YELLOW)
        self.play(m.Transform(self.headline, next_headline),
                  m.FadeOut(overshoot),
                  m.Create(tangent), m.FadeIn(tangent_note), run_time=1.0)
        self.wait(max(0, duration - 1.0))

        # Same bowl and start, with a shorter step that still crosses the minimum.
        duration = self.begin_beat(5)
        next_headline = label("Crossing can still help", 5.9, 44, WHITE)
        moderate_end = m.Dot(self.axes.c2p(-0.5, loss(-0.5)), radius=0.12,
                             color=TEAL)
        moderate_arrow = m.Arrow(self.axes.c2p(1, 0), self.axes.c2p(-0.5, 0),
                                 buff=0, color=TEAL, stroke_width=5)
        moderate_summary = equation(
            r"\eta=1.5\;w'=-0.5\;L:0.50\to0.125",
            -3.25, 36, TEAL)
        self.play(m.Transform(self.headline, next_headline),
                  m.FadeOut(tangent), m.FadeOut(tangent_note),
                  m.FadeOut(self.large_end), m.FadeOut(self.end_note),
                  m.FadeOut(self.values),
                  m.GrowArrow(moderate_arrow), m.FadeIn(moderate_end),
                  m.Write(moderate_summary), run_time=1.2)
        self.wait(max(0, duration - 1.2))

        # One focused rule and its boundary.
        duration = self.begin_beat(6)
        next_headline = label("Match the step to the curve", 5.9, 44, WHITE)
        scope = label("Too large for this bowl’s curvature", -5.55, 36, MUTED)
        self.play(m.Transform(self.headline, next_headline),
                  m.FadeOut(moderate_arrow), m.FadeOut(moderate_summary),
                  m.FadeIn(scope),
                  run_time=1.0)
        self.play(m.FadeIn(self.large_end), m.FadeIn(self.end_note),
                  m.Indicate(self.large_end, color=PINK), run_time=0.5)
        self.wait(max(0, duration - 1.5))
