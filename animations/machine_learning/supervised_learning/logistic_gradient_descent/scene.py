"""A vertical, narrated explanation using the actual logistic-loss contours."""


import contourpy
import manimgx as m

from shared.scenes import NarratedScene
import numpy as np

from model import gradient, loss, trajectory

m.config.pixel_width = 1080
m.config.pixel_height = 1920
m.config.frame_rate = 30
m.config.background_color = "#0B1020"

WHITE = "#EDF2FA"
MUTED = "#9BAAC3"
BLUE = "#64B5F6"
TEAL = "#4DE0C1"
YELLOW = "#FFD166"
RED = "#FF7D8E"


def text(value, y, size=30, color=WHITE):
    return m.Text(value, font="DejaVu Sans", font_size=size, color=color).move_to([0, y, 0])


def math(value, y, size=34, color=WHITE):
    return m.MathTex(value, font_size=size, color=color).move_to([0, y, 0])


class LogisticGradientDescent(NarratedScene):
    def construct(self):
        self.load_narration(__file__)
        self.add(text("MACHINE LEARNING  /  THE WHY", 5.85, 26, TEAL))
        title = text("Learning, one\ndownhill step at a time", 4.87, 38)
        self.add(title)
        footer = text("LOGISTIC REGRESSION", -5.65, 28, MUTED)
        self.add(footer)

        # 1. A visual question: a path through weight space.
        duration = self.begin_beat(0)
        dots = m.VGroup(*[
            m.Dot([x, 1.8 - 0.35 * i, 0], radius=0.10, color=TEAL)
            for i, x in enumerate([-2.4, -1.4, -0.55, 0.15, 0.7, 1.1])
        ])
        route = m.VMobject(color=TEAL, stroke_width=5).set_points_as_corners(
            [d.get_center() for d in dots])
        question = text("Why does this direction work?", -1.15, 28)
        description = text("2 weights  •  1 loss to minimize", -2.1, 25, MUTED)
        self.play(m.Create(route), m.FadeIn(dots), run_time=1.7)
        self.play(m.FadeIn(question), m.FadeIn(description), run_time=0.7)
        self.wait(duration - 3.0)
        self.play(m.FadeOut(route), m.FadeOut(dots), m.FadeOut(question),
                  m.FadeOut(description), run_time=0.6)

        # 2. Define exactly the two free parameters used below.
        duration = self.begin_beat(1)
        model_formula = math(r"p=\sigma(w_1x_1+w_2x_2)", 2.75, 42)
        sigmoid_formula = math(r"\sigma(z)=\frac{1}{1+e^{-z}}", 1.1, 39, TEAL)
        feature_label = text("features → score → probability", -0.25, 26, MUTED)
        bias_label = text("Two trainable weights. Bias fixed at 0.", -1.3, 23, MUTED)
        prob_label = text("Predict class 1 when p ≥ 0.5", -2.35, 27)
        model_group = m.VGroup(model_formula, sigmoid_formula, feature_label,
                               bias_label, prob_label)
        self.play(m.Write(model_formula), run_time=1.2)
        self.play(m.FadeIn(sigmoid_formula), m.FadeIn(feature_label),
                  m.FadeIn(bias_label), m.FadeIn(prob_label), run_time=1.0)
        self.wait(duration - 2.8)
        self.play(m.FadeOut(model_group), run_time=0.6)

        # 3. Numerically computed contours, not decorative ellipses.
        duration = self.begin_beat(2)
        self.axes = m.Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                           x_length=5.6, y_length=5.6, tips=False,
                           axis_config={"color": "#52617C", "stroke_width": 2,
                                        "include_ticks": True})
        self.axes.move_to([-0.15, 0.35, 0])
        x_label = math(r"w_1", -2.87, 40, MUTED).shift(2.7 * m.RIGHT)
        y_label = math(r"w_2", 3.36, 40, MUTED).shift(0.15 * m.LEFT)
        self.plot_labels = m.VGroup(x_label, y_label)
        levels = [0.62, 0.66, 0.74, 0.86, 1.02, 1.22, 1.46, 1.74]
        grid = np.linspace(-3, 3, 141)
        xx, yy = np.meshgrid(grid, grid)
        zz = loss(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        generator = contourpy.contour_generator(x=grid, y=grid, z=zz)
        contours = m.VGroup()
        for index, level in enumerate(levels):
            for points in generator.lines(level):
                curve = m.VMobject(color=TEAL if index < 2 else BLUE,
                                   stroke_width=2.3, stroke_opacity=0.85 - index * 0.06)
                curve.set_points_as_corners([self.axes.c2p(*p) for p in points])
                contours.add(curve)
        legend = text("Contours = equal cross-entropy loss", 3.95, 24, MUTED)
        self.path = trajectory()
        current = m.Dot(self.axes.c2p(*self.path[0]), radius=0.09, color=YELLOW)
        optimum = m.Dot(self.axes.c2p(*self.path[-1]), radius=0.06, color=TEAL)
        self.counter = self.status(0)
        self.play(m.Create(self.axes), m.FadeIn(contours), m.FadeIn(self.plot_labels),
                  m.FadeIn(legend), run_time=1.4)
        self.play(m.FadeIn(current), m.FadeIn(optimum), m.FadeIn(self.counter), run_time=0.6)
        self.wait(duration - 2.0)

        # 4. The actual negative gradient at the starting point.
        duration = self.begin_beat(3)
        w = self.path[0]
        uphill = gradient(w)
        origin = self.axes.c2p(*w)
        arrow = m.Arrow(origin, self.axes.c2p(*(w - 2.8 * uphill)),
                        buff=0, color=YELLOW, stroke_width=5)
        direction = text("−gradient = downhill", 3.95, 26, YELLOW)
        update = math(r"\mathbf{w}_{t+1}=\mathbf{w}_t-\eta\nabla L(\mathbf{w}_t)", -3.75, 40)
        self.play(m.Transform(legend, direction), m.GrowArrow(arrow),
                  m.Write(update), run_time=1.2)
        self.wait(duration - 2.4)
        self.play(m.FadeOut(arrow), run_time=0.3)
        segment = m.Line(origin, self.axes.c2p(*self.path[1]), color=YELLOW, stroke_width=4)
        self.play(m.Create(segment), current.animate.move_to(self.axes.c2p(*self.path[1])),
                  run_time=0.9)
        self.set_counter(1)

        # 5. Explain the analytic gradient while holding the landscape fixed.
        duration = self.begin_beat(4)
        explanation = text("Prediction error × feature", 3.95, 27, TEAL)
        gradient_formula = math(r"\nabla L=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)\mathbf{x}_i", -3.75, 40)
        self.play(m.Transform(legend, explanation), m.Transform(update, gradient_formula), run_time=1)
        self.wait(duration - 1)

        # 6. Trace the computed iterates, preserving every intermediate point.
        duration = self.begin_beat(5)
        self.play(m.Transform(legend, text("Small steps. Better predictions.", 3.95, 26, TEAL)),
                  m.Transform(update, math(r"\eta=1.5\qquad L:\ 1.342\ \longrightarrow\ 0.609", -3.75, 30)),
                  run_time=0.7)
        indices = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 32, 40, 52, 65, 80]
        for start, stop in zip(indices[:-1], indices[1:]):
            trail = m.VMobject(color=YELLOW, stroke_width=4).set_points_as_corners(
                [self.axes.c2p(*w) for w in self.path[start:stop + 1]])
            self.play(m.Create(trail),
                      m.MoveAlongPath(current, trail),
                      run_time=(duration - 1.5) / (len(indices) - 1), rate_func=m.linear)
            self.set_counter(stop)
        self.wait(0.8)

        # 7. First-order Taylor reasoning: the central "why".
        duration = self.begin_beat(6)
        reason = math(r"\Delta L\approx-\eta\|\nabla L\|^2\leq 0", -3.75, 35, TEAL)
        self.play(m.Transform(legend, text("Why downhill works locally", 3.95, 28)),
                  m.Transform(update, reason), run_time=1.0)
        self.wait(duration - 1.0)

        # 8. Qualified conclusion: convexity alone does not ensure a finite minimizer.
        duration = self.begin_beat(7)
        self.play(m.Transform(legend, text("Convex loss. A global minimum.", 3.95, 27, TEAL)),
                  m.Transform(update, text("Overlapping data • suitable step size", -3.75, 24, MUTED)),
                  m.Indicate(optimum, color=TEAL, scale_factor=1.8), run_time=1.2)
        self.wait(duration - 1.2)

    def status(self, step):
        return text(f"STEP {step:02d}     LOSS {loss(self.path[step]):.3f}", -2.88, 30, YELLOW)

    def set_counter(self, step):
        # Numerals change discretely; morphing glyphs makes fast updates illegible.
        self.remove(self.counter)
        self.counter = self.status(step)
        self.add(self.counter)
