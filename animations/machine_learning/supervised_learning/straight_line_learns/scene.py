"""One persistent graph connects parameters, residuals, loss and learning."""
import numpy as np
import manimgx as m
from shared.scenes import NarratedScene
from model import X, Y, loss, train

m.config.pixel_width = 1080
m.config.pixel_height = 1920
m.config.frame_rate = 30
m.config.background_color = '#0B1020'
WHITE, MUTED, TEAL, GOLD, PINK = '#F3F2ED', '#AFBED3', '#4DE0C1', '#FFD166', '#FF819F'


def text(s, y, size=29, color=WHITE, font='Inter'):
    obj = m.Text(s, font=font, font_size=size, color=color,
                 weight='BOLD' if font == 'Space Grotesk' else 'MEDIUM')
    obj.move_to([-.2, y, 0])
    if obj.width > 5.8:
        raise ValueError(f'Reflow oversized text: {s}')
    return obj


def eq(s, y=-1.3, size=39, color=WHITE):
    obj = m.MathTex(s, font_size=size, color=color).move_to([-.2, y, 0])
    if obj.width > 5.9:
        raise ValueError(f'Reflow oversized equation: {s}')
    return obj


def point(x, y):
    return np.array([-2.5 + x * 1.1, .35 + y * .034, 0.])


def fitted_line(w, b):
    lo, hi = 0., 4.5
    if w:
        ends = sorted([(0-b)/w, (100-b)/w])
        lo, hi = max(lo, ends[0]), min(hi, ends[1])
    return m.Line(point(lo, w*lo+b), point(hi, w*hi+b), color=GOLD, stroke_width=5)


class StraightLineLearns(NarratedScene):
    def panel(self, *objects):
        animations = [m.FadeOut(o) for o in self.current_panel]
        animations += [m.FadeIn(o) for o in objects]
        if animations:
            self.play(*animations, run_time=.4)
        self.current_panel = list(objects)

    def title(self, s):
        self.play(m.Transform(self.heading, text(s, 5.12, 38, font='Space Grotesk')), run_time=.4)

    def line_to(self, w, b, duration=1.5):
        self.play(m.Transform(self.line, fitted_line(w, b)), run_time=duration)

    def construct(self):
        self.load_narration(__file__)
        self.current_panel = []
        self.heading = text('Can a line learn?', 5.12, 42, font='Space Grotesk')
        grid = m.VGroup()
        for y in range(20,101,20):
            grid.add(m.Line(point(0,y),point(4.5,y),stroke_width=1,color='#27334A'))
            label=m.Text(str(y),font='Inter',font_size=28,color=MUTED)
            label.next_to(m.Dot(point(0,y),radius=.001),m.LEFT,buff=.16)
            grid.add(label)
        for x in range(1,5):
            grid.add(m.Line(point(x,0),point(x,100),stroke_width=1,color='#27334A'))
            label=m.Text(str(x),font='Inter',font_size=28,color=MUTED)
            label.move_to(point(x,0)+[0,-.25,0]); grid.add(label)
        axes=m.VGroup(m.Line(point(0,0),point(4.5,0),color=MUTED,stroke_width=2),
                      m.Line(point(0,0),point(0,100),color=MUTED,stroke_width=2))
        labels=m.VGroup(text('Exam score',4.12,28,MUTED),text('Study hours',-.25,26,MUTED))
        dots=m.VGroup(*[m.Dot(point(x,y),radius=.075,color=TEAL) for x,y in zip(X,Y)])
        self.line=fitted_line(3,20)
        self.begin_beat('hook')
        self.add(self.heading,grid,axes,labels,self.line)
        self.play(m.LaggedStart(*[m.FadeIn(d) for d in dots],lag_ratio=.18),run_time=1.5)
        self.cue("Instead of guessing")
        self.line_to(10,40,3)
        self.panel(text('Two numbers. One pattern.',-1.3,31,GOLD))
        self.finish()

        self.begin_beat('data'); self.title('Four observations')
        self.remove(*dots); self.line_to(0,0,.5)
        self.play(m.FadeOut(self.line),run_time=.3)
        self.panel(text('Simplified example',-1.0,27,MUTED))
        for cue,d in zip(['One studied','Two hours','three hours','four hours'],dots):
            self.cue(cue); self.play(m.FadeIn(d),run_time=.4)
        self.cue('Place these examples')
        self.panel(eq(r'x=[1,2,3,4]',-1.1,37),eq(r'y=[50,60,70,80]',-2.0,37))
        self.finish()

        self.begin_beat('equation'); self.title('Two parameters')
        self.panel(eq(r'\hat y=wx+b',-1.,50,GOLD))
        self.line=fitted_line(10,40); self.play(m.FadeIn(self.line),run_time=.5)
        self.cue('The input')
        self.panel(eq(r'\hat y=wx+b',-1.,50,GOLD),text('x: hours • ŷ: predicted score',-1.85,26))
        self.cue('We call w')
        self.panel(eq(r'\hat y=wx+b',-1.,50,GOLD),text('w: slope • b: intercept',-1.85,28))
        self.cue('For example')
        self.panel(eq(r'\hat y=10(3)+40=70',-1.2,41,GOLD))
        trace=m.VGroup(m.DashedLine(point(3,0),point(3,70),color=WHITE),
                       m.DashedLine(point(3,70),point(0,70),color=WHITE))
        self.play(m.Create(trace),m.Indicate(dots[2],color=GOLD),run_time=1.2)
        self.finish(); self.remove(trace)

        self.begin_beat('weight'); self.title('Change the tilt')
        self.panel(text('b = 40 fixed',-1.,29,MUTED),eq(r'w=0\quad\hat y=40',-1.9,39,GOLD))
        self.cue('When w is zero'); self.line_to(0,40,2)
        for cue,w,formula in [('Increase w to five',5,r'w=5\quad\hat y=5x+40'),('Make it ten',10,r'w=10\quad\hat y=10x+40')]:
            self.cue(cue); self.panel(text('b = 40 fixed',-1.,29,MUTED),eq(formula,-1.9,37,GOLD)); self.line_to(w,40,2)
        self.cue('A negative weight'); self.panel(text('b = 40 fixed',-1.,29,MUTED),eq(r'w=-5\quad\hat y=-5x+40',-1.9,36,GOLD)); self.line_to(-5,40,2)
        self.finish()

        self.begin_beat('bias'); self.title('Change the height')
        self.panel(text('w = 10 fixed',-1.,29,MUTED),eq(r'b=0\quad\hat y=10x',-1.9,39,GOLD))
        self.line_to(10,0,2)
        for cue,b in [('With b equal to twenty',20),('At forty',40)]:
            self.cue(cue); self.panel(text('w = 10 fixed',-1.,29,MUTED),eq(rf'b={b}\quad\hat y=10x+{b}',-1.9,36,GOLD)); self.line_to(10,b,2)
        self.cue('So remember'); self.panel(text('Weight → tilt',-1.1,32,GOLD),text('Bias → height',-2.,32,GOLD))
        self.finish()

        self.begin_beat('loss'); self.title('Measure the mistakes')
        self.panel(text('Error = prediction − actual',-1.2,27,PINK))
        self.cue('Suppose w'); self.line_to(5,40,1.5)
        residuals=m.VGroup(*[m.DashedLine(point(x,5*x+40),point(x,y),color=PINK,stroke_width=4) for x,y in zip(X,Y)])
        self.play(m.Create(residuals),run_time=1)
        self.cue('After one hour'); self.play(m.Indicate(residuals[0],color=PINK),run_time=.7)
        self.cue('After four hours'); self.play(m.Indicate(residuals[3],color=PINK),run_time=.7)
        self.cue('Square each'); self.panel(eq(r'J=\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i)^2',-1.3,40),text('m = 4 examples',-2.25,26,MUTED))
        self.cue('here'); self.panel(eq(r'J=\frac{25+100+225+400}{4}',-1.,38),eq(r'J=187.5',-2.05,46,PINK))
        self.cue('Squaring'); self.panel(text('Opposite errors cannot cancel',-1.,26),text('Large errors cost more',-1.85,28,PINK))
        self.cue('At w ten'); self.play(m.FadeOut(residuals),m.Transform(self.line,fitted_line(10,40)),run_time=1.4)
        self.panel(eq(r'J(10,40)=0',-1.3,46,TEAL)); self.finish()

        self.begin_beat('gradients'); self.title('Find a direction')
        self.panel(eq(r'w=0,\quad b=0',-1.0,42),eq(r'J=4350',-2.,42,PINK)); self.line_to(0,0,1.5)
        self.cue('Then we use derivatives')
        self.panel(eq(r'\frac{\partial J}{\partial w}=\frac{2}{m}\sum_i(\hat y_i-y_i)x_i',-1.15,37),text('Loss sensitivity to the weight',-2.3,26,MUTED))
        self.cue('These are called gradients')
        self.panel(eq(r'\frac{\partial J}{\partial b}=\frac{2}{m}\sum_i(\hat y_i-y_i)',-1.15,37),text('Loss sensitivity to the bias',-2.3,26,MUTED))
        self.cue('A positive gradient'); self.panel(text('Positive gradient → decrease',-1.1,26,GOLD),text('Negative gradient → increase',-2.,26,GOLD))
        self.finish()

        self.begin_beat('training'); self.title('Adjust and repeat')
        self.panel(eq(r'w\leftarrow w-\alpha\frac{\partial J}{\partial w}',-1.,43),eq(r'b\leftarrow b-\alpha\frac{\partial J}{\partial b}',-2.15,43))
        self.cue('Here'); self.panel(eq(r'\alpha=0.05',-1.,45,GOLD),text('Same errors → both updates',-2.,27,MUTED))
        self.cue('Repeat this many times')
        history=train()
        def metrics(i):
            w,b,j=history[i]
            return m.VGroup(text(f'Step {i} / 500',- .9,27,MUTED),
                            text(f'w = {w:.2f}    b = {b:.2f}',-1.65,29,GOLD),
                            text(f'MSE = {j:.4f}',-2.45,30,TEAL))
        # Actual computed iterates; slow early steps expose the initial overshoot.
        stat=metrics(0); self.panel(stat)
        indices=[1,2,3,5,7,10,15,20,30,50,75,100,150,200,250,300,400,500]
        remaining=self.beat_duration-(float(self.time)-self.beat_start)-1
        for i in indices:
            self.play(m.Transform(self.line,fitted_line(*history[i,:2])),m.Transform(stat,metrics(i)),run_time=remaining/len(indices),rate_func=m.linear)
        self.finish()

        self.begin_beat('python'); self.title('The learning loop')
        self.panel(text('for epoch in range(500):',-1.2,25,GOLD,font='DejaVu Sans Mono'))
        self.cue('Compute predictions'); self.panel(text('y_pred = w*x + b',-1.2,25,GOLD,font='DejaVu Sans Mono'))
        self.cue('Subtract actual'); self.panel(text('error = y_pred - y',-1.2,25,PINK,font='DejaVu Sans Mono'))
        self.cue('Calculate one gradient'); self.panel(text('dw = 2*np.mean(error*x)',-1.,23,font='DejaVu Sans Mono'),text('db = 2*np.mean(error)',-1.9,23,font='DejaVu Sans Mono'))
        self.cue('and update both'); self.panel(text('w -= lr*dw',-1.,27,GOLD,font='DejaVu Sans Mono'),text('b -= lr*db',-1.9,27,GOLD,font='DejaVu Sans Mono'))
        self.cue('Run the loop'); self.panel(eq(r'w\approx10,\quad b\approx40',-1.4,43,TEAL))
        self.cue('two and a half hours'); self.panel(eq(r'10(2.5)+40\approx65',-1.15,41,GOLD),text('Prediction, not a guarantee',-2.2,27,MUTED))
        prediction=m.Dot(point(2.5,65),radius=.1,color=GOLD)
        trace=m.VGroup(m.DashedLine(point(2.5,0),point(2.5,65),color=GOLD),m.DashedLine(point(2.5,65),point(0,65),color=GOLD))
        self.play(m.Create(trace),m.FadeIn(prediction),run_time=1)
        self.finish(); self.remove(trace,prediction)

        self.begin_beat('closing'); self.title('What did it learn?')
        self.panel(eq(r'\hat y=10x+40',-1.,48,GOLD),text('Fit slope and intercept',-2.05,28))
        self.cue('Real world dots'); self.panel(text('Real data has more variation',-1.,27),text('Check predictions on new data',-2.,27,MUTED))
        self.cue('But the core process'); self.panel(text('Predict → measure → update',-1.,28,TEAL),text('Reduce the mistakes. Repeat.',-2.,27))
        self.finish()
