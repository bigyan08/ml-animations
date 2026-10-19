"""From a checked scalar correction to schematic layered language models."""

import manimgx as m

from shared.scenes import NarratedScene
from shared.visuals import node, connection
import numpy as np

from model import X, Y, fit_path, regression_loss, scalar_loss

m.config.pixel_width = 1080
m.config.pixel_height = 1920
m.config.frame_rate = 30
m.config.background_color = '#0B1020'
WHITE, MUTED = '#F3F2ED', '#ACBAD0'
TEAL, GOLD, BLUE, PINK = '#4DE0C1', '#FFD166', '#75AEFF', '#FF819F'


def txt(value, x=0, y=0, size=28, color=WHITE, bold=False):
    return m.Text(value, font='Inter', weight='SEMIBOLD' if bold else 'MEDIUM',
                  font_size=size, color=color).move_to([x, y, 0])


def math(value, y=0, size=46, color=WHITE):
    return m.MathTex(value, font_size=size, color=color).move_to([-.2, y, 0])


def net(counts=(2, 3, 3, 1), top=2.7, bottom=-.2):
    columns = []
    for x, count in zip(np.linspace(-2.4, 2.1, len(counts)), counts):
        ys = np.linspace(top, bottom, count) if count > 1 else [(top + bottom) / 2]
        columns.append(m.VGroup(*[node(x, y, radius=.19, color=BLUE if len(columns) == 0 else TEAL)
                                  for y in ys]))
    edges = []
    for before, after in zip(columns[:-1], columns[1:]):
        layer = m.VGroup()
        for a in before:
            for b in after:
                layer.add(connection(a, b))
        edges.append(layer)
    return m.VGroup(*edges, *columns), columns, edges


class SimpleMathToChatGPT(NarratedScene):
    pulse_color = GOLD

    def construct(self):
        self.load_narration(__file__)
        self.heading = None
        self.stage = None

        self.begin(0, 'From error to AI', first=True)
        pred = txt('7', -1.6, 1.7, 66, GOLD, True)
        target = txt('10', 1.4, 1.7, 66, TEAL, True)
        labels = m.VGroup(txt('Prediction', -1.6, 2.75, 27, MUTED),
                          txt('Target', 1.4, 2.75, 27, MUTED))
        link = m.Arrow([-1, 1.7, 0], [.65, 1.7, 0], buff=0, color=GOLD)
        tiny, _, _ = net(top=-.5, bottom=-1.9)
        self.stage = m.VGroup(pred, target, labels, link, tiny)
        self.add(self.stage)
        self.play(m.Indicate(pred, color=GOLD, scale_factor=1.1), run_time=.7)
        self.finish()

        self.begin(1, 'One prediction')
        axis = m.Line([-2.6, .65, 0], [2.4, .65, 0], color=MUTED, stroke_width=3)
        a = m.Dot([-1.6, .65, 0], color=GOLD, radius=.11)
        b = m.Dot([1.4, .65, 0], color=TEAL, radius=.11)
        gap = m.Arrow([-1.6, -.1, 0], [1.4, -.1, 0], buff=0, color=PINK,
                      stroke_width=5)
        self.stage = m.VGroup(labels, pred, target, axis, a, b)
        self.play(m.FadeIn(self.stage), run_time=.5)
        self.cue('prediction minus actual')
        error = math(r'\text{Error}=7-10=-3', -1.4, 42, PINK)
        self.play(m.GrowArrow(gap), m.Write(error), run_time=.8)
        self.stage.add(gap, error)
        self.finish()

        self.begin(2, 'Measure the mistake')
        loss_eq = math(r'L=(7-10)^2=9', 2.2, 53, GOLD)
        tiles = m.VGroup(*[m.Square(side_length=.55, color=TEAL, fill_color=TEAL,
                                    fill_opacity=.22, stroke_width=2)
                           .move_to([-.8 + .61*i, .35 - .61*j, 0])
                           for j in range(3) for i in range(3)])
        count = txt('3 × 3 = 9', -.2, -1.8, 32, TEAL, True)
        self.stage = m.VGroup(loss_eq, tiles, count)
        self.cue('here we square the error')
        self.play(m.Write(loss_eq), run_time=.8)
        self.play(m.LaggedStart(*[m.FadeIn(t) for t in tiles], lag_ratio=.1), run_time=1)
        self.play(m.FadeIn(count), run_time=.3)
        self.finish()

        self.begin(3, 'Take a downhill step')
        axes = m.Axes(x_range=[5, 11, 2], y_range=[0, 25, 5], x_length=5.2,
                      y_length=3.1, tips=False,
                      axis_config={'color': '#60718B', 'stroke_width': 2})
        axes.move_to([-.2, 1.15, 0])
        bowl = axes.plot(scalar_loss, x_range=[5, 11], color=TEAL, stroke_width=5)
        point = m.Dot(axes.c2p(7, 9), color=GOLD, radius=.12)
        axis_labels = m.VGroup(txt('Loss', -2.2, 3.15, 30, MUTED),
                              txt('Prediction p', -.1, -1.0, 27, MUTED))
        toy = txt('Toy model: p is the parameter', -.2, 3.7, 25, MUTED)
        self.stage = m.VGroup(axes, bowl, point, axis_labels, toy)
        self.play(m.Create(axes), m.Create(bowl), m.FadeIn(point),
                  m.FadeIn(axis_labels), m.FadeIn(toy), run_time=1)
        self.cue('the gradient tells us')
        tangent = axes.plot(lambda p: 9 - 6*(p-7), x_range=[6.1, 8.0],
                            color=GOLD, stroke_width=3)
        step = math(r'p:7\longrightarrow7.6', -2, 44, GOLD)
        rate = txt('Learning rate 0.1', -.2, -2.7, 26, MUTED)
        self.play(m.Create(tangent), m.FadeIn(rate), run_time=.6)
        self.cue('one update moves')
        path = axes.plot(scalar_loss, x_range=[7, 7.6], color=GOLD, stroke_width=6)
        self.play(m.MoveAlongPath(point, path), m.Write(step), run_time=1.2)
        self.play(m.Transform(rate,txt('Loss: 9 → 5.76',-.2,-2.7,28,TEAL)),run_time=.4)
        self.stage.add(tangent, step, rate)
        self.finish()

        self.begin(4, 'Repeat the loop')
        positions = [(-1.7, 2.1), (1.2, 2.1), (1.2, -.25), (-1.7, -.25)]
        words = ['Predict', 'Measure\nloss', 'Update', 'Repeat']
        boxes, texts, arrows = m.VGroup(), m.VGroup(), m.VGroup()
        for (x,y), word in zip(positions, words):
            boxes.add(m.RoundedRectangle(width=2.05, height=1.25, corner_radius=.15,
                                          color=BLUE, stroke_width=2).move_to([x,y,0]))
            texts.add(txt(word, x, y, 28, WHITE, True))
        for i in range(4):
            a, b = boxes[i], boxes[(i+1)%4]
            d = b.get_center() - a.get_center()
            d /= np.linalg.norm(d)
            trim = 1.12 if abs(d[0]) > .5 else .72
            arrows.add(m.Arrow(a.get_center()+trim*d, b.get_center()-trim*d,
                                buff=0, color=TEAL, stroke_width=3))
        scope = txt('Aim: lower training loss', -.2, -2.3, 29, MUTED)
        self.stage = m.VGroup(boxes, texts, arrows, scope)
        self.play(m.FadeIn(self.stage), run_time=.5)
        for t in texts:
            self.play(m.Indicate(t, color=TEAL, scale_factor=1.06), run_time=.45)
        self.finish()

        self.begin(5, 'Fit many examples')
        axes = m.Axes(x_range=[0, 3.2, 1], y_range=[0, 5, 1], x_length=5.2,
                      y_length=3.8, tips=False,
                      axis_config={'color': '#60718B', 'stroke_width': 2})
        axes.move_to([-.2, .9, 0])
        dots = m.VGroup(*[m.Dot(axes.c2p(x,y), color=TEAL, radius=.08) for x,y in zip(X,Y)])
        path = fit_path()
        def fitted(w):
            return axes.plot(lambda x: w[0]*x+w[1], x_range=[0,3.2], color=GOLD, stroke_width=5)
        line = fitted(path[0])
        labels = m.VGroup(txt('Price (scaled)', -1.7, 3.45, 25, MUTED),
                          txt('Size (scaled)', 1.2, -1.55, 25, MUTED),
                          txt('Synthetic houses', -.2, -2.8, 28, MUTED))
        self.stage = m.VGroup(axes,dots,line,labels)
        self.play(m.Create(axes),m.FadeIn(dots),m.Create(line),m.FadeIn(labels),run_time=1)
        counter=txt(f'Average loss: {regression_loss(path[0]):.2f}',-.2,-2.05,28,GOLD)
        self.add(counter)
        self.stage.add(counter)
        self.cue('gradient descent adjusts')
        for i in [1,2,3,5,8,12,20,35,60]:
            self.play(m.Transform(line,fitted(path[i])),run_time=.6)
            self.remove(counter)
            self.stage.remove(counter)
            counter=txt(f'Average loss: {regression_loss(path[i]):.2f}',-.2,-2.05,28,GOLD)
            self.add(counter)
            self.stage.add(counter)
        self.finish()

        self.begin(6, 'Combine simple functions')
        sum_box = m.RoundedRectangle(width=2.4,height=1.3,corner_radius=.12,
                                     color=BLUE).move_to([-1.55,2.6,0])
        bend_box = m.RoundedRectangle(width=2.4,height=1.3,corner_radius=.12,
                                      color=TEAL).move_to([1.25,2.6,0])
        labels = m.VGroup(txt('Weighted\nsum',-1.55,2.6,30,BLUE),
                          txt('Nonlinear\nbend',1.25,2.6,30,TEAL))
        arrow = m.Arrow([-.28,2.6,0],[-.02,2.6,0],buff=0,color=MUTED)
        axes = m.Axes(x_range=[-2,2,1], y_range=[0,2,1], x_length=4.7,
                      y_length=2.4,tips=False,axis_config={'color':'#60718B','stroke_width':2})
        axes.move_to([-.2,-.1,0])
        bend = axes.plot(lambda x: max(0,x),x_range=[-2,2],color=TEAL,stroke_width=5)
        note = txt('Richer relationships',-.2,-2.15,30,TEAL,True)
        self.stage=m.VGroup(sum_box,bend_box,labels,arrow,axes,bend,note)
        self.play(m.FadeIn(sum_box),m.FadeIn(labels[0]),run_time=.5)
        self.play(m.GrowArrow(arrow),m.FadeIn(bend_box),m.FadeIn(labels[1]),run_time=.5)
        self.cue('those bends are crucial')
        self.play(m.Create(axes),m.Create(bend),m.FadeIn(note),run_time=1)
        self.finish()

        self.begin(7, 'Layers build\nrepresentations')
        network, columns, edges = net()
        names=m.VGroup(txt('Input',-2.4,-1.0,25,BLUE),
                       txt('Hidden layers',-.15,-1.0,25,MUTED),
                       txt('Output',2.1,-1.0,25,TEAL),
                       txt('Schematic network',-.2,-2.3,30,MUTED))
        self.stage=m.VGroup(network,names)
        self.play(m.FadeIn(network),m.FadeIn(names),run_time=.8)
        self.cue('passes it forward')
        for e in edges:
            self.pulse(e,.5)
        self.cue('training adjusts')
        self.play(*[line.animate.set_stroke(width=2+(i%3),opacity=.75)
                    for e in edges for i,line in enumerate(e)],run_time=.7)
        self.cue('in image models')
        visual_features=txt('Vision: edges → complex features',-.2,-2.3,25,MUTED)
        self.play(m.Transform(names[-1],visual_features),run_time=.5)
        self.finish()

        self.begin(8, 'More layers, more depth')
        network, columns, edges = net(counts=(2,3,3,3,3,2),top=2.5,bottom=.1)
        apps=m.VGroup(txt('Images',-2, -1.25,28,BLUE),txt('Speech',-.2,-1.25,28,TEAL),
                      txt('Language',1.65,-1.25,28,GOLD))
        note=txt('Depth alone is not enough',-.2,-2.5,27,MUTED)
        self.stage=m.VGroup(network,apps,note)
        self.play(m.FadeIn(network),run_time=.6)
        for e in edges:
            self.pulse(e,.4)
        self.cue('these systems can handle')
        self.play(m.FadeIn(apps),run_time=.5)
        self.cue('depth alone')
        self.play(m.FadeIn(note),run_time=.5)
        self.finish()

        self.begin(9, 'From patterns to tools')
        applications=m.VGroup(txt('Recognize & recommend',-.2,2.2,32,BLUE,True),
                               txt('Translate & generate',-.2,.7,32,TEAL,True),
                               txt('ChatGPT',-.2,-.8,40,GOLD,True))
        self.play(m.FadeIn(applications),run_time=.6)
        self.cue('for language models')
        new_heading=m.Text('Now predict language',font='Space Grotesk',weight='BOLD',
                            font_size=39,color=WHITE)
        new_heading.move_to([-2.9+new_heading.width/2,5.1-new_heading.height/2,0])
        self.play(m.FadeOut(applications),m.Transform(self.heading,new_heading),run_time=.4)
        context=txt('The sky is',-.2,2.75,40,WHITE,True)
        label=txt('Next-token probabilities',-.2,1.8,27,MUTED)
        bars=m.VGroup()
        for word,p,y,color in [('blue',.7,.7,BLUE),('green',.3,-.4,TEAL)]:
            bar=m.Rectangle(width=3.3*p,height=.38,color=color,fill_color=color,
                            fill_opacity=.9,stroke_width=0).move_to([-.85+3.3*p/2,y,0])
            bars.add(txt(word,-2.05,y,29,color),bar,txt(f'{p:.1f}',2.15,y,29,color))
        illustrative=txt('Illustrative two-token model',-.2,-1.45,27,MUTED)
        self.stage=m.VGroup(context,label,bars,illustrative)
        self.play(m.FadeIn(context),m.FadeIn(label),run_time=.6)
        self.play(m.FadeIn(bars),m.FadeIn(illustrative),run_time=.7)
        self.cue('transformer attention uses context')
        attention=txt('Attention uses context',-.2,.35,32,TEAL,True)
        training=txt('Further training\nshapes responses',-.2,-1.35,30,MUTED)
        self.play(m.FadeOut(label),m.FadeOut(bars),m.FadeOut(illustrative),
                  m.FadeIn(attention),run_time=.6)
        connector=m.Arrow([-.2,2.05,0],[-.2,1.0,0],buff=0,color=TEAL,stroke_width=4)
        self.play(m.GrowArrow(connector),run_time=.5)
        self.cue('further training helps')
        self.play(m.FadeIn(training),run_time=.5)
        self.stage.add(attention,training,connector)
        self.finish()

        self.begin(10, 'One shared principle')
        self.heading.move_to([-2.9+self.heading.width/2,3.4,0])
        loop=txt('Predict → measure → update',-.2,1.8,31,TEAL,True)
        foundation=txt('Data + architecture\n+ training',-.2,.2,34,WHITE,True)
        limit=txt('Useful patterns. Mistakes remain.',-.2,-1.55,26,MUTED)
        self.stage=m.VGroup(loop,foundation,limit)
        self.play(m.FadeIn(loop),m.FadeIn(foundation),run_time=.7)
        self.play(m.FadeIn(limit),run_time=.4)
        self.finish()

    def begin(self,index,title,first=False):
        self.begin_beat(index)
        heading=m.Text(title,font='Space Grotesk',weight='BOLD',font_size=39,
                       color=WHITE)
        heading.move_to([-2.9+heading.width/2,5.1-heading.height/2,0])
        anims=[]
        if self.heading is not None: anims.append(m.FadeOut(self.heading))
        if self.stage is not None: anims.append(m.FadeOut(self.stage))
        if first: self.add(heading)
        else:
            anims.append(m.FadeIn(heading))
            self.play(*anims,run_time=.35)
            self.clear()
            self.add(heading)
        self.heading=heading
        self.stage=None
