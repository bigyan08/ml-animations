"""One calculation expands into a layer, network, and medical-imaging workflow."""

import manimgx as m

from shared.scenes import NarratedScene
from shared.visuals import node, connection
import numpy as np

from model import layer_outputs

m.config.pixel_width = 1080
m.config.pixel_height = 1920
m.config.frame_rate = 30
m.config.background_color = '#0B1020'
WHITE, MUTED = '#F3F2ED', '#9BAAC3'
TEAL, BLUE, GOLD, PINK = '#4DE0C1', '#75AEFF', '#FFD166', '#FF819F'


def label(value, x=0, y=0, size=28, color=WHITE):
    return m.Text(value, font='Inter', weight='MEDIUM', font_size=size,
                  color=color).move_to([x, y, 0])


def equation(value, y, size=40, color=WHITE):
    return m.MathTex(value, font_size=size, color=color).move_to([-.15, y, 0])


def network(rows=(2, 4, 4, 2), top=2.7, spacing=1.55):
    layers = []
    for i, count in enumerate(rows):
        xs = np.linspace(-2.15, 1.85, count)
        layers.append(m.VGroup(*[node(x, top - i * spacing, color=BLUE if i == 0 else TEAL)
                                for x in xs]))
    links = []
    for before, after in zip(layers[:-1], layers[1:]):
        links.append(m.VGroup(*[connection(a, b) for a in before for b in after]))
    return m.VGroup(*links, *layers), layers, links


class NeuronToNetwork(NarratedScene):
    pulse_radius = .055

    def construct(self):
        self.load_narration(__file__)
        self.heading = None

        # Meaningful content is present in frame one; pulses begin immediately.
        self.begin(0, 'One neuron.\nThen what?', animate=False)
        inputs = m.VGroup(node(-1.65, 2.7, .38, BLUE), node(1.35, 2.7, .38, BLUE))
        input_values = m.VGroup(label('0.8', -1.65, 2.7, 29), label('0.4', 1.35, 2.7, 29))
        core = node(-.15, .75, .65)
        core_text = equation(r'\Sigma', .75, 55)
        out = node(-.15, -1.4, .4, GOLD)
        out_text = label('0.6', -.15, -1.4, 30, GOLD)
        inbound = m.VGroup(*[connection(n, core, width=4, opacity=.7) for n in inputs])
        outbound = connection(core, out, GOLD, 4, .8)
        neuron = m.VGroup(inbound, outbound, inputs, core, out, input_values, core_text, out_text)
        self.add(neuron)
        self.pulse(inbound, .8)
        self.play(m.Indicate(core, color=GOLD, scale_factor=1.12), run_time=.4)
        self.pulse(m.VGroup(outbound), .7, GOLD)
        identity = label('An artificial neuron', -.15, -2.5, 30, MUTED)
        self.play(m.FadeIn(identity), run_time=.3)
        self.wait(.8)
        self.play(m.FadeOut(self.heading), run_time=.25)
        self.heading = None
        self.finish()

        self.begin(1, 'Multiply. Add.\nActivate.')
        weights = m.VGroup(label('× 1.5', -1.85, 1.55, 32, BLUE),
                           label('× −1', 1.55, 1.55, 32, BLUE))
        bias = label('bias −0.2', 1.65, .3, 29, PINK)
        sum_formula = equation(r'z=0.8(1.5)+0.4(-1)-0.2', -2.55, 32)
        result = equation(r'a=\max(0,z)=0.6', -3.3, 40, GOLD)
        self.play(m.FadeOut(identity), m.FadeIn(weights), m.FadeIn(bias), run_time=.5)
        self.pulse(inbound, .7)
        self.play(m.Write(sum_formula), run_time=.9)
        relu_text = label('ReLU', -.15, .75, 28, TEAL)
        self.play(m.FadeOut(core_text), m.FadeIn(relu_text), m.Write(result), run_time=.7)
        self.pulse(m.VGroup(outbound), .7, GOLD)
        self.finish()

        self.begin(2, 'The bend\nmakes a difference')
        current = m.VGroup(neuron, relu_text, weights, bias, sum_formula, result)
        self.play(m.FadeOut(current), run_time=.4)
        axes = m.Axes(x_range=[-2, 2, 1], y_range=[0, 2, 1], x_length=5.0,
                      y_length=2.5, tips=False, axis_config={'color': MUTED, 'stroke_width': 2})
        axes.move_to([-.15, 1.1, 0])
        curve = axes.plot(lambda z: max(0, z), x_range=[-2, 2], color=TEAL, stroke_width=6)
        activation = equation(r'\operatorname{ReLU}(z)=\max(0,z)', -1.0, 42, TEAL)
        axis_labels = m.VGroup(label('input z', 2.0, -.45, 26, MUTED),
                               label('output a', -1.9, 2.95, 26, MUTED))
        dot = m.Dot(axes.c2p(.6, .6), radius=.10, color=GOLD)
        self.play(m.Create(axes), m.Create(curve), m.FadeIn(axis_labels), m.Write(activation), run_time=1.2)
        self.play(m.FadeIn(dot), run_time=.25)
        collapse = equation(r'W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2', -2.35, 32)
        collapsed = equation(r"=W'\mathbf{x}+\mathbf{b}'", -3.05, 40)
        note = label('Without the activation:', -.15, -1.75, 29, MUTED)
        self.play(m.FadeIn(note), m.Write(collapse), m.Write(collapsed), run_time=1.0)
        self.finish()
        graph = m.VGroup(axes, curve, axis_labels, activation, dot, collapse, collapsed, note)

        self.begin(3, 'Same inputs.\nDifferent responses.')
        self.play(m.FadeOut(graph), run_time=.35)
        layer_nodes = m.VGroup(*[node(x, .4, .42) for x in [-2.25, -.85, .55, 1.95]])
        layer_inputs = m.VGroup(node(-1.45, 2.6, .43, BLUE), node(1.15, 2.6, .43, BLUE))
        links = m.VGroup(*[connection(a, b, width=2.5) for a in layer_inputs for b in layer_nodes])
        nums = m.VGroup(label('0.8', -1.45, 2.6, 30), label('0.4', 1.15, 2.6, 30))
        # A single unit is followed by its peers, making the composition explicit.
        self.add(layer_inputs, nums)
        self.play(m.FadeIn(layer_nodes[0]), run_time=.35)
        self.play(m.LaggedStart(*[m.FadeIn(n) for n in list(layer_nodes)[1:]], lag_ratio=.25),
                  m.Create(links), run_time=1.0)
        self.pulse(links, .8)
        values = m.VGroup(*[label(f'{v:.1f}', n.get_center()[0], -.55, 30, GOLD)
                             for n, v in zip(layer_nodes, layer_outputs())])
        frame = m.RoundedRectangle(width=5.55, height=2.0, corner_radius=.22,
                                    color=TEAL, stroke_width=2).move_to([-.15, .1, 0])
        layer_name = label('One layer · four neurons', -.15, -1.8, 29, TEAL)
        own_weights = label('Different weights + biases', -.15, -2.65, 29, MUTED)
        self.play(m.FadeIn(values), m.Create(frame), m.FadeIn(layer_name), m.FadeIn(own_weights), run_time=.7)
        self.finish()
        layer = m.VGroup(layer_nodes, layer_inputs, links, nums, values, frame, layer_name, own_weights)

        self.begin(4, 'Layers become\na network')
        net, layers, edges = network()
        self.play(m.FadeOut(nums), m.FadeOut(values), m.FadeOut(frame),
                  m.FadeOut(layer_name), m.FadeOut(own_weights), run_time=.3)
        self.play(m.Transform(layer_inputs, layers[0]), m.Transform(layer_nodes, layers[1]),
                  m.Transform(links, edges[0]), run_time=.8)
        self.remove(layer_inputs, layer_nodes, links)
        self.add(layers[0], layers[1], edges[0])
        row_labels = m.VGroup(label('Inputs', -.15, 3.45, 25, BLUE),
                              label('Outputs', -.15, -2.7, 25, GOLD))
        for edge, row in zip(edges[1:], layers[2:]):
            self.play(m.Create(edge), m.FadeIn(row), run_time=.5)
        self.play(m.FadeIn(row_labels), run_time=.25)
        for edge in edges:
            self.pulse(edge, .45)
        chain = label('Outputs become inputs', -.15, -3.4, 29, MUTED)
        self.play(m.FadeIn(chain), run_time=.25)
        self.finish()

        self.begin(5, 'Training changes\nthe connections')
        self.play(m.FadeOut(chain), run_time=.25)
        error = label('Prediction error', -.15, -3.35, 29, PINK)
        self.play(m.FadeIn(error), run_time=.25)
        for edge in reversed(edges):
            reversed_paths = m.VGroup(*[m.Line(line.get_end(), line.get_start()) for line in edge])
            self.pulse(reversed_paths, .45, PINK)
        self.play(*[line.animate.set_stroke(width=1.4 + (i % 4), opacity=.75)
                    for edge in edges for i, line in enumerate(edge)], run_time=.8)
        learned = label('Adjust weights + biases', -.15, -3.35, 29, TEAL)
        self.play(m.FadeOut(error), m.FadeIn(learned), run_time=.35)
        for edge in edges:
            self.pulse(edge, .35)
        self.finish()

        self.begin(6, 'From neurons\nto medicine')
        self.play(m.FadeOut(net), m.FadeOut(row_labels), m.FadeOut(learned), run_time=.4)
        scan = self.scan(-.15, 2.0)
        scan_label = label('Medical image', -.15, 3.45, 27, BLUE)
        schematic = label('Schematic illustration', -.15, .6, 29, MUTED)
        mini, mini_layers, mini_edges = network(rows=(3, 4, 2), top=.1, spacing=.8)
        mini.scale(.6).move_to([-.15, -1.0, 0])
        arrow = m.Arrow([-.15, .35, 0], [-.15, -.2, 0], color=BLUE, buff=0, stroke_width=3)
        review = label('Flag a region for review', -.15, -2.85, 30, GOLD)
        clinician = label('Supports clinician review', -.15, -3.55, 29, MUTED)
        self.play(m.FadeIn(scan), m.FadeIn(scan_label), m.FadeIn(schematic), run_time=.7)
        self.play(m.Create(arrow), m.FadeIn(mini), run_time=.5)
        for edge in mini_edges:
            self.pulse(edge, .5)
        region = m.Circle(radius=.23, color=GOLD, stroke_width=4).move_to([.4, 2.1, 0])
        self.play(m.Create(region), m.FadeIn(review), m.FadeIn(clinician), run_time=.7)
        self.finish()
        medical = m.VGroup(scan, scan_label, schematic, mini, arrow, region, review, clinician)

        self.begin(7, 'One network.\nA real task.')
        self.play(m.FadeOut(medical), run_time=.4)
        small = node(-.15, 2.7, .33)
        many = m.VGroup(*[node(x, 1.15, .22) for x in [-1.65, -.65, .35, 1.35]])
        final_net, _, final_edges = network(rows=(3, 4, 2), top=0, spacing=.7)
        final_net.scale(.6).move_to([-.15, -1.1, 0])
        labels = m.VGroup(label('Neuron', 1.25, 2.7, 28), label('Layer', 2.2, 1.15, 28),
                          label('Trained network', -.15, -2.5, 29, TEAL))
        self.play(m.FadeIn(small), m.FadeIn(many), m.FadeIn(final_net), m.FadeIn(labels), run_time=.6)
        for edge in final_edges:
            self.pulse(edge, .4, GOLD)
        self.finish()

    def begin(self, index, title, animate=True):
        self.begin_beat(index)
        heading = m.Text(title, font='Space Grotesk', weight='BOLD', font_size=46,
                         color=WHITE, line_spacing=.1)
        if heading.width > 6.0:
            heading.scale(6.0 / heading.width)
        heading.move_to([-3.05 + heading.width / 2, 4.6, 0])
        if animate:
            animations = [m.FadeIn(heading, shift=.1 * m.UP)]
            if self.heading is not None:
                animations.append(m.FadeOut(self.heading))
            self.play(*animations, run_time=.25)
        else:
            self.add(heading)
        self.heading = heading


    def scan(self, x, y):
        # An original abstract chest-scan icon, not patient data or model evidence.
        plate = m.RoundedRectangle(width=2.6, height=2.3, corner_radius=.2,
                                    color=BLUE, fill_color='#1B2941', fill_opacity=1,
                                    stroke_width=2).move_to([x, y, 0])
        left = m.Ellipse(width=.75, height=1.5, color='#9FB9D9', fill_opacity=.15,
                         stroke_width=2).move_to([x-.48, y, 0])
        right = left.copy().shift(.96*m.RIGHT)
        spine = m.Line([x, y+.8, 0], [x, y-.8, 0], color=MUTED, stroke_width=3)
        ribs = m.VGroup(*[m.Arc(radius=.85, start_angle=.25, angle=2.65,
                                color=MUTED, stroke_width=1.5).stretch(.28, 1)
                          .move_to([x, y+dy, 0]) for dy in [-.5, -.15, .2, .55]])
        return m.VGroup(plate, left, right, spine, ribs)
