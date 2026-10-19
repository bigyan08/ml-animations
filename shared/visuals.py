"""Small diagram primitives; scene composition remains a lesson decision."""
import manimgx as m
import numpy as np


def node(x, y, radius=.27, color='#4DE0C1'):
    return m.Circle(radius=radius, color=color, fill_color='#14283A',
                    fill_opacity=1, stroke_width=3).move_to([x, y, 0])


def connection(a, b, color='#75AEFF', width=2, opacity=.5):
    direction = b.get_center()-a.get_center()
    direction /= np.linalg.norm(direction)
    return m.Line(a.get_center()+direction*a.width/2,
                  b.get_center()-direction*b.width/2, color=color,
                  stroke_width=width, stroke_opacity=opacity)
