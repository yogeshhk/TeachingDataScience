from collections import deque

import numpy as np
from pyx import canvas, color, deco, path, text, unit

class Fibonacci():
    def __init__(self, nsquares):
        self.nsquares = nsquares
        self.corners = deque([np.array([0, 0]), np.array([1, 0]),
                              np.array([1, 1]), np.array([0, 1])])
        self.counter = 1
        self.initialize_pyx()
        self.c = canvas.canvas()
        self.draw()

    def initialize_pyx(self):
        text.set(text.LatexRunner)
        text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
        unit.set(xscale=2, wscale=7)

    def draw(self):
        p = path.path(path.moveto(*self.corners[0]),
                      path.lineto(*self.corners[1]),
                      path.lineto(*self.corners[2]),
                      path.lineto(*self.corners[3]),
                      path.closepath())
        fillcolor = color.hsb(2/3*(1-(self.counter-1)/(self.nsquares-1)), 0.2, 1)
        self.c.stroke(p, [deco.filled([fillcolor])])
        x, y = 0.5*(self.corners[0]+self.corners[2])
        s = int(np.sum(np.abs(self.corners[1]-self.corners[0])))
        self.c.text(x, y, str(s),
                    [text.halign.center, text.valign.middle,
                     text.size(min(s, 5))])
        self.counter = self.counter+1

    def draw_next(self):
        corners_old = [a.copy() for a in self.corners]
        self.corners[0] = corners_old[1].copy()
        self.corners[3] = corners_old[2].copy()
        rotate = np.array([[0, 1], [-1, 0]])
        delta = np.dot(rotate, (self.corners[2]-self.corners[1]))
        self.corners[1] = corners_old[1]+delta
        self.corners[2] = corners_old[2]+delta
        self.draw()
        self.corners[0] = corners_old[0].copy()
        self.corners[3] = corners_old[3].copy()
        self.corners.rotate(-1)

    def writePDF(self):
        for n in range(self.nsquares-1):
            self.draw_next()
        self.c.writePDFfile()

Fibonacci(8).writePDF()
