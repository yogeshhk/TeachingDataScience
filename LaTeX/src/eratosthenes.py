import os
import sys

import numpy as np
from pyx import canvas, color, path, text, unit

def draw_grid():
    c.stroke(path.rect(0, 0, 25, 2))
    for n in range(24):
        c.stroke(path.line(n+1, 0, n+1, 2))
    c.stroke(path.line(0, 1, 25, 1))

text.set(text.LatexRunner)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
unit.set(xscale=1.2, wscale=2.5)

c = canvas.canvas()
c.fill(path.rect(0, 1, 2, 1), [color.grey(0.7)])
c.text(0.5, 1.5, '0', [text.halign.center, text.valign.middle])
c.text(1.5, 1.5, '1', [text.halign.center, text.valign.middle])
basename = os.path.splitext(sys.argv[0])[0]
baseprimes = [0, 2, 3, 5, 7]
ncolor = len(baseprimes)-1
cancelled = set([0, 1])
for nr, baseprime in enumerate(baseprimes):
    if nr == 0:
        for n in range(2, 50):
            x = n % 25
            y = 2-(n//25)
            c.text(x+0.5, y-0.5, str(n), [text.halign.center, text.valign.middle])
    else:
        cancelled.add(baseprime)
        hvalue = 1.1*(nr-1)/(ncolor-1)
        hvalue = hvalue-int(hvalue)
        primecolor = color.hsb(hvalue, 1, 0.8)
        x = baseprime % 25
        y = 2-(baseprime//25)
        c.fill(path.rect(x, y, 1, -1), [primecolor])
        c.text(x+0.5, y-0.5, r'\textbf{%s}' % baseprime,
               [text.halign.center, text.valign.middle, color.grey(1)])
        for n in range(baseprime**2, 50, baseprime):
            if not n in cancelled:
                cancelled.add(n)
                x = n % 25
                y = 2-(n//25)
                c.stroke(path.line(x, y-1, x+1, y), [primecolor])
                c.stroke(path.line(x, y, x+1, y-1), [primecolor])
    draw_grid()
    c.writePDFfile('%s_%s' % (basename, nr+1))

for n in range(50):
    if not n in cancelled:
        x = n % 25
        y = 2-(n//25)
        c.fill(path.rect(x, y, 1, -1), [color.hsb(0.15, 1, 0.8)])
        c.text(x+0.5, y-0.5, r'\textbf{%s}' % n,
               [text.halign.center, text.valign.middle, color.grey(1)])
draw_grid()
c.writePDFfile('%s_%s' % (basename, nr+2))

