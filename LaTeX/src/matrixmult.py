from itertools import product

import numpy as np
from pyx import canvas, color, deco, path, text, trafo, unit

def matrix_22(m, dx=0.5, dy=0.5, pd=0.1, pdx=0.1, pdy=0.3):
    c = canvas.canvas()
    for nx in range(2):
        for ny in range(2):
            c.text(nx*dx, -ny*dy, str(m[ny, nx]), [text.halign.center])
    box = c.bbox()
    xoff = box.left()-pd
    c.stroke(path.curve(xoff, box.top()+pd,
                        xoff-pdx, box.top()-pdy,
                        xoff-pdx, box.bottom()+pdy,
                        xoff, box.bottom()-pd))
    xoff = box.right()+pd
    c.stroke(path.curve(xoff, box.top()+pd,
                        xoff+pdx, box.top()-pdy,
                        xoff+pdx, box.bottom()+pdy,
                        xoff, box.bottom()-pd))
    return c


text.set(text.LatexRunner)
color0 = color.rgb(0.8, 0, 0)
color1 = color.rgb(0, 0, 0.8)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
text.preamble(r'\usepackage{color}')
text.preamble(r'\definecolor{axis0}{rgb}{%s, %s, %s}' % (color0.r, color0.g, color0.b))
text.preamble(r'\definecolor{axis1}{rgb}{%s, %s, %s}' % (color1.r, color1.g, color1.b))
unit.set(xscale=1.2, wscale=1.5)

c = canvas.canvas()
m1 = np.arange(4).reshape(2, 2)
c_m1 = matrix_22(m1)
m2 = np.arange(4, 8).reshape(2, 2)
c_m2 = matrix_22(m2)
m3 = np.dot(m1, m2)
c_m3 = matrix_22(m3, dx=0.7)
c.insert(c_m1)
c.insert(c_m2, [trafo.translate(c_m1.bbox().width()+0.1, 0)])
end  = c_m1.bbox().right()+c_m2.bbox().width()+0.1
dist2 = 0.6
c.insert(c_m3, [trafo.translate(end+dist2-c_m3.bbox().left(), 0)])
ycenter = 0.5*(c_m1.bbox().top()+c_m1.bbox().bottom())
for dy in (-0.05, 0.05):
    c.stroke(path.line(end+0.15, ycenter+dy,
                       end+dist2-0.15, ycenter+dy))

c_tot = canvas.canvas()
for y in range(4):
    c_tot.insert(c, [trafo.translate(0, 1.5*y)])

dx = 0.2
colorprops = [color.rgb(0.8, 0.2, 0), color.transparency(0.2)]
arrowprops = [deco.earrow]+colorprops
for lineno, (ny, nx) in enumerate(product((0, 1), repeat=2)):
    yoff = 1.5*(3-lineno)+0.17
    c_tot.stroke(path.line(-dx, yoff-0.5*ny, 0.5+dx, yoff-0.5*ny), arrowprops)
    xoff = c_m1.bbox().width()+0.1+0.5*nx
    c_tot.stroke(path.line(xoff, yoff+0.2, xoff, yoff-0.7), arrowprops)
    wd = 0.6
    ht = 0.5
    xoff = end+dist2-c_m3.bbox().left()+0.7*nx
    c_tot.stroke(path.rect(xoff-0.5*wd, yoff-0.5*ny-0.5*ht, wd, ht), colorprops)
c_tot.writePDFfile()
