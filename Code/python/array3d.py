from itertools import product
from math import atan2, pi, sqrt

from pyx import canvas, color, deco, path, text, trafo, unit

text.set(text.LatexRunner)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
unit.set(xscale=1.2, wscale=1.5)

frontplane = canvas.canvas()
backplane = canvas.canvas()
xcells = 4
ycells = 3
xshift = 0.8
yshift = 1.2
dist = 0.2
myred = color.rgb(0.8, 0, 0)
mygreen = color.rgb(0, 0.6, 0)
myblue = color.rgb(0, 0, 0.8)
for c, start in ((frontplane, 0), (backplane, xcells*ycells)):
    c.stroke(path.rect(0, 0, 4, 3),
             [deco.filled([color.grey(1), color.transparency(0.2)])])
    for x in range(1, xcells):
        c.stroke(path.line(x, 0, x, ycells))
    for y in range(1, ycells):
        c.stroke(path.line(0, y, xcells, y))
    for entry in range(xcells*ycells):
        x = entry % 4
        y = ycells - entry // 4
        c.text(x+0.5, y-0.5, str(start+entry),
               [text.halign.center, text.valign.middle])
c = canvas.canvas()
c.insert(backplane, [trafo.translate(xshift, yshift)])
for x, y in product((0, xcells), (0, ycells)):
    c.stroke(path.line(x, y, x+xshift, y+yshift))
c.insert(frontplane)
dx = -dist*yshift/sqrt(xshift**2+yshift**2)
dy = dist*xshift/sqrt(xshift**2+yshift**2)
c.stroke(path.line(dx, ycells+dy, dx+xshift, ycells+dy+yshift),
         [deco.earrow, myred])
c.text(0.5*xshift+2*dx, ycells+0.5*yshift+2*dy, 'axis 0',
       [text.halign.center, myred,
        trafo.rotate(180/pi*atan2(yshift, xshift))])
c.stroke(path.line(-dist, ycells, -dist, 0),
         [deco.earrow, mygreen])
c.text(-2*dist, 0.5*ycells, 'axis 1',
       [text.halign.center, mygreen, trafo.rotate(90)])
c.stroke(path.line(0, -dist, xcells, -dist),
         [deco.earrow, myblue])
c.text(0.5*xcells, -2*dist, 'axis 2',
       [text.halign.center, text.valign.top, myblue])
c.writePDFfile()
