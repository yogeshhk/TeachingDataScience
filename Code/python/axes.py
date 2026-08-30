from pyx import canvas, color, deco, path, text, trafo, unit

text.set(text.LatexRunner)
color0 = color.rgb(0.8, 0, 0)
color1 = color.rgb(0, 0, 0.8)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
text.preamble(r'\usepackage{color}')
text.preamble(r'\definecolor{axis0}{rgb}{%s, %s, %s}' % (color0.r, color0.g, color0.b))
text.preamble(r'\definecolor{axis1}{rgb}{%s, %s, %s}' % (color1.r, color1.g, color1.b))
unit.set(xscale=1.2, wscale=1.5)

dx = 2
dy = 0.8
c = canvas.canvas()
for nx in range(3):
    for ny in range(3):
        c.text(nx*dx, -ny*dy,
               r'a[\textcolor{axis0}{%s}, \textcolor{axis1}{%s}]' % (ny, nx),
               [text.halign.center])
box = c.bbox()
pd = 0.1
xoff = box.left()-pd
pdx = 0.2
pdy = 0.5
c.stroke(path.curve(xoff, box.top()+pd,
                    xoff-pdx, box.top()-pdy,
                    xoff-pdx, box.bottom()+pdy,
                    xoff, box.bottom()-pd))
xoff = box.right()+pd
c.stroke(path.curve(xoff, box.top()+pd,
                    xoff+pdx, box.top()-pdy,
                    xoff+pdx, box.bottom()+pdy,
                    xoff, box.bottom()-pd))
x = box.left()-pdx-0.4
c.stroke(path.line(x, box.top(), x, box.bottom()), [deco.earrow, color0])
c.text(x-0.1, 0.5*(box.top()+box.bottom()), 'axis 0',
       [text.halign.center, color0, trafo.rotate(90)])
y = box.top()+0.4
c.stroke(path.line(box.left(), y, box.right(), y), [deco.earrow, color1])
c.text(0.5*(box.left()+box.right()), y+0.1, 'axis 1', [text.halign.center, color1])
c.writePDFfile()
