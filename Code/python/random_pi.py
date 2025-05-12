from pyx import canvas, color, path, text, unit

text.set(text.LatexRunner)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}\usepackage{nicefrac}')
unit.set(xscale=1.2, wscale=1.2)

c = canvas.canvas()
side = 4
lightcolor = color.hsb(0.65, 0.2, 1)
darkcolor = color.hsb(0.65, 1, 1)
c.fill(path.path(path.moveto(0, 0),
                 path.lineto(side, 0),
                 path.arc(0, 0, side, 0, 90),
                 path.closepath()), [lightcolor])
c.stroke(path.path(path.arc(0, 0, side, 0, 90)), [darkcolor])
c.stroke(path.rect(0, 0, side, side))
ticklen = 0.15
for tick in (0, 1):
    dist = tick*side
    c.stroke(path.line(dist, 0, dist, -ticklen))
    c.text(dist, -1.5*ticklen, str(tick), [text.halign.center, text.valign.top])
    c.stroke(path.line(0, dist, -ticklen, dist))
    c.text(-1.5*ticklen, dist, str(tick), [text.halign.right, text.valign.middle])
c.text(0.4*side, 0.4*side, r'\huge$\nicefrac{\pi}{4}$',
       [text.halign.center, text.valign.middle, darkcolor])
c.writePDFfile()
