from pyx import canvas, color, deco, path, style, text, unit

def draw_square(x, y, kante):
    c.fill(path.rect(x, y, kante, kante), [color.grey(1)])
    c.stroke(path.line(x, y, x+kante, y+kante), [style.linewidth.thick, color.grey(0.5)])
    c.stroke(path.rect(x, y, kante, kante), [style.linewidth.thick])

text.set(text.LatexRunner)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
unit.set(xscale=0.85, wscale=1.2)
c = canvas.canvas()

kante = 1
dist = 0.15
punkte = 1
nrboxes = 3
nrpoints = 3

ldist = 0.05
boxcolor = color.rgb(1, 0.7, 0.4)
c.fill(path.rect(-0.3*dist, -0.2, 7*kante+6.6*dist, kante+0.4), [boxcolor])
for n in range(nrboxes):
    x = n*(kante+dist)
    draw_square(x, 0, kante)
    c.text(x+ldist*kante, (1-ldist)*kante, n, [text.valign.top])
    nstr = ""
    if n>0: nstr = "%+i" % n
    c.text(x+(1-0.5*ldist)*kante, ldist*kante, '-N'+nstr, [text.halign.right])
    x = (n+nrboxes)*(kante+dist)+dist+punkte
    draw_square(x, 0, kante)
    c.text(x+ldist*kante, (1-ldist)*kante, 'N'+str(n-3), [text.valign.top])
    c.text(x+(1-0.5*ldist)*kante, ldist*kante, str(n-3), [text.halign.right])

xoffset = nrboxes*(kante+dist)
for n in range(nrpoints):
    c.fill(path.circle(xoffset+(0.5+n)*punkte/nrpoints, 0.5*kante, 0.05*kante))

c.writePDFfile()
