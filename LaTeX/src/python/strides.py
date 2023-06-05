import sys
import os.path

from pyx import canvas, color, deco, path, text, unit

def make_stride_figure(c, lowerstride, uperstride=1, nrentries=6):
    ht = 0.5
    wd = 2
    dist = 0.2
    textcolor = color.hsb(0.02, 1, 0.6)
    for n in range(nrentries):
        x = n*(wd+dist)
        c.stroke(path.rect(x, 0, wd, ht))
        c.text(x+0.5*wd, 0.5*ht, str(n), [text.halign.center, text.valign.middle])

    for n in range(nrentries-1):
        x = n*(wd+dist)
        c.stroke(path.curve(x-dist/3, ht+0.5*dist,
                            x+0.3*wd, ht+3*dist,
                            x+0.7*wd, ht+3*dist,
                            x+wd+dist/3, ht+0.5*dist),
                 [deco.earrow.large])
        c.text(x+0.5*wd, ht+3.2*dist, r'\Large 8', [text.halign.center, textcolor])

    if lowerstride:
        for n in range((nrentries-1)//lowerstride):
            x = n*lowerstride*(wd+dist)
            c.stroke(path.curve(x-dist/3, -0.5*dist,
                                x+0.5*wd, -5*dist,
                                x+(lowerstride-0.5)*wd+lowerstride*dist, -5*dist,
                                x+lowerstride*wd+(lowerstride-0.7)*dist, -0.5*dist),
                     [deco.earrow.large])
            c.text(x+0.5*lowerstride*wd+dist,-5.2*dist, r'\Large %i' % (lowerstride*8),
                   [text.halign.center, text.valign.top, textcolor])

text.set(text.LatexRunner)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
unit.set(xscale=1.2, wscale=1.5)

for stride in (0, 2, 3):
    c = canvas.canvas()
    make_stride_figure(c, stride)
    c.writePDFfile('_'.join([os.path.splitext(sys.argv[0])[0], str(stride)]))
