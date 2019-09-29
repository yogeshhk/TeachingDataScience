from itertools import product
import os
import sys

import numpy as np
from pyx import canvas, color, path, text, unit

def arraygraphics(a, idxstr, title=True, xscale=1.0,
               fgcolor=color.grey(1), bgcolor=color.hsb(0.9, 1, 0.5)):
    """create a graphical representation of a two-dimensional array
    
    a         array containing the data to be shown
    slicestr  string defining the slice to be highlighted
    xscale    PyX scaling for text
    fgcolor   color of highlighted data
    bgcolor   color of highlighted cells

    """
    assert a.ndim == 2
    n0, n1 = a.shape
    highlighted = np.zeros_like(a, dtype=bool)
    exec("highlighted{} = True".format(idxstr))
    unit.set(xscale=xscale)
    text.set(text.LatexRunner)
    text.preamble(r"""\usepackage{tgheros}
        \renewcommand*\familydefault{\sfdefault}
        \usepackage[T1]{fontenc}""")
    c = canvas.canvas()
    for ny, nx in zip(*np.nonzero(highlighted)):
        c.fill(path.rect(nx, n0-ny, 1, -1), [bgcolor])
    c.stroke(path.rect(0, 0, n1, n0))
    for nx in range(1, n1):
        c.stroke(path.line(nx, 0, nx, n0))
    for ny in range(1, n0):
        c.stroke(path.line(0, ny, n1, ny))
    textcentered = [text.halign.center, text.valign.middle]
    textcentered_highlighted = textcentered+[fgcolor]
    for nx in range(n1):
        for ny in range(n0):
            if highlighted[ny, nx]:
                textattrs = textcentered_highlighted
            else:
                textattrs = textcentered
            c.text(nx+0.5, n0-ny-0.5, a[ny, nx], textattrs)
    if title:
        textcolor = bgcolor
    else:
        textcolor = color.grey(1)
    titlestr = r"\Large a"+idxstr.replace('%', '\%')
    c.text(0.5*n1, n0+0.4, titlestr, [text.halign.center, textcolor])
    return c

if __name__ == '__main__':
    a = np.arange(40).reshape(5, 8)
    basename = os.path.splitext(sys.argv[0])[0]
    for nr, idxstr in enumerate(('[2, -3]', '[:3, :5]', '[-3:, -3:]',
                                 '[:, 3]', '[1, 3:6]', '[1::2, ::3]',
                                 '[a % 3 == 0]',
                                 '[(1, 1, 2, 2, 3, 3), (3, 4, 2, 5, 3, 4)]')):
        for title in (True, False):
            filename = '_'.join([basename, str(nr)])
            if not title:
                filename = filename+'_wo'
            arraygraphics(a, idxstr, title=title).writePDFfile(filename)
