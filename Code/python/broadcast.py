from pyx import canvas, color, deco, path, text, style, trafo, unit

def drawgrid(c, nxcells, nycells, yoff, gridcolor=color.grey(0), arange=None):
    c.stroke(path.rect(0, yoff, nxcells, nycells), [gridcolor])
    for nx in range(nxcells-1):
        c.stroke(path.line(nx+1, yoff, nx+1, yoff+nycells), [gridcolor])
    for ny in range(nycells-1):
        c.stroke(path.line(0, yoff+ny+1, nxcells, yoff+ny+1), [gridcolor])
    entry = '1'
    if arange is not None:
        for nx in range(nxcells):
            for ny in range(nycells):
                if arange:
                    entry = str(4*ny+nx)
                c.text(nx+0.5, 2.5-ny, entry,
                       [text.halign.center, text.valign.middle, gridcolor])

def array34(arange, hlshape=None):
    c = canvas.canvas()
    if hlshape is None:
        c.text(2, 3.3, 'shape=(3, 4)', [text.halign.center])
    else:
        c.text(2, 3.3, 'shape=%s' % repr(hlshape), [text.halign.center])
    if hlshape is not None:
        if len(hlshape) == 1:
            hlshape = (1, hlshape[0])
    if arange:
        gridcolor = color.grey(0)
    else:
        gridcolor = color.grey(0.5)
    if hlshape is None:
        arange = True
    elif (hlshape[0] in (1, 3)) and (hlshape[1] in (1, 4)):
        arange = False
    else:
        arange = None
    drawgrid(c, 4, 3, 0, gridcolor, arange=arange)
    if hlshape is not None:
        c.stroke(path.rect(0, 3, hlshape[1], -hlshape[0]),
                 [deco.filled([color.rgb(1, 0.8, 0.4)])])
        drawgrid(c, hlshape[1], hlshape[0], 3-hlshape[0], arange=False)
    if arange is None:
        alertcolor = color.rgb(0.6, 0, 0)
        c.stroke(path.line(0, 0, 4, 3), [alertcolor, style.linewidth.Thick])
        c.stroke(path.line(0, 3, 4, 0), [alertcolor, style.linewidth.Thick])
    return c


text.set(text.LatexRunner)
text.preamble(r'\usepackage{arev}\usepackage[T1]{fontenc}')
unit.set(xscale=1.2, wscale=1.5)

xcells = 4
ycells = 3
gridcolor = color.grey(0.5)

c = canvas.canvas()

c.insert(array34(True))
c.insert(array34(False, (1,)), [trafo.translate(5, 0)])
c.insert(array34(False, (4,)), [trafo.translate(10, 0)])
c.insert(array34(False, (3,)), [trafo.translate(5, -4.5)])
c.insert(array34(False, (3, 1)), [trafo.translate(10, -4.5)])

c.writePDFfile()
