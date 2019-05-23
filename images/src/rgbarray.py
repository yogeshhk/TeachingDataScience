from pyx import canvas, color, graph, path, style, unit

def frontplane(z, nxmax, mymax, facecolor, edgecolor, trans):
    p = path.path(path.moveto(*projector(0, z, 0)),
                  path.lineto(*projector(nxmax, z, 0)),
                  path.lineto(*projector(nxmax, z, nymax)),
                  path.lineto(*projector(0, z, nymax)),
                  path.closepath())
    c.fill(p, [facecolor, color.transparency(trans)])
    c.stroke(p, [edgecolor])
    for nx in range(1, nxmax):
        x0, y0 = projector(nx, z, 0)
        x1, y1 = projector(nx, z, nymax)
        c.stroke(path.line(x0, y0, x1, y1), [edgecolor])
    for ny in range(1, nymax):
        x0, y0 = projector(0, z, ny)
        x1, y1 = projector(nxmax, z, ny)
        c.stroke(path.line(x0, y0, x1, y1), [edgecolor])

def corner(nx, ny, z, facecolor, edgecolor, trans, xdir, ydir):
    if xdir:
        p = path.path(path.moveto(*projector(nx, z, ny)),
                      path.lineto(*projector(nx-1, z, ny)),
                      path.lineto(*projector(nx-1, z+1, ny)),
                      path.lineto(*projector(nx, z+1, ny)),
                      path.closepath())
        c.fill(p, [facecolor, color.transparency(trans)])
    if ydir:
        p = path.path(path.moveto(*projector(nx, z, ny)),
                      path.lineto(*projector(nx, z, ny+1)),
                      path.lineto(*projector(nx, z+1, ny+1)),
                      path.lineto(*projector(nx, z+1, ny)),
                      path.closepath())
        c.fill(p, [facecolor, color.transparency(trans)])
    x0, y0 = projector(nx, z, ny)
    x1, y1 = projector(nx, z+1, ny)
    c.stroke(path.line(x0, y0, x1, y1), [edgecolor])

projector = graph.graphxyz.central(60, -50, 25).point

unit.set(wscale=1.5)
c = canvas.canvas()
nxmax = 7
nymax = 5
trans = 0.4
edgecolors = (color.rgb(0, 0, 0.8),
              color.rgb(0, 0.6, 0),
              color.rgb(0.8, 0, 0))
w = 0.3
facecolors = (color.rgb(w, w, 1),
              color.rgb(w, 1, w),
              color.rgb(1, w, w))
for nplane, (edgecolor, facecolor) in enumerate(zip(edgecolors, facecolors)):
    zoff = 1.04*(2-nplane)
    frontplane(zoff+1, nxmax, nymax, facecolor, edgecolor, trans)
    for nx in range(nxmax, -1, -1):
        for ny in range(nymax+1):
            corner(nx, ny, zoff, facecolor, edgecolor, trans,
                   nx != 0, ny != nymax)
    frontplane(zoff, nxmax, nymax, facecolor, edgecolor, trans)
    x0, y0 = projector(nxmax, zoff+1, nymax)
    x1, y1 = projector(0, zoff+1, nymax)
    x2, y2 = projector(0, zoff+1, 0)
    p = path.path(path.moveto(x0, y0), path.lineto(x1, y1),
                  path.lineto(x2, y2))
    c.stroke(p, [edgecolor])
c.writePDFfile()
