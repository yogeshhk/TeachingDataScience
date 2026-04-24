import numpy as np
import numpy.polynomial.polynomial as P

from pyx import color, deco, graph, style

np.random.seed(987)
x = np.pi*np.linspace(0, 1, 100)
y = np.sin(x)+0.1*np.random.rand(100)
fit = P.Polynomial(P.polyfit(x, y, 2))

g = graph.graphxy(width=8,
        x=graph.axis.lin(title=r'\Large $x$', divisor=np.pi,
            texter=graph.axis.texter.rational(suffix=r'\pi')),
        y=graph.axis.lin(min=0, max=1.1, title=r'\Large $y$',
            parter=graph.axis.parter.lin(tickdists=[0.2])))
origdata = list(zip(x, y))
symbolattrs = [deco.filled, color.hsb(0.6, 1, 0.7)]
g.plot(graph.data.points(origdata, x=1, y=2),
       [graph.style.symbol(graph.style.symbol.circle, 0.07,
                           symbolattrs=symbolattrs)])
fitdata = list(zip(x, fit(x)))
lineattrs = [color.hsb(0.05, 1, 0.7), style.linewidth.THick]
g.plot(graph.data.points(fitdata, x=1, y=2),
       [graph.style.line(lineattrs=lineattrs)])
g.writePDFfile()
