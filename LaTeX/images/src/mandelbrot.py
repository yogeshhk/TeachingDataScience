# image needs to be cropped before including it into
# the presentation

import numpy as np
import matplotlib.pyplot as plt

npts = 1500
nthres = 50
niter = 1000
xmin = -2
xmax = 1
ymin = -1.5
ymax = 1.5

y, x = np.ogrid[ymin:ymax:npts*1j, xmin:xmax:npts*1j]
c = x+1j*y
z = c
for j in range(niter):
    z = z**2+c
imdata = (np.abs(z) < nthres)
plt.imshow(imdata, cmap='gray',
           extent=(xmin, xmax, ymin, ymax), origin='bottom')
plt.xlabel('Re(c)', fontsize=20)
plt.ylabel('Im(c)', fontsize=20)
plt.savefig('mandelbrot.png', dpi=200)
