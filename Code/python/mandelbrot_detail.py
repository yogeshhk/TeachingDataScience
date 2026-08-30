import numpy as np
import matplotlib.pyplot as plt

npts = 1500
nthres = 5000
niter = 2000
xmin = -1.74892
xmax = -1.74887
ymin = 0.0006
ymax = 0.00065

y, x = np.ogrid[ymin:ymax:npts*1j, xmin:xmax:npts*1j]
c = x+1j*y
z = np.ma.zeros((npts, npts), dtype=complex)
imdata = np.ma.zeros((npts, npts), dtype=float)
for j in range(niter):
    z = z**2+c
    imdata[np.abs(z) > nthres] = imdata[np.abs(z) > nthres]+j
    z[np.abs(z) > nthres] = np.ma.masked
    imdata[np.abs(z) > nthres] = np.ma.masked
imdata.mask = False
imdata = np.sqrt(imdata)
plt.imsave('mandelbrot_detail.png', imdata, cmap='afmhot')
