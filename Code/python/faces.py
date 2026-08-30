import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

face = misc.face(gray=True)

plt.imshow(face, cmap=plt.cm.gray)
plt.savefig('face0.png')
plt.clf()

bwimage = np.zeros_like(face)
bwimage[face > 128] = 255
plt.imshow(bwimage, cmap=plt.cm.gray)
plt.savefig('face1.png')
plt.clf()

framedface = np.zeros_like(face)
framedface[31:-30, 31:-30] = face[31:-30, 31:-30]
plt.imshow(framedface, cmap=plt.cm.gray)
plt.savefig('face2.png')
plt.clf()

darkface = 255*(face/255)**1.5
plt.imshow(darkface, cmap=plt.cm.gray)
plt.savefig('face3.png')
plt.clf()

sy, sx = face.shape
y, x = np.ogrid[0:sy, 0:sx]
centerx, centery = (660, 300)
mask = ((y - centery)**2 + (x - centerx)**2) > 230**2
face[mask] = 0
plt.imshow(face, cmap=plt.cm.gray)
plt.savefig('face4.png')
