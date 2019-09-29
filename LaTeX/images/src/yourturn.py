from itertools import product
from math import pi

import numpy as np
import bpy

npts = 100
radius = 3
height = 0.2
phi = np.linspace(0, pi, npts)

lowervertices = [(x, y, -height) for x, y in zip(radius*np.cos(phi), radius*np.sin(phi))]
uppervertices = [(x, y, height) for x, y in zip(radius*np.cos(phi), radius*np.sin(phi))]
arrowvertices = [(radius, 0, 2*height), (radius, 0, -2*height), (radius, -6*height, 0)]
vertices = lowervertices + uppervertices + arrowvertices
offset = len(vertices)

loweredges = [(n, n+1) for n in range(npts-1)]
upperedges = [(n, n+1) for n in range(npts, 2*npts-1)]
verticaledges = [(n, n+npts) for n in range(npts)]
arrowedges = [(offset-1, offset-2), (offset-2, offset-3), (offset-3, offset-1)]
edges = loweredges + upperedges + verticaledges + arrowedges

cylinderfaces = [(n, n+1, n+1+npts, n+npts) for n in range(npts-1)]
arrowfaces = [(offset-3, offset-2, offset-1)]
faces = cylinderfaces + arrowfaces

mesh = bpy.data.meshes.new(name='Cylinder Arrow Mesh')
mesh.from_pydata(vertices, edges, faces)
obj = bpy.data.objects.new('Cylinder Arrow', mesh)
obj.location = bpy.context.scene.cursor_location

mat = bpy.data.materials.new(name='Cylinder Arrow Material')
mat.diffuse_color = (1, 0.6, 0.0)
obj.active_material = mat

scene = bpy.context.scene
scene.objects.link(obj)
scene.objects.active = obj

lamp = bpy.data.lamps.new(name="Arrow Lamp", type='POINT')
lamp_object = bpy.data.objects.new(name="Arrow Lamp", object_data=lamp)
scene.objects.link(lamp_object)
lamp_object.location = (5, -0.4, 0.2)
lamp_object.select = True
scene.objects.active = lamp_object

cam = bpy.data.cameras.new("Camera")
cam_object = bpy.data.objects.new("Camera", cam)
cam_object.location = (-2.0, -1.0, 2.0)
scene.objects.link(cam_object)

world = scene.world
world.horizon_color = (1, 1, 1)
