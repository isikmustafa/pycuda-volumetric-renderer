import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import matplotlib.pyplot as plt
import struct

from pycuda.compiler import SourceModule

with open('C:/Users/Mustafa/Desktop/smoke_density.xyz', 'rb') as volume_file:
    x_dim = struct.unpack('i', volume_file.read(4))[0]
    y_dim = struct.unpack('i', volume_file.read(4))[0]
    z_dim = struct.unpack('i', volume_file.read(4))[0]
    min_density = struct.unpack('f', volume_file.read(4))[0]
    max_density = struct.unpack('f', volume_file.read(4))[0]
    volume_data = np.frombuffer(volume_file.read(), dtype=np.float32).reshape((x_dim, y_dim, z_dim))

with open('kernel.cpp', 'r') as code_file:
    source_file = code_file.read()

#Inputs
#TODO: Get these with argparse!
RENDERER_WIDTH = 512
RENDERER_HEIGHT = 768
CAMERA_POSITION = np.asarray([0.5, 0.5, -0.3]).astype(np.float32)
BLOCK_SIZE = 16 #Kernel launches blocks of size BLOCK_SIZExBLOCK_SIZE. Adjust according to your GPU.

#Compute bbox of the volume such that largest edge is normalized to length 1.
bbox_size = np.asarray([x_dim, y_dim, z_dim]).astype(np.float32)
bbox_size = bbox_size / np.max(bbox_size)

#Compute camera coordinate system.
camera_lookat = bbox_size * 0.5
w = camera_lookat - CAMERA_POSITION
w /= np.linalg.norm(w)
v = np.asarray([0.0, 1.0, 0.0]).astype(np.float32)
u = np.cross(v, w)
u /= np.linalg.norm(u)
v = np.cross(w, u)
camera_system = np.concatenate((u[np.newaxis], v[np.newaxis], w[np.newaxis]), axis=0)
    
mod = SourceModule(source_file, no_extern_c=True, include_dirs=[os.getcwd()])
render = mod.get_function("render")

#Settings for the volume density texture.
texture = mod.get_texref("volume_tex")
texture.set_address_mode(0, cuda.address_mode.BORDER)
texture.set_address_mode(1, cuda.address_mode.BORDER)
texture.set_address_mode(2, cuda.address_mode.BORDER)
texture.set_filter_mode(cuda.filter_mode.LINEAR)
texture.set_format(cuda.array_format.FLOAT, 1)
texture.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
texture.set_array(cuda.np_to_array(volume_data, order="F"))

output = np.zeros((RENDERER_HEIGHT, RENDERER_WIDTH, 3)).astype(np.float32)

render(cuda.InOut(output), np.int32(RENDERER_WIDTH), np.int32(RENDERER_HEIGHT),
        cuda.In(bbox_size), cuda.In(CAMERA_POSITION), cuda.In(camera_system), cuda.In(np.float32()),
        block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(RENDERER_WIDTH // BLOCK_SIZE + 1, RENDERER_HEIGHT // BLOCK_SIZE + 1, 1))

plt.imsave('C:/Users/Mustafa/Desktop/name.png', output)