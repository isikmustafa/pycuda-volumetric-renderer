import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import matplotlib.pyplot as plt
import struct

from pycuda.compiler import SourceModule

with open('C:/Users/Mustafa/Desktop/bunny_cloud_density.xyz', 'rb') as volume_file:
    x_dim = struct.unpack('i', volume_file.read(4))[0]
    y_dim = struct.unpack('i', volume_file.read(4))[0]
    z_dim = struct.unpack('i', volume_file.read(4))[0]
    min_density = struct.unpack('f', volume_file.read(4))[0]
    max_density = struct.unpack('f', volume_file.read(4))[0]
    volume_data = np.frombuffer(volume_file.read(), dtype=np.float32).reshape((x_dim, y_dim, z_dim))

with open('kernel.cpp', 'r') as code_file:
    source_file = code_file.read()

RENDERER_WIDTH = 512
RENDERER_HEIGHT = 512
CAMERA_POSITION = np.asarray([0.5, 0.5, -1.0]).astype(np.float32)
BBOX_SIZE = np.asarray([x_dim, y_dim, z_dim]).astype(np.float32)
BBOX_SIZE = BBOX_SIZE / np.max(BBOX_SIZE)
    
mod = SourceModule(source_file, no_extern_c=True, include_dirs=[os.getcwd()])
render = mod.get_function("render")

texture = mod.get_texref("volume_tex")
texture.set_address_mode(0, cuda.address_mode.BORDER)
texture.set_address_mode(1, cuda.address_mode.BORDER)
texture.set_address_mode(2, cuda.address_mode.BORDER)
texture.set_filter_mode(cuda.filter_mode.LINEAR)
texture.set_format(cuda.array_format.FLOAT, 1)
texture.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
texture.set_array(cuda.np_to_array(volume_data, order="F"))

output = np.zeros((RENDERER_WIDTH, RENDERER_HEIGHT, 3)).astype(np.float32)

render(
        cuda.InOut(output), np.int32(RENDERER_WIDTH), np.int32(RENDERER_HEIGHT), cuda.In(CAMERA_POSITION), cuda.In(BBOX_SIZE),
        block=(16,16,1), grid=(32,32,1))

plt.imsave('C:/Users/Mustafa/Desktop/name.png', output)