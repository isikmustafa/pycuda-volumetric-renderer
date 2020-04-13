import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.autoinit import context
import numpy as np
import os
import matplotlib.pyplot as plt
import struct
import tkinter as tk
from PIL import Image, ImageTk
import configargparse
from pycuda.compiler import SourceModule

p = configargparse.ArgParser(default_config_files=['configs.txt'])
p.add('--inputVolumePath', required=True, help='Path to file in .xyz format')
p.add('--numberOfSamples', required=True, help='Number of samples per pixel')
p.add('--outputWidth', required=True)
p.add('--outputHeight', required=True)
p.add('--cameraPosition', required=True, action="append")
p.add('--blockSize', required=True, help='Kernel launches blocks of size (block_size)x(block_size). Adjust according to your GPU.')
p.add('--absorptionFactor', required=True, help='To get absorption coefficient of a voxel, multiply voxel density with absorptionFactor.')
p.add('--scatteringFactor', required=True, help='To get scattering coefficient of a voxel, multiply voxel density with scatteringFactor.')
configs = p.parse_args()

with open(configs.inputVolumePath, 'rb') as volume_file:
    x_dim = struct.unpack('i', volume_file.read(4))[0]
    y_dim = struct.unpack('i', volume_file.read(4))[0]
    z_dim = struct.unpack('i', volume_file.read(4))[0]
    min_density = struct.unpack('f', volume_file.read(4))[0]
    max_density = struct.unpack('f', volume_file.read(4))[0]
    volume_data = np.frombuffer(volume_file.read(), dtype=np.float32).reshape((x_dim, y_dim, z_dim)) / max_density

with open('kernel.cpp', 'r') as code_file:
    source_file = code_file.read()

number_of_samples = int(configs.numberOfSamples)
output_width = int(configs.outputWidth)
output_height = int(configs.outputHeight)
camera_position = np.asarray([float(el) for el in configs.cameraPosition]).astype(np.float32)
block_size = int(configs.blockSize)
absorption_factor = float(configs.absorptionFactor)
scattering_factor = float(configs.scatteringFactor)

#Compute bbox of the volume such that largest edge is normalized to length 1.
bbox_size = np.asarray([x_dim, y_dim, z_dim]).astype(np.float32)
bbox_size = bbox_size / np.max(bbox_size)

#Compute camera coordinate system.
camera_lookat = bbox_size * 0.5
w = camera_lookat - camera_position
w /= np.linalg.norm(w)
v = np.asarray([0.0, 1.0, 0.0]).astype(np.float32)
u = np.cross(v, w)
u /= np.linalg.norm(u)
v = np.cross(w, u)
camera_system = np.concatenate((u[np.newaxis], v[np.newaxis], w[np.newaxis]), axis=0)
    
mod = SourceModule(source_file % { "NUMBER_OF_GENERATORS" : output_width * output_height }, no_extern_c=True, include_dirs=[os.getcwd()])
init_rand_num_generators = mod.get_function("initRandNumGenerators")
render = mod.get_function("render")

#Settings for the volume density texture.
texture = mod.get_texref("gVolumeTexture")
texture.set_address_mode(0, cuda.address_mode.BORDER)
texture.set_address_mode(1, cuda.address_mode.BORDER)
texture.set_address_mode(2, cuda.address_mode.BORDER)
texture.set_filter_mode(cuda.filter_mode.LINEAR)
texture.set_format(cuda.array_format.FLOAT, 1)
texture.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
texture.set_array(cuda.np_to_array(volume_data, order="F"))

output = np.zeros((output_height, output_width, 3)).astype(np.float32)

init_rand_num_generators(block=(block_size * block_size,1,1), grid=((output_width * output_height) // (block_size * block_size) + 1, 1, 1))

#GUI
window = tk.Tk()
canvas = tk.Canvas(window, width=output_width, height=output_height)
canvas.pack()
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW)

for sample_no in range(1, number_of_samples + 1):
    render(cuda.InOut(output), np.int32(output_width), np.int32(output_height),
            cuda.In(bbox_size), cuda.In(camera_position), cuda.In(camera_system),
            np.float32(absorption_factor), np.float32(scattering_factor),
            block=(block_size,block_size,1), grid=(output_width // block_size + 1, output_height // block_size + 1, 1)
            )
    context.synchronize()
    
    window.title("A Mini Volumetric Renderer on GPU --- " + "Sample Count: " + str(sample_no))
    photo = ImageTk.PhotoImage(image = Image.fromarray((output / sample_no * 255.0).astype(np.uint8)))
    canvas.itemconfig(image_on_canvas, image=photo)
    window.update()

plt.imsave('render.png', output / number_of_samples)
window.mainloop()