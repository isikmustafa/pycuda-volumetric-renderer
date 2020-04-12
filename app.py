import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.autoinit import context
import numpy as np
import os
import matplotlib.pyplot as plt
import struct
import tkinter as tk
from PIL import Image, ImageTk

from pycuda.compiler import SourceModule

with open('C:/Users/Mustafa/Desktop/smoke_density.xyz', 'rb') as volume_file:
    x_dim = struct.unpack('i', volume_file.read(4))[0]
    y_dim = struct.unpack('i', volume_file.read(4))[0]
    z_dim = struct.unpack('i', volume_file.read(4))[0]
    min_density = struct.unpack('f', volume_file.read(4))[0]
    max_density = struct.unpack('f', volume_file.read(4))[0]
    volume_data = np.frombuffer(volume_file.read(), dtype=np.float32).reshape((x_dim, y_dim, z_dim)) / max_density

with open('kernel.cpp', 'r') as code_file:
    source_file = code_file.read()

#Inputs
#TODO: Get these with argparse!
NUMBER_OF_SAMPLES = 1024
RENDERER_WIDTH = 768
RENDERER_HEIGHT = 1024
CAMERA_POSITION = np.asarray([0.5, 0.5, -0.3]).astype(np.float32)
BLOCK_SIZE = 16 #Kernel launches blocks of size BLOCK_SIZExBLOCK_SIZE. Adjust according to your GPU.
ABSORPTION_FACTOR = 400 #To get absorption coefficient of a voxel, multiply voxel density with ABSORPTION_FACTOR
SCATTERING_FACTOR = 250 #To get scattering coefficient of a voxel, multiply voxel density with SCATTERING_FACTOR

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
    
mod = SourceModule(source_file % { "NUMBER_OF_GENERATORS" : RENDERER_WIDTH * RENDERER_HEIGHT }, no_extern_c=True, include_dirs=[os.getcwd()])
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

output = np.zeros((RENDERER_HEIGHT, RENDERER_WIDTH, 3)).astype(np.float32)

init_rand_num_generators(block=(BLOCK_SIZE * BLOCK_SIZE,1,1), grid=((RENDERER_WIDTH * RENDERER_HEIGHT) // (BLOCK_SIZE * BLOCK_SIZE) + 1, 1, 1))

#GUI
window = tk.Tk()
canvas = tk.Canvas(window, width=RENDERER_WIDTH, height=RENDERER_HEIGHT)
canvas.pack()
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW)

for sample_no in range(1, NUMBER_OF_SAMPLES + 1):
    render(cuda.InOut(output), np.int32(RENDERER_WIDTH), np.int32(RENDERER_HEIGHT),
            cuda.In(bbox_size), cuda.In(CAMERA_POSITION), cuda.In(camera_system),
            np.float32(ABSORPTION_FACTOR), np.float32(SCATTERING_FACTOR),
            block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=(RENDERER_WIDTH // BLOCK_SIZE + 1, RENDERER_HEIGHT // BLOCK_SIZE + 1, 1)
            )
    context.synchronize()
    
    window.title("A Minimal Volumetric Renderer on GPU --- " + "Sample Count: " + str(sample_no))
    photo = ImageTk.PhotoImage(image = Image.fromarray((output / sample_no * 255.0).astype(np.uint8)))
    canvas.itemconfig(image_on_canvas, image=photo)
    window.update()

plt.imsave('C:/Users/Mustafa/Desktop/render.png', output / NUMBER_OF_SAMPLES)
window.mainloop()