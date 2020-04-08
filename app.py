import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import matplotlib.pyplot as plt

from pycuda.compiler import SourceModule

with open('kernel.cpp', 'r') as file:
    source_file = file.read()

RENDERER_WIDTH = 512
RENDERER_HEIGHT = 512
CAMERA_POSITION = np.asarray([0.0, 0.0, -1.5]).astype(np.float32)
    
mod = SourceModule(source_file, no_extern_c=True, include_dirs=[os.getcwd()])
renderer = mod.get_function("renderDeltaTracking")

output = np.zeros((RENDERER_WIDTH, RENDERER_HEIGHT, 3)).astype(np.float32)

renderer(
        cuda.InOut(output), np.int32(RENDERER_WIDTH), np.int32(RENDERER_HEIGHT), cuda.In(CAMERA_POSITION),
        block=(16,16,1), grid=(32,32,1))

plt.imsave('C:/Users/Mustafa/Desktop/name.png', output)