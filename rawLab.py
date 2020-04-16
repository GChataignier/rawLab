# -*- coding: utf-8 -*-
####################################################################################################
### Import
####################################################################################################
import os
from time import time
import numpy as np
import rawpy as rp
import matplotlib.pyplot as plt

if '__file__' in (globals()):
    curr_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    if not curr_dir == file_dir:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print('Changing current directory to script directory')

import imgConstant as imc
import imgIO as imio
import imgModification as imm
import imgFilter as imf

####################################################################################################
### RAW Processing
####################################################################################################

rawPath = "./sample.CR2"
rawNumpy, raw = imio.raw2numpy(rawPath)
plt.matshow(rawNumpy, cmap='gray', aspect='equal')

bayered = imm.raw2bayer(rawNumpy, pattern = imc.RGGBPattern)
plt.figure()
plt.imshow(bayered, aspect='equal')

start = time()
debayeredUINT8 = raw.postprocess(demosaic_algorithm=rp.DemosaicAlgorithm.LINEAR)
end = time()
print("Ellapsed time for LibRaw linear demosaicing : ", end-start)
plt.figure()
plt.imshow(debayeredUINT8, aspect='equal')

start = time()
debayeredPersoCPU = imf.linearDemosaicing(bayered)
end = time()
print("Ellapsed time for handmade linear demosaicing (CPU) : ", end-start)
plt.figure()
plt.imshow(debayeredPersoCPU, aspect='equal')

# Comment the following block if no GPU with CUDA is available
###########################################################################
import torch
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

start = time()
debayeredPersoGPU = imf.imconvTorchGPU(bayered, imc.bilinearDK)
end = time()
print("Ellapsed time for handmade linear demosaicing (GPU) : ", end-start)
plt.figure()
plt.imshow(debayeredPersoGPU, aspect='equal')
###########################################################################


test = imm.color2bayer(debayeredUINT8, pattern = imc.RGGBPattern)
plt.figure()
plt.imshow(test)


# EOF