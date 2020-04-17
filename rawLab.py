# -*- coding: utf-8 -*-
"""
@author: Guillaume Chataignier
"""
####################################################################################################
### Import
####################################################################################################
import os
from time import time
import torch
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
# Reading RAW file
rawPath = "./sample.CR2"
rawNumpy, raw = imio.raw2numpy(rawPath)
plt.matshow(rawNumpy, cmap='gray', aspect='equal')
plt.title("RAW")

# Applying Bayer pattern for better visualisation
bayered = imm.raw2bayer(rawNumpy, pattern = imc.RGGBPattern)
plt.figure()
plt.imshow(bayered, aspect='equal')
plt.title("Bayer RGB")

# Showing bilinear kernel for demosaicing
print("Bilinear Kernel red / blue : ")
print(imc.bilinearDK[:,:,0])
print("Bilinear Kernel green : ")
print(imc.bilinearDK[:,:,1])

# Bilinear Demosaicing using integrated rawPy/LibRaw
start = time()
debayeredUINT8 = raw.postprocess(demosaic_algorithm=rp.DemosaicAlgorithm.LINEAR)
end = time()
print("Ellapsed time for LibRaw's linear demosaicing : ", end-start)
plt.figure()
plt.imshow(debayeredUINT8, aspect='equal')
plt.title("Demosaic, LibRaw")

# Bilinear Demosaicing using convolution (CPU mode)
start = time()
debayeredPersoCPU = imf.imconvScipy(bayered, imc.bilinearDK)
end = time()
print("Ellapsed time for handmade linear demosaicing (CPU) : ", end-start)
plt.figure()
plt.imshow(debayeredPersoCPU, aspect='equal')
plt.title("Demosaic, hand CPU")

# Bilinear Demosaicing using convolution (GPU mode)
if torch.cuda.is_available():
    # Do a small convolution to initialize pytorch
    imf.imconvTorchGPU(np.random.rand(100,100,3), np.ones([3,3,3]))
    start = time()
    debayeredPersoGPU = imf.imconvTorchGPU(bayered, imc.bilinearDK)
    end = time()
    print("Ellapsed time for handmade linear demosaicing (GPU) : ", end-start)
    plt.figure()
    plt.imshow(debayeredPersoGPU, aspect='equal')
    plt.title("Demosaic, hand GPU")


####################################################################################################
### Image Processing
####################################################################################################
# Normalize debayered image
debayeredNorm = imm.normalize(debayeredPersoCPU)
plt.figure()
plt.imshow(debayeredNorm, aspect='equal')
plt.title("Normalized debayered image")

# White Balance : selection of a supposed white pixel or small area
wbc_img = imm.setWB(debayeredNorm, pCoord=(450,1450), extend=25, cR=1, cG=1, cB=0.95)
plt.figure()
plt.imshow(wbc_img, aspect='equal')
plt.title("White Balanced image")

# Simple exposure coefficient
exc_img = imm.simpleExposure(wbc_img, EV=2.5)
plt.figure()
plt.imshow(exc_img, aspect='equal')
plt.title("Simple exposure correction")

# Blending
E = 8
P = 0.9
O = 0
x=np.arange(0, 3, 1e-3)
y=1-np.exp(-E*(x+O)**P)
plt.figure()
plt.plot(x,y)
plt.title("Blending curve")
blended_img = imm.HDRfunc(wbc_img, exposure=E, power=P, offset=O)
plt.figure()
plt.imshow(blended_img, aspect='equal')
plt.title("Blended")

# Set Saturation, hue, Luminance via HSV (then come back to RGB)
cH = 0 # Additive coeff (color angle)
cS = 1.6 # multiplicative coeff
cV = 1.05 # multiplicative coeff
hsv = imm.rgb2hsv(blended_img)
hsv[:,:,0] += cH
hsv[:,:,1] *= cS
hsv[:,:,2] *= cV
final_img = np.clip(imm.hsv2rgb(hsv), 0, 1)
plt.figure()
plt.imshow(final_img, aspect='equal')
plt.title("Final")


####################################################################################################
### Some Metrics and filters
####################################################################################################
# edgesLaplacian = imf.Laplacian(final_img)
# plt.matshow(np.abs(edgesLaplacian), cmap="gray")

# gradX = imf.SobelX(final_img)
# plt.matshow(np.abs(gradX), cmap="gray")

# gradY = imf.SobelY(final_img)
# plt.matshow(np.abs(gradY), cmap="gray")

# Compute edges
gradT = imf.Sobel(final_img)
plt.matshow(gradT, cmap="gray")








# EOF