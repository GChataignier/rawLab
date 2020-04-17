# -*- coding: utf-8 -*-
"""
@author: Guillaume Chataignier
"""
####################################################################################################
### Import
####################################################################################################
import numpy as np
import torch
from scipy import ndimage
import imgConstant as imc

####################################################################################################
### Convolution functions
####################################################################################################
# Convolution 2D + multiple channels, chooses between CPU and GPU version
def imconv2d(imin, kernel, gpuID = 0):
    if torch.cuda.is_available():
        return imconvTorchGPU(imin, kernel, gpuID)
    else:
        return imconvScipy(imin, kernel)

# Convolution RGB using scipy convolve
def imconvScipy(imin, kernel):
    # Kernel dimension: (Height, Width, Chan)
    # Image dimension : (Height, Width, Chan)
    lostY, lostX = kernel.shape[0]-1, kernel.shape[1]-1
    bY, eY, bX, eX = int(np.floor(lostY/2)), int(np.ceil(lostY/2)), int(np.floor(lostX/2)), int(np.ceil(lostX/2))

    if kernel.ndim == 2 and imin.ndim == 2:
        return ndimage.convolve(imin, kernel, mode='constant', cval=0)[bY:-eY, bX:-eX] # Valid convolution

    elif kernel.ndim == 3 and imin.ndim ==3 and kernel.shape[2]==imin.shape[2]:
        chan_out = [ndimage.convolve(imin[:,:,cc], kernel[:,:,cc], mode='constant', cval=0) for cc in range(imin.shape[2])]
        return np.stack(chan_out, axis=2)[bY:-eY, bX:-eX] # Valid convolution
    else:
        raise Exception("Problem with dimensions")

# Convolution RGB using pytorch conv2d, GPU
# Faster than Scipy CPU version if CUDA is available
def imconvTorchGPU(imin, kernel,gpuID=0):
    # Kernel, numpy array of dimension: (Height, Width, Chan)
    # Image, numpy array of dimension : (Height, Width, Chan)
    device = torch.device("cuda:"+str(int(gpuID)))
    # Force cpu to see the RAM consumption peak vs the imconvTorchCPU version

    if kernel.ndim == 2 and imin.ndim == 2:
        kernel = torch.from_numpy(kernel[np.newaxis, np.newaxis].astype(imin.dtype)).to(device)
        imin = torch.from_numpy(imin[np.newaxis, np.newaxis]).to(device)
        return torch.nn.functional.conv2d(imin, kernel).squeeze().cpu().numpy()

    elif kernel.ndim == 3 and imin.ndim ==3 and kernel.shape[2]==imin.shape[2]:
        kY, kX, kC = kernel.shape[0], kernel.shape[1], kernel.shape[2]
        kernelT = torch.zeros(kC, kC, kY, kX)
        for cc in range(kC):
            kernelT[cc,cc] = torch.from_numpy(kernel[:,:,cc])
        kernelT = kernelT.to(device)

        imin = torch.from_numpy(imin[np.newaxis]).type(kernelT.dtype).permute(0,3,1,2).to(device)
        imout = torch.nn.functional.conv2d(imin, kernelT)
        return imout.squeeze().squeeze().permute(1,2,0).cpu().numpy()
    else:
        raise Exception("Problem with dimensions")


####################################################################################################
### Filters functions
####################################################################################################
# Compute Laplacian
def Laplacian(imin):
    kernel = imc.laplacianKernel
    if imin.ndim == 3:
        L = np.sum(imin, 2)
        L= imconv2d(L, kernel)
    elif imin.ndim == 2:
        L = imconv2d(imin, kernel)
    else:
        raise Exception("Invalid dimension")
    return L

def SobelX(imin):
    kernel = imc.sobelXKernel
    if imin.ndim == 3:
        L = np.sum(imin, 2)
        L= imconv2d(L, kernel)
    elif imin.ndim == 2:
        L = imconv2d(imin, kernel)
    else:
        raise Exception("Invalid dimension")
    return L

def SobelY(imin):
    kernel = imc.sobelYKernel
    if imin.ndim == 3:
        L = np.sum(imin, 2)
        L= imconv2d(L, kernel)
    elif imin.ndim == 2:
        L = imconv2d(imin, kernel)
    else:
        raise Exception("Invalid dimension")
    return L

def Sobel(imin):
    return np.sqrt(SobelX(imin)**2 + SobelY(imin)**2)

# EOF
