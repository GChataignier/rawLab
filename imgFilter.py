# -*- coding: utf-8 -*-
####################################################################################################
### Import
####################################################################################################
import numpy as np
import torch
from scipy import ndimage
from imgConstant import bilinearDK

####################################################################################################
### Filters functions
####################################################################################################
# Convolution 2D + multiple channels, chooses between CPU and GPU version
def imconv2d(imin, kernel, gpuID = 0):
    if torch.cuda.is_available():
        return imconvTorchGPU(imin, kernel, gpuID)
    else:
        return imconvScipy(imin, kernel)


# Convolution RGB using scipy convolve
def imconvScipy(imin, kernel, f_valid = True):
    # Kernel dimension: (Height, Width, Chan)
    # Image dimension : (Height, Width, Chan)
    chan_out = [ndimage.convolve(imin[:,:,cc], kernel[:,:,cc], mode='constant', cval=0) for cc in range(imin.shape[2])]
    lostY, lostX = kernel.shape[0]-1, kernel.shape[1]-1
    if f_valid:
        bY, eY, bX, eX = int(np.floor(lostY/2)), int(np.ceil(lostY/2)), int(np.floor(lostX/2)), int(np.ceil(lostX/2))
        return np.stack(chan_out, axis=2)[bY:-eY, bX:-eX] # Valid convolution
    else:
        return np.stack(chan_out, axis=2)


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


# Convolution RGB using pytorch conv2d, CPU
# Useless, for testing only. Scipy version is better
def imconvTorchCPU(imin, kernel):
    # Kernel, numpy array of dimension: (Height, Width, Chan)
    # Image, numpy array of dimension : (Height, Width, Chan)
    device = torch.device('cpu')

    if kernel.ndim == 2 and imin.ndim == 2:
        kernel = torch.from_numpy(kernel[np.newaxis, np.newaxis].astype(imin.dtype)).to(device)
        imin = torch.from_numpy(imin[np.newaxis, np.newaxis]).to(device)
        return torch.nn.functional.conv2d(imin, kernel).squeeze().cpu().numpy()

    elif kernel.ndim == 3 and imin.ndim ==3 and kernel.shape[2]==imin.shape[2]:
        # Avoid huge RAM consumption by procesing each channel sequentially
        kernel = torch.from_numpy(kernel.astype(imin.dtype)).unsqueeze(0).permute(0,3,1,2).to(device) # dims : batch=1, Chan, H, W
        imin = torch.from_numpy(imin).unsqueeze(0).permute(0,3,1,2).to(device)
        chan_out = [torch.nn.functional.conv2d(imin[0:1, cc:cc+1,:,:], kernel[0:1, cc:cc+1,:,:]) for cc in range(imin.shape[1])]
        return torch.stack([chan_out[cc].squeeze() for cc in range(len(chan_out))], axis=2).cpu().numpy()
    else:
        raise Exception("Problem with dimensions")


# Bilinear demosaicing perso
def linearDemosaicing(imin):
    kernel = bilinearDK
    Rchan = ndimage.convolve(imin[:,:,0], kernel[:,:,0], mode='constant', cval=0)
    Gchan = ndimage.convolve(imin[:,:,1], kernel[:,:,1], mode='constant', cval=0)/2
    Bchan = ndimage.convolve(imin[:,:,2], kernel[:,:,2], mode='constant', cval=0)

    lostY, lostX = kernel.shape[0]-1, kernel.shape[1]-1
    bY, eY, bX, eX = int(np.floor(lostY/2)), int(np.ceil(lostY/2)), int(np.floor(lostX/2)), int(np.ceil(lostX/2))
    return np.stack([Rchan, Gchan, Bchan], axis=2)[bY:-eY, bX:-eX] # Valid convolution


# EOF
