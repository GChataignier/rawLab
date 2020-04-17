# -*- coding: utf-8 -*-
"""
@author: Guillaume Chataignier
"""
###############################################################################
### Import
###############################################################################
import numpy as np
from imgConstant import RGGBPattern

####################################################################################################
### Mosaic functions
####################################################################################################
# Check validity of Bayer Pattern (a single 1 for each pixel)
def checkBayer(BP):
    if BP.ndim == 3 and BP.dtype == 'bool':
        sY, sX = BP.shape[0], BP.shape[1]
        flagVal = True
        errMsg = ''
        for ii in range(sY):
            for jj in range(sX):
                if not np.sum(BP[ii,jj])==1:
                    flagVal=False
                    errMsg='Bayer pattern not valid: at least one pixel sees multiple channels.'
    else:
        flagVal=False
        errMsg = 'Bayer pattern must be boolean array in 3 dimensions.'
    return flagVal, errMsg

# Full color images (single or pile) to bayered images given a CFA pattern
def color2bayer(npImin, pattern = RGGBPattern, f_check = True):
    if f_check:
        bayerVal, errMsg = checkBayer(pattern)
        if not bayerVal:
            raise Exception(errMsg)

    if npImin.shape[2] == pattern.shape[2]:
        if npImin.ndim==3: # Single RGB image
            sY, sX = npImin.shape[0], npImin.shape[1]
            Ny = sY//pattern.shape[0] + 1
            Nx = sX//pattern.shape[1] + 1
            tiledPattern = np.tile(pattern, (Ny, Nx, 1))
            return npImin * tiledPattern[:sY, :sX, :]

        elif npImin.ndim==4: # Multiple RGB images stacked in the 1st dimension
            sY, sX = npImin.shape[1], npImin.shape[2]
            Ny = sX//pattern.shape[0] + 1
            Nx = sX//pattern.shape[1] + 1
            tiledPattern = np.tile(pattern, (Ny, Nx, 1))
            np_imout = np.zeros(npImin.shape, dtype=np.float32)
            for ii in range(npImin.shape[0]):
                np_imout[ii] = npImin[ii] * tiledPattern[:sY, :sX, :]
            return np_imout

    else:
        raise Exception('Input image and input pattern have not the same depth')

# RAW images (single or pile) to bayered images given a CFA pattern
def raw2bayer(npImin, pattern = RGGBPattern, f_check = True):
    if f_check:
        bayerVal, errMsg = checkBayer(pattern)
        if not bayerVal:
            raise Exception(errMsg)

    if npImin.ndim==2: # Single image
        sY, sX = npImin.shape[0], npImin.shape[1]
        Ny = sY//pattern.shape[0] + 1
        Nx = sX//pattern.shape[1] + 1
        tiledPattern = np.tile(pattern, (Ny, Nx, 1))

        np_imout = np.zeros((sY, sX, pattern.shape[2]), dtype=np.float32)
        for ii in range(pattern.shape[2]):
            np_imout[:,:,ii] = npImin * tiledPattern[:sY,:sX,ii]
        return np_imout

    elif npImin.ndim==3: # Multiple images stacked in the 1st dimension
        sY, sX = npImin.shape[1], npImin.shape[2]
        Ny = sY//pattern.shape[0] + 1
        Nx = sX//pattern.shape[1] + 1
        tiledPattern = np.tile(pattern, (Ny, Nx, 1))

        np_imout = np.zeros((npImin.shape[0], sY, sX, pattern.shape[2]), dtype=np.float32)
        for jj in range(npImin.shape[0]):
            for ii in range(pattern.shape[2]):
                np_imout[jj, :,:,ii] = npImin[jj] * tiledPattern[:sX,:sY,ii]
        return np_imout

####################################################################################################
### Exposure functions
####################################################################################################
def normalize(imin):
    return (imin-imin.min()) / (imin.max() - imin.min())

def simpleExposure(imin, EV):
    exposureCoeff = 2**EV
    return np.clip(imin * exposureCoeff, 0, 1) # -> exc = EXposure corrected


def HDRfunc(imin, exposure=1, power=1, offset=0):
    return 1-np.exp(-exposure*(imin+offset)**power)


####################################################################################################
### Color functions
####################################################################################################
def setWB(imin, pCoord, extend, cR=1, cG=1, cB=1):
    pY, pX = pCoord # -> coordinates of pixel supposed white
    ext = extend # -> patches of size 2*ext+1
    WBPatch = imin[pY-ext:pY+ext+1, pX-ext:pX+ext+1]
    meanR = np.mean(WBPatch[:,:,0])
    meanG = np.mean(WBPatch[:,:,1])
    meanB = np.mean(WBPatch[:,:,2])
    wbc = imin # wbc = White Balanced Corrected
    wbc[:,:,0] *= cR/meanR
    wbc[:,:,1] *= cG/meanG
    wbc[:,:,2] *= cB/meanB
    wbc /= wbc.max()
    return wbc

# Code from https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python
def rgb2hsv(imin):
    maxv = np.amax(imin, axis=2)
    maxc = np.argmax(imin, axis=2)
    minv = np.amin(imin, axis=2)
    minc = np.argmin(imin, axis=2)
    hsv = np.zeros(imin.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((imin[..., 1] - imin[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((imin[..., 2] - imin[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((imin[..., 0] - imin[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv
    return hsv

def hsv2rgb(imin):
    hi = np.floor(imin[..., 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = imin[..., 2].astype('float')
    f = (imin[..., 0] / 60.0) - np.floor(imin[..., 0] / 60.0)
    p = v * (1.0 - imin[..., 1])
    q = v * (1.0 - (f * imin[..., 1]))
    t = v * (1.0 - ((1.0 - f) * imin[..., 1]))
    rgb = np.zeros(imin.shape)
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]
    return rgb





# EOF
