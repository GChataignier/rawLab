# -*- coding: utf-8 -*-
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

    if npImin.shape[-1] == pattern.shape[2]:
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
### Other functions
####################################################################################################
def HDRfunc(imin, exposure=1, power=1, offset=0):
    return 1-np.exp(-exposure*(imin+offset)**power)






# EOF
