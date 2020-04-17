# -*- coding: utf-8 -*-
"""
@author: Guillaume Chataignier
"""
###############################################################################
### Import
###############################################################################
import numpy as np

####################################################################################################
### Define constants
####################################################################################################
UnityPattern = np.ones((1,1,3), dtype = bool)

RGGBPattern = np.zeros((2,2,3), dtype = bool)
RGGBPattern[:, :, 0] = [[1, 0],
                        [0, 0]] # Red
RGGBPattern[:, :, 1] = [[0, 1],
                        [1, 0]] # Green
RGGBPattern[:, :, 2] = [[0, 0],
                        [0, 1]] # Blue

bilinearDK = np.zeros([3,3,3])
bilinearDK[:,:,0] = np.array([[1,2,1], [2,4,2], [1,2,1]])/4
bilinearDK[:,:,1] = np.array([[0,1,0], [1,4,1], [0,1,0]])/4
bilinearDK[:,:,2] = np.array([[1,2,1], [2,4,2], [1,2,1]])/4

LaplacianKernel = -np.array([[1,1,1], [1,-8,1], [1,1,1]])
SobelXKernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
SobelYKernel = np.transpose(SobelXKernel)


imgFormat = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pbm', '.pgm', '.ppm')
rawFormat = ('.cr2', '.nef', '.arw')
INF = np.inf
PI = np.pi

# EOF
