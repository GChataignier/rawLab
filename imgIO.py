# -*- coding: utf-8 -*-
####################################################################################################
### Import
####################################################################################################
import rawpy as rp
import numpy as np
import PIL

####################################################################################################
### Read and reshape images
####################################################################################################
def img2numpy(imName, f_verbose = False, f_keep3chan = True):
    if f_verbose: print('Reading', imName)
    img = np.asarray(PIL.Image.open(imName), dtype = np.float32)/255
    if img.ndim ==2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if f_verbose: print('B&W image: duplicating data on third axis')
    if img.ndim==3 and img.shape[2]>3 and f_keep3chan:
        img = img[:,:,0:3]
        if f_verbose: print('Image has more than 3 channels: taking only the 3 first ones')
    if img.ndim==3 and img.shape[2]>3 and not f_keep3chan:
        if f_verbose: print('Image has more than 3 channels: keeping all channels')
    if img.ndim <2 or img.ndim>3:
        raise Exception('Problem with dimensions')
    return img

def raw2numpy(rawPath, f_norm=True):
    raw = rp.imread(rawPath)
    rawData = np.array(raw.raw_image)
    if f_norm:
        maxValue = rawData.max()
    else:
        maxValue=1
    return rawData/maxValue, raw

def numpy2img(imin, filename):
    im2save = (imin * 255 / np.max(imin)).astype('uint8')
    im2save = PIL.Image.fromarray(im2save)
    im2save.save(filename, compress_level=1)


# EOF

