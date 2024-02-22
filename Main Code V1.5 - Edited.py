import numpy as np
from os import walk
from skimage.io import imread_collection
import imageio

dir = './Outputs/Main Code V1.5/All Mask'

names = next(walk(dir), (None, None, []))[2]
data_names = []
dir_name = dir + '/'
data_tmp = [dir_name + s for s in names]
data_names.append(data_tmp)
data = np.array(imread_collection(data_names[0]), dtype='float')
data[data < 128] = 0
data[data >= 128] = 255

for i in range(data.shape[0]):
    imageio.imwrite('./Outputs/Main Code V1.5/All Mask - Edited/' + list(names)[i], data[i])
