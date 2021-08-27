import os
import numpy as np
import torchvision.transforms.functional as F

DATA_ROOT = r'/u01/khienpv1/DATA/antispoof/public/CASIA-SURF'

TRN_IMGS_DIR = DATA_ROOT + '/Training/'
VAL_IMGS_DIR = DATA_ROOT + '/Val/'
TST_IMGS_DIR = DATA_ROOT + '/Testing/'
RESIZE_SIZE = 112

def load_test_list():
    list = []
    f = open(DATA_ROOT + '/train_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list


class NonZeroCrop(object):
    """Cut out black regions.
    """
    def __call__(self, img):
        arr = np.asarray(img)
        pixels = np.transpose(arr.nonzero())
        if len(arr.shape) > 2:
            pixels = pixels[:, :-1]
        top = pixels.min(axis=0)
        h, w = pixels.max(axis=0) - top
        return F.crop(img, top[0], top[1], h, w)

