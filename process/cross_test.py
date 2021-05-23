import os


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