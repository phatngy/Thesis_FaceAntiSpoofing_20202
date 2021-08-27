import os
import random
from utils import *

# DATA_ROOT = r'/u01/DATA/DATA/antispoofing/CASIA-SURF'
# TRN_IMGS_DIR = DATA_ROOT + '/Training/'
# VAL_IMGS_DIR = DATA_ROOT + '/Val/'
# TST_IMGS_DIR = DATA_ROOT

DATA_ROOT = r'/u01/khienpv1/Phat/face_anti/FaceBagNet_FAS_CVPR2019/data/split_norm'
TRN_IMGS_DIR = DATA_ROOT + '/train/'
VAL_IMGS_DIR = DATA_ROOT + '/val/'
TST_IMGS_DIR = DATA_ROOT + '/test/'

RESIZE_SIZE = 112
def load_train_list():
    list = []
    f = open(DATA_ROOT + '/train_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    random.shuffle(list)
    return list

def load_val_list():
    list = []
    # f = open(DATA_ROOT + '/val_private_list.txt')
    f = open(DATA_ROOT + '/val_list.txt')

    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list():
    list = []
    # f = open(DATA_ROOT + '/test_private_list.txt')
    f = open(DATA_ROOT + '/test_list.txt')

    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list



