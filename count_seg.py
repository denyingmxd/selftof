import timm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines



split='train'
fpath = os.path.join(os.path.dirname(__file__), "splits", 'nyu', "{}_files.txt")
base_dir = '/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/'
filenames = readlines(fpath.format(split))
maxs=[]
mins=[]
for idx,file in enumerate(filenames):
    folder, frame_index = file.split()
    fff = os.path.join(base_dir, folder, str(frame_index) + ".jpg")
    print(idx,fff)
    if not os.path.exists(fff):
        print('not exist')
        raise ValueError('not exist')


    seg_path = os.path.join(base_dir, folder, str(frame_index) + "_segformer.npz")
    if not os.path.exists(seg_path):
        print('not exist')
        raise ValueError('not exist')

    seg = np.load(seg_path)['arr_0']
    maxs.append(np.max(seg))
    mins.append(np.min(seg))
print(np.max(maxs),np.min(mins))