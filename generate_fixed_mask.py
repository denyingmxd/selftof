import numpy as np
import torch
from PIL import Image as pil
import random
import os
import tqdm
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


import numpy as np


def generate_mask(shape, drop_rate):
    while True:
        # Generate a random array of the given shape with values between 0 and 1
        random_array = np.random.rand(shape)

        # Create a mask where each element is 1 if the corresponding element in random_array is greater than the drop_rate
        mask = (random_array > drop_rate).astype(int)

        # Ensure that the mask is not all zeros
        if np.any(mask):
            # print("wow")
            break
        else:
            print('not lucky')

    return mask





train_files = readlines('/data/laiyan/codes/PLNet/splits/nyu/train_files.txt')
test_files = readlines('/data/laiyan/codes/PLNet/splits/nyu/test_files.txt')
data_root = '/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/'
train=False
files = train_files if train else test_files

seed = 117010053 if train else 115105
random.seed(seed)  # Fix the seed for reproducibility
np.random.seed(seed)  # Fix the seed for reproducibility
torch.manual_seed(seed)  # Fix the seed for reproducibility

drop_rate=0.2
shape=64
for i in tqdm.tqdm(range(len(files))):
    index = files[i].split()[-1]
    scene = files[i].split()[0]
    pp = os.path.join(data_root, scene, index+ "_fixed_mask_drop_{}.npz".format(str(drop_rate)))
    fixed_mask = generate_mask(shape=shape,drop_rate=drop_rate)
    # print(fixed_mask.shape)
    # print(fixed_mask)
    np.savez(pp, fixed_mask=fixed_mask)
    # print(pp)
    # exit()