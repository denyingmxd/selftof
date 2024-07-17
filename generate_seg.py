import timm
import matplotlib.pyplot as plt
from mmseg.apis import MMSegInferencer
import numpy as np
import torch
import os
#list all avaibale models from mmseg.api
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

inferencer = MMSegInferencer(model='segformer_mit-b5_8xb2-160k_ade20k-640x640',
                             weights="/data/laiyan/codes/PLNet/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth")
for idx,file in enumerate(filenames):
    folder, frame_index = file.split()
    fff = os.path.join(base_dir, folder, str(frame_index) + ".jpg")
    print(idx,fff)
    if not os.path.exists(fff):
        print('not exist')
        continue

    result = inferencer(fff, show=False, return_datasamples=True)
    pred_sem_seg = result.pred_sem_seg.data.detach().cpu()[0]
    # plt.imshow(pred_sem_seg)
    # plt.show()
    np.savez_compressed(os.path.join(base_dir, folder, str(frame_index) + "_segformer.npz"), pred_sem_seg.numpy())
    # exit()
# result = inferencer(data_path, show=True,return_datasamples=True)
# pred_sem_seg = result.pred_sem_seg.data.detach().cpu()
