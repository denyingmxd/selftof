import torch, torchvision
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print("mmdetection:",mmdet.__version__)

# Check mmcv installation
import mmcv
print("mmcv:",mmcv.__version__)

# Check mmengine installation
import mmengine
print("mmengine:",mmengine.__version__)
import os
#list all avaibale models from mmseg.api
from tqdm import tqdm
import mmcv
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

# exit()

split='train'
fpath = os.path.join(os.path.dirname(__file__), "splits", 'nyu', "{}_files.txt")
base_dir = '/data/laiyan/codes/calibrated-backprojection-network/data/nyu_hl/data/'
filenames = readlines(fpath.format(split))

# inferencer = MMSegInferencer(model='mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640',
#                              weights="/data/laiyan/codes/PLNet/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth"
#                             )
#list all avaibale models

import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
# Choose to use a config and initialize the detector
config_file = "/data/laiyan/codes/PLNet/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py"
checkpoint_file = "/data/laiyan/codes/PLNet/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

# config_file = "/data/laiyan/codes/PLNet/xxx.py"
# checkpoint_file = "/data/laiyan/codes/PLNet/xxx.pth"
#
# config_file = "/data/laiyan/codes/PLNet/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py"
# checkpoint_file = "/data/laiyan/codes/PLNet/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d.pth"
# register all modules in mmdet into the registries
register_all_modules()


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

from mmdet.registry import VISUALIZERS

# init visualizer(run the block only once in jupyter notebook)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta


for idx,file in enumerate(filenames):
    folder, frame_index = file.split()
    fff = os.path.join(base_dir, folder, str(frame_index) + ".jpg")
    print(idx,fff)
    if not os.path.exists(fff):
        print('not exist')
        continue

    image = mmcv.imread(fff, channel_order='rgb')
    result = inference_detector(model, image)
    print(result)
    # result = inference_model(model, fff)
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=None,
        wait_time=0,
    )
    visualizer.show()
    # pred_sem_seg = result.pred_sem_seg.data.detach().cpu()[0]
    # plt.imshow(pred_sem_seg)
    # plt.show()
    # np.savez_compressed(os.path.join(base_dir, folder, str(frame_index) + "_segformer.npz"), pred_sem_seg.numpy())
    exit()
# result = inferencer(data_path, show=True,return_datasamples=True)
# pred_sem_seg = result.pred_sem_seg.data.detach().cpu()
