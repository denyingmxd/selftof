# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os, sys

sys.path.append(os.getcwd())
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import ScannetTestPoseDataset
import networks

from tqdm import tqdm


def compute_pose_errors(gt, pr, opt):
    """from https://github.com/princeton-vl/DeepV2D/blob/master/evaluation/eval_utils.py
    """
    # seperate rotations and translations
    R1, t1 = gt[:3, :3], gt[:3, 3]
    R2, t2 = pr[:3, :3], pr[:3, 3]

    costheta = (np.trace(np.dot(R1.T, R2)) - 1.0) / 2.0
    costheta = np.minimum(costheta, 1.0)
    rdeg = np.arccos(costheta) * (180 / np.pi)

    t1mag = np.sqrt(np.dot(t1, t1))
    t2mag = np.sqrt(np.dot(t2, t2))
    costheta = np.dot(t1, t2) / (t1mag * t2mag)
    tdeg = np.arccos(costheta) * (180 / np.pi)

    # fit scales to translations
    if opt.disable_median_scaling:
        a = 1
    else:
        a = np.dot(t1, t2) / np.dot(t2, t2)
    tcm = 100 * np.sqrt(np.sum((t1 - a * t2) ** 2, axis=-1))

    if np.isnan(rdeg) or np.isnan(tdeg) or np.isnan(tcm):
        raise ValueError
    return rdeg, tdeg, tcm


def prepare_model_for_test(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    if opt.rgbd_pose_encoder:
        pose_encoder = networks.RGBD_Pose_Encoder(opt.num_layers,
                             False,
                             num_input_images=2, args=opt)
    else:
        pose_encoder = networks.ResnetEncoder(opt.num_layers,
                             False,
                             num_input_images=2)

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 1)

    pose_encoder_dict  = torch.load(pose_encoder_path, map_location=torch.device('cpu'))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(pose_encoder_dict, "module.")
    pose_encoder.load_state_dict(pose_encoder_dict)

    pose_decoder_dict = torch.load(pose_decoder_path, map_location=torch.device('cpu'))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(pose_decoder_dict, "module.")
    pose_decoder.load_state_dict(pose_decoder_dict)

    pose_encoder.cuda().eval()
    pose_decoder.cuda().eval()

    return pose_encoder, pose_decoder


def evaluate(opt):
    pose_errors = []
    pose_encoder, pose_decoder = prepare_model_for_test(opt)

    filenames = readlines('./splits/scannet_test_pose_deepv2d.txt')
    dataset = ScannetTestPoseDataset(
        opt.data_path,
        filenames,
        opt.height, opt.width,
        frame_idxs=opt.frame_ids,
        opt=opt
    )

    dataloader = DataLoader(
        dataset, 1, shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("-> Computing pose predictions")

    with torch.no_grad():
        for ind, inputs in enumerate(tqdm(dataloader)):
            for key, ipt in inputs.items():
                if type(ipt) == dict:
                    for k, v in ipt.items():
                        inputs[key][k] = v.cuda()
                else:
                    inputs[key] = ipt.cuda()
            color = torch.cat(
                [inputs[("color", i, 0)] for i in opt.frame_ids],
                dim=1,
            )
            if opt.rgbd_pose_encoder:
                features = pose_encoder(color, inputs,[i for i in opt.frame_ids])
            else:
                features = pose_encoder(color)[0]
            axisangle, translation = pose_decoder([features])
            this_pose = transformation_from_parameters(
                axisangle[:, 0],
                translation[:, 0]
            )
            this_pose = this_pose.data.cpu().numpy()[0]
            gt_pose = inputs['pose_gt'].data.cpu().numpy()[0]
            pose_errors.append(compute_pose_errors(gt_pose, this_pose,opt))
            # In P^2Net's code, gt_pose and this_pose are in a wrong order according to DeepV2D's code.

    mean_pose_errors = np.array(pose_errors).mean(0)
    print("\n  " + ("{:>8} | " * 3).format("rot", "tdeg", "tcm"))
    print(("&{: 8.3f}  " * 3).format(*mean_pose_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    result_path = os.path.join(opt.load_weights_folder, "result_scannet_pose.txt")
    f = open(result_path, 'w+')
    print("\n  " + ("{:>8} | " * 3).format("rot", "tdeg", "tcm"), file=f)
    print(("&{: 8.3f}  " * 3).format(*mean_pose_errors.tolist()) + "\\\\", file=f)
    print("\n-> Done!", file=f)


if __name__ == "__main__":
    options = MonodepthOptions()
    options = options.parse()
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Epoch", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3", 'ratio'])
    if options.vis_epoch is not None:
        epochs = [options.vis_epoch]
    else:
        epochs = range(options.num_epochs)
        # epochs = [39]
    for i in epochs:
        options.load_weights_folder = os.path.join(options.load_weights_root, "weights_{}".format(i))
        if not os.path.exists(options.load_weights_folder):
            continue
        assert options.model_name in options.load_weights_folder
        result = [i]
        result.extend(evaluate(options))
        ws.append(result)

    # if options.disable_median_scaling:
    #     ext = 'scannet_pose_no_median'
    # else:
    #     ext = 'scannet_pose_median'
    # ranges = ""
    # if options.eval_min_depth != 0.01 or options.eval_max_depth != 10:
    #     ranges += "_{}_{}".format(options.eval_min_depth, options.eval_max_depth)
    # if options.eval_in_zone:
    #     ranges += "_in_zone"
    # elif options.eval_out_zone:
    #     ranges += "_out_zone"
    # assert options.eval_in_zone + options.eval_out_zone <= 1
    # if options.vis_epoch is not None:
    #     wb.save(
    #         os.path.join(options.load_weights_root, "evaluation_{}_{}{}.xlsx".format(options.vis_epoch, ext, ranges)))
    # else:
    #     wb.save(os.path.join(options.load_weights_root, "evaluation_{}{}.xlsx".format(ext, ranges)))
    #
