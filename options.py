# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)
# parser = argparse.ArgumentParser()



class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Unsupervised Indoor Depth-Pose Learning options",
                                              fromfile_prefix_chars='@',
                                              conflict_handler='resolve'
                                              )
        self.parser.convert_arg_line_to_args = convert_arg_line_to_args
        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "NYU_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.getcwd(), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["nyu"],
                                 default="nyu")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="nyu",
                                 choices=["nyu"])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=384)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=10.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -2, 2])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # conventional options
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        # hyper-parameter
        self.parser.add_argument("--smoothness_weight",
                                 type=float,
                                 help="smoothness weight",
                                 default=0.2)
        self.parser.add_argument("--num_plane_keysets",
                                 type=int,
                                 help="the number of keysets for plane regularization",
                                 default=512)
        self.parser.add_argument("--plane_weight",
                                 type=float,
                                 help="plane regularization weight",
                                 default=2.0)
        self.parser.add_argument("--num_line_keysets",
                                 type=int,
                                 help="the number of keysets for line regularization",
                                 default=128)
        self.parser.add_argument("--line_weight",
                                 type=float,
                                 help="line regularization weight",
                                 default=0.5)

        # ABLATION options
        self.parser.add_argument("--disable_pixel_coordinate_modulation",
                                 help="if set, do not use pixel coordinate modulation,"
                                      "and apply a ReLU to the obtained disparity",
                                 action="store_true")
        self.parser.add_argument("--disable_plane_smoothness",
                                 help="if set, the image-edge-aware smoothness will be applied on the "
                                      "conventional disparity instead of the proposed planar coefficients",
                                 action="store_true")
        self.parser.add_argument("--disable_plane_regularization",
                                 help="if set, do not use plane regularization",
                                 action="store_true")
        self.parser.add_argument("--disable_line_regularization",
                                 help="if set, do not use line regularization",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="nyu",
                                 choices=["nyu"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")


        #add my configs
        self.parser.add_argument("--distributed",action="store_true")
        self.parser.add_argument('--port', default=15000, type=int, help='Which port to use')
        self.parser.add_argument("--load_weights_root",type=str,)
        self.parser.add_argument("--depth_repr",type=str,default='plane',choices=['plane','disp'])
        self.parser.add_argument("--vis_epoch",type=int,default=None)
        self.parser.add_argument("--sparse_depth_input_type",type=str,default=None)
        self.parser.add_argument("--sparse_d_type",type=str,default='none')
        self.parser.add_argument("--extend_fov",action="store_true")
        self.parser.add_argument("--simu_max_distance",type=float,default=4.0)
        self.parser.add_argument("--zone_num",type=int,default=8)
        self.parser.add_argument("--rgbd_encoder",action="store_true")
        self.parser.add_argument("--rgbd_pose_encoder",action="store_true")
        self.parser.add_argument("--sparse_depth_loss_weight",default=0,type=float)
        self.parser.add_argument("--sparse_depth_loss_type",default='none',type=str)
        self.parser.add_argument("--std_weight",default=1.0,type=float)
        self.parser.add_argument("--tof_median_scaling",action="store_true")
        self.parser.add_argument("--load_seg",action="store_true")
        self.parser.add_argument("--seg_ignore_rate",default=0.0,type=float)
        self.parser.add_argument("--margin_level",default=0.0,type=float)
        self.parser.add_argument("--oracle",action="store_true")
        self.parser.add_argument("--scale_by_tof_disp",action="store_true")
        self.parser.add_argument("--scale_by_tof_disp_all",action="store_true")
        self.parser.add_argument("--scale_by_tof_depth_type",default=0,type=int)
        self.parser.add_argument("--adpative_scale_type",default=0,type=int)
        self.parser.add_argument("--global_scale_type",default=0,type=int)
        self.parser.add_argument("--global_scale_pose_type",default=0,type=int)
        self.parser.add_argument("--fuse_scale_type",default=0,type=int)
        self.parser.add_argument("--guided_loss_type",default=0,type=int)
        self.parser.add_argument("--weighted_ls_type",default=0,type=int)
        self.parser.add_argument("--zone_boundary_smoothness_weight",default=0,type=float)
        self.parser.add_argument("--dilate_zone_boundary_rate",default=0,type=int)
        self.parser.add_argument("--num_scale_banks",default=4,type=int)
        self.parser.add_argument("--flow_type",default=0,type=int)
        self.parser.add_argument("--sl_type",default=0,type=int)
        self.parser.add_argument("--eval_do_save",action="store_true")
        self.parser.add_argument("--smoothness_scale_weight",default=0,type=float)
        self.parser.add_argument("--smoothness_scale_zone_boundary_weight",default=0,type=float)
        self.parser.add_argument("--reg_scale_map_weight",default=0,type=float)
        self.parser.add_argument("--drop_hist",default=0.34,type=float)
        self.parser.add_argument("--noise_mean",default=0.17,type=float)
        self.parser.add_argument("--noise_sigma",default=0.20,type=float)
        self.parser.add_argument("--sl_loss_weight",default=0.0,type=float)
        self.parser.add_argument("--noise_prob",default=0.30,type=float)
        self.parser.add_argument("--noise_prob",default=0.30,type=float)
        self.parser.add_argument("--eval_max_depth",default=10.0,type=float)
        self.parser.add_argument("--eval_min_depth",default=0.01,type=float)
        self.parser.add_argument("--detach_scale",action="store_true")
        self.parser.add_argument("--eval_in_zone",action="store_true")
        self.parser.add_argument("--eval_out_zone",action="store_true")
        self.parser.add_argument("--encoder_type",default=0,type=int)
        self.parser.add_argument("--conv_block_type",default=0,type=int)
        self.parser.add_argument("--addition_type",default=0,type=int)
        self.parser.add_argument("--drop_rate",default=0.0,type=float)



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

