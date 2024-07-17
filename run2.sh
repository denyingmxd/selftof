#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
###

modelname="mono_drop_20_oracle_rgbd_pose_rgbd_enc_7_add_4_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8"
python train.py @./exps/$modelname.txt --port 16002
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002

modelname="mono_drop_20_oracle_rgbd_pose_rgbd_enc_7_add_3_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8"
python train.py @./exps/$modelname.txt --port 16002
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002