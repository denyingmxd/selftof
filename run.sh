#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
###

modelname="mono_naive_oracle_rgbd_pose_rgbd_enc_7_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8"
python train.py @./exps/$modelname.txt --port 16001
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001