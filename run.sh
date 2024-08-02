#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
###

modelname="mono_drop_20_oracle_rgbd_pose_rgbd_enc_6_add_17_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8"
python train.py @./exps/$modelname.txt --port 16001
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_in_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_out_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --vis_epoch 39 --eval_do_save
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001
#
modelname="mono_drop_20_oracle_rgbd_pose_rgbd_enc_6_add_18_ep_40_sm_0.1_area_mean_area_L2_weighted_12_0.01_scale_depth_8"
python train.py @./exps/$modelname.txt --port 16001
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_in_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_out_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --vis_epoch 39 --eval_do_save
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001
