#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
###

modelname="mono_rgbtof_256_gt_supervision"
python train.py @./exps/$modelname.txt --port 16001
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_in_zone
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_out_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001

#modelname="mono_rgbtof_tp_256_L2_2_0.01_scale_3_enc_1_add_9_9"
#python train.py @./exps/$modelname.txt --port 16001
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_in_zone
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_out_zone
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001
