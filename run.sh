#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
###

modelname="mono_drop_0.2_rgbtof_tp_256_L2_2_0.01_scale_3_add_1"
python train.py @./exps/$modelname.txt --port 16001
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_in_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_out_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001

modelname="mono_drop_0.2_rgbtof_tp_256_L2_2_0.01_scale_3_add_2"
python train.py @./exps/$modelname.txt --port 16001
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_in_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling --eval_out_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16001
