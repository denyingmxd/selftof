#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
###

modelname="mono_rgbtof_tp_256_L2_2_0.01_scale_3_enc_1_add_9_9"
python train.py @./exps/$modelname.txt --port 16002
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling --eval_in_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling --eval_out_zone
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling
python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002

#modelname="mono_drop_0.4_rgbtof_tp_256_L2_2_0.01_scale_3_enc_2_add_9_9"
#python train.py @./exps/$modelname.txt --port 16002
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling --eval_in_zone
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling --eval_out_zone
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002 --disable_median_scaling
#python evaluate_all_nyu_depth.py @./exps/$modelname.txt --port 16002
#
