# Self-Supervised Enhancement for Depth from a Lightweight ToF Sensor with Monocular Images 
###  [Paper]

The paper has been accepted to IROS 2025. The arxiv version of the paper is at [here](https://arxiv.org/abs/2506.13444).


### Installation
```bash
pip install -r requirements.txt
```
### Prepare the data and pretrained model
Please refer to [PLNet](https://github.com/HalleyJiang/PLNet) for the data preparation on NYU and ScanNet datasets.

Please download the pretrained model from [Baidu Yun](https://pan.baidu.com/s/1IP8x9esWXY2WyhpNtudQbA) (password: bs61) and put it in the correct directory.

The resulting structure should look sth like this:
```
selftof
├── data
│   ├── nyuv2_test
│   └── demo
│   └── nyu2_train
└── tmp
│   ├── mono_drop_0.2_rgbtof_tp_256_L2_2_0.01_scale_3_enc_2_add_9_9
│       └── models
│           ├── Epoch 39
│           │   └── encoder.pth
```

### Simulating ToF missing
```
python generate_fixed_mask.py
```
The reproducibility is by the random number set in the generate_fixed_mask.py file.
### Command to train on the NYU dataset
```bash
python train.py @./exps/mono_rgbtof_tp_256_L2_2_0.01_scale_3.txt --port 16001
```


### Command to evaluate on the NYU dataset
```bash
python evaluate_all_nyu_depth.py @./exps/mono_drop_0.2_rgbtof_tp_256_L2_2_0.01_scale_3_enc_2_add_9_9.txt --vis_epoch 39 --eval_min_depth 0.01 --eval_max_depth 10.0 --disable_median_scaling --eval_do_save
python evaluate_all_nyu_depth.py @./exps/mono_drop_0.2_rgbtof_tp_256_L2_2_0.01_scale_3_enc_2_add_9_9.txt --eval_min_depth 0.01 --eval_max_depth 10.0 --disable_median_scaling 
```

Note that we train on the NYU dataset and evaluate on both the NYU and ScanNet datasets. The model that perform the best
on NYU dataset will be chosen to evaluate on both NYU and ScanNet datasets.
If you do not set the selected_epoch, the code will go through all available epochs and generate an excel file that contains the result for all epochs.



## Citation

If you find this code useful for your research, please use the following BibTeX entry. 

```bash
@misc{ding2025selfsupervised,
    title={Self-Supervised Enhancement for Depth from a Lightweight ToF Sensor with Monocular Images},
    author={Laiyan Ding and Hualie Jiang and Jiwei Chen and Rui Huang},
    year={2025},
    eprint={2506.13444},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements

We would like to thank the authors of [DELTAR](https://github.com/zju3dv/deltar), [PLNet](https://github.com/HalleyJiang/PLNet) for open-sourcing their projects.
