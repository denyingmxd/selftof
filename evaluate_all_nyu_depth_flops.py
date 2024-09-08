from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from PIL import Image
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import tqdm
import matplotlib
from thop import profile
import time
import  matplotlib.pyplot as plt
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def linear_model(x, a, b):
    return a * x + b
def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    log10 = np.mean(np.abs(np.log10(pred / gt)))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = opt.eval_min_depth
    MAX_DEPTH = opt.eval_max_depth

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path,map_location=torch.device('cpu'))
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(encoder_dict, "module.")


        dataset = datasets.NYUDataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
                                      [0], 1, is_test=True, return_plane=False, num_plane_keysets=0,
                                      return_line=False, num_line_keysets=0,opt=opt)

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=4,
                                pin_memory=True, drop_last=False)


        if opt.rgbd_encoder:
            encoder = networks.RGBD_Encoder(opt.num_layers, False,args=opt)
        else:
            encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, [0],
                                                     PixelCoorModu=not opt.disable_pixel_coordinate_modulation,
                                                     depth_repr=opt.depth_repr,args=opt)
        # self.models["depth"] = networks.DepthDecoder(self.models["encoder"].module.num_ch_enc, self.opt.scales,
        #                                              PixelCoorModu=not self.opt.disable_pixel_coordinate_modulation,
        #                                              depth_repr=self.opt.depth_repr,args=self.opt)
        model_dict = encoder.state_dict()

        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        model_dict = depth_decoder.state_dict()
        decoder_dict = torch.load(decoder_path,map_location=torch.device('cpu'))
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(decoder_dict, "module.")
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in model_dict})

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        gt_depths = []
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(encoder_dict['width'], encoder_dict['height']))


        flops = []
        params = []

        latencies = []
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)


        with torch.no_grad():
            for idx,data in enumerate(tqdm.tqdm(dataloader)):
                input_color = data[("color", 0, 0)].cuda()
                norm_pix_coords = [data[("norm_pix_coords", s)].cuda() for s in opt.scales]

                gt_depth = data["depth_gt"][:, 0].numpy()
                gt_depths.append(gt_depth)


                if 'tof' in opt.sparse_d_type:
                    data[('tof_depth',0)] = data[('tof_depth',0)].cuda()
                    data[('additional',0)]['hist_data'] = data[('additional',0)]['hist_data']
                    data[('additional',0)]['rect_data'] = data[('additional',0)]['rect_data']
                    data[('additional',0)]['mask'] = data[('additional',0)]['mask']

                break

            for j in tqdm.tqdm(range(1000)):
                # torch.cuda.synchronize()
                start_time.record()
                rgb_features, tof_features = encoder(input_color, data)
                output = depth_decoder(rgb_features, norm_pix_coords, data, opt, tof_features)
                end_time.record()
                torch.cuda.synchronize()

                latencies.append(start_time.elapsed_time(end_time))

            encoder_flops, encoder_params = profile(encoder, inputs=(input_color, data))
            rgb_features, tof_features = encoder(input_color, data)
            decoder_flops, decoder_params = profile(depth_decoder, inputs=(rgb_features, norm_pix_coords, data, opt, tof_features))
            flops.append(encoder_flops + decoder_flops)
            params.append(encoder_params + decoder_params)
            # break


        actual_latency = np.mean(latencies)
        #print latency and fps with 3 decimal points
        print('Latency: {:.3f} ms'.format(actual_latency))
        print('FPS: {:.3f}'.format(1000/actual_latency))
        from thop import clever_format
        macs, params = clever_format([np.mean(encoder_flops), np.mean(params)], "%.3f")
        print("MACs: ", macs)
        print("Params: ", params)
        print('example macs:',flops[:10])
        print('example params:',params[:10])




    return


if __name__ == "__main__":
    options = MonodepthOptions()
    options = options.parse()
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Epoch","abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3",'ratio'])
    if options.vis_epoch is not None:
        epochs = [options.vis_epoch]
    else:
        epochs = range(options.num_epochs)
        # epochs = [39]
    for i in epochs:
        options.load_weights_folder = os.path.join(options.load_weights_root,  "weights_{}".format(i))
        if not os.path.exists(options.load_weights_folder):
            continue
        assert options.model_name in options.load_weights_folder
        result = [i]
        evaluate(options)
