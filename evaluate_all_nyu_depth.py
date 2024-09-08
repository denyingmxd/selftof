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
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import math
from einops import rearrange
from scipy.optimize import curve_fit

import torch.nn.functional as F

import  matplotlib.pyplot as plt
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


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
        planes = []
        lines = []
        pred_disps = []
        colors = np.zeros((len(filenames),3,encoder_dict['height'], encoder_dict['width']))
        # colors_full = np.zeros((len(filenames),3,448, 608))
        rect_data = []
        rect_mask = []
        hist_data = []
        tof_mask = []
        tof_depths = np.zeros((len(filenames),1, encoder_dict['height'], encoder_dict['width']))
        print("-> Computing predictions with size {}x{}".format(encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for idx,data in enumerate(tqdm.tqdm(dataloader)):
                input_color = data[("color", 0, 0)].cuda()
                norm_pix_coords = [data[("norm_pix_coords", s)].cuda() for s in opt.scales]

                gt_depth = data["depth_gt"][:, 0].numpy()
                gt_depths.append(gt_depth)

                if opt.vis_epoch is not None and opt.eval_do_save:
                    colors[idx] = data[("color", 0, 0)][0].numpy()
                    # colors_full[idx] = data[("color", 0, -1)][0].numpy()
                if 'tof' in opt.sparse_d_type:
                    data[('tof_depth',0)] = data[('tof_depth',0)].cuda()
                    data[('additional',0)]['hist_data'] = data[('additional',0)]['hist_data']
                    data[('additional',0)]['rect_data'] = data[('additional',0)]['rect_data']
                    data[('additional',0)]['mask'] = data[('additional',0)]['mask']

                    rect_data.append(data[('additional',0)]['rect_data'])
                    rect_mask.append(data[('additional',0)]['mask'])
                    hist_data.append(data[('additional',0)]['hist_data'])
                    tof_depths[idx] = data[('tof_depth',0)][0].detach().cpu().numpy()
                    tof_mask_full = F.interpolate(data[('additional',0)]['tof_mask'].float(), size=gt_depth.shape[1:], mode='nearest').bool()
                    # folder, frame_index = filenames[idx].split()
                    # tof_mask_full = np.load(os.path.join(opt.data_path, folder, "{:05d}".format(int(frame_index)) + "_fixed_mask_drop_{}.npz").format(str(0.2)))['fixed_mask']
                    # tof_mask_full = torch.from_numpy(tof_mask_full.reshape(8,8)).float()
                    # tof_mask_full = F.interpolate(tof_mask_full.unsqueeze(0).unsqueeze(0), size=gt_depth.shape[1:], mode='nearest').bool()
                    tof_mask.append(tof_mask_full)
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                    norm_pix_coords = [torch.cat((pc, torch.flip(pc, [3])), 0) for pc in norm_pix_coords]
                    norm_pix_coords[0][norm_pix_coords[0].shape[0] // 2:, 0] *= -1

                rgb_features,tof_features = encoder(input_color,data)
                output = depth_decoder(rgb_features, norm_pix_coords, data, opt, tof_features)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                #nearest neighbor
                # pred_disp = 1./data[('tof_depth',0)][0].detach().cpu().numpy()
                # # # #GF
                # from some_codes.GF import upscale_depth
                # pred_disp = upscale_depth(pred_disp[0], data[("color", 0, 0)][0].numpy().transpose(1,2,0))[0][None,::]



                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                pred_disps.append(pred_disp)

                # break

        gt_depths = np.concatenate(gt_depths)
        pred_disps = np.concatenate(pred_disps)
        if 'tof' in opt.sparse_d_type:
            rect_data = np.concatenate(rect_data)
            rect_mask = np.concatenate(rect_mask)
            hist_data = np.concatenate(hist_data)
            tof_mask = np.concatenate(tof_mask)



    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    print("-> Evaluating")
    if not opt.disable_median_scaling:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []


    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp



        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        if 'tof' in opt.sparse_d_type:
            tof_mask_local = tof_mask[i][0]
            if opt.eval_in_zone:
                mask = np.logical_and(mask, tof_mask_local)
            elif opt.eval_out_zone:
                mask = np.logical_and(mask, ~tof_mask_local)
            else:
                pass
        else:
            pass

        crop_mask = np.zeros(mask.shape)
        crop_mask[dataset.default_crop[2]:dataset.default_crop[3], \
                    dataset.default_crop[0]:dataset.default_crop[1]] = 1
        mask = np.logical_and(mask, crop_mask)
        if mask.sum()==0:
            continue

        mask_pred_depth = pred_depth[mask]
        mask_gt_depth = gt_depth[mask]

        mask_pred_depth *= opt.pred_depth_scale_factor


        if not opt.disable_median_scaling:
            ratio = np.median(mask_gt_depth) / np.median(mask_pred_depth)
            ratios.append(ratio)
            mask_pred_depth *= ratio
        else:
            ratio = 1
            ratios.append(ratio)

        mask_pred_depth[mask_pred_depth < MIN_DEPTH] = MIN_DEPTH
        mask_pred_depth[mask_pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(mask_gt_depth, mask_pred_depth))

        if opt.vis_epoch is not None and opt.eval_do_save:
            rgb = colors[i]
            rgb = (rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            rgb_image = Image.fromarray(rgb)

            # if 'tof' in opt.sparse_d_type:
            #     draw = ImageDraw.Draw(rgb_image)
            #     for j, xxx in enumerate(rect_data[i][rect_mask[i] == 1]):
            #         a, b, c, d = xxx
            #         draw.rectangle((b, a, d, c), outline=(255, 0, 0))
            #     for j, xxx in enumerate(rect_data[i][rect_mask[i] == 0]):
            #         a, b, c, d = xxx
            #         draw.rectangle((b, a, d, c), outline=(0, 0, 0))

            rgb_image = rgb_image.resize((gt_width, gt_height), Image.ANTIALIAS)

            cap_depth = np.percentile(gt_depth, 95)
            pred_depth_image = colorize(torch.from_numpy(pred_depth*ratio).unsqueeze(0), vmin=0.0, vmax=cap_depth, cmap='viridis')
            pred_depth_image = Image.fromarray(pred_depth_image)

            tof_depth = tof_depths[i]
            tof_depth_image = colorize(torch.from_numpy(tof_depth), vmin=0.0, vmax=cap_depth, cmap='viridis')
            tof_depth_image = Image.fromarray(tof_depth_image)
            tof_depth_image = tof_depth_image.resize((gt_width,gt_height),Image.NEAREST)

            gt_depth_image = colorize(torch.from_numpy(gt_depth).unsqueeze(0), vmin=0.0, vmax=cap_depth, cmap='viridis')
            gt_depth_image = Image.fromarray(gt_depth_image)

            error_image = np.abs(gt_depth - pred_depth*ratio) * mask
            error_image = colorize(torch.from_numpy(error_image).unsqueeze(0), vmin=0.0, vmax=1.2, cmap='jet')
            error_image = Image.fromarray(error_image)

            impath = filenames[i].split(' ')[-1].split('.')[0]
            if not os.path.exists(os.path.join(opt.load_weights_folder, opt.model_name)):
                os.makedirs(os.path.join(opt.load_weights_folder, opt.model_name))
            rgb_image.save(os.path.join(opt.load_weights_folder, opt.model_name, f"{impath}_rgb.jpg"))
            pred_depth_image.save(os.path.join(opt.load_weights_folder,opt.model_name , f"{impath}_pred_depth.jpg"))
            gt_depth_image.save(os.path.join(opt.load_weights_folder,opt.model_name , f"{impath}_gt_depth.jpg"))
            tof_depth_image.save(os.path.join(opt.load_weights_folder,opt.model_name , f"{impath}_tof_depth.jpg"))
            error_image.save(os.path.join(opt.load_weights_folder, opt.model_name, f"{impath}_error.jpg"))


    mean_errors = np.array(errors).mean(0)

    ratio_med = 0
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        ratio_med = med

    mean_errors = mean_errors.tolist()
    mean_errors.extend([ratio_med])
    mean_errors = np.array(mean_errors)
    print("\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3",'ratios'))
    print(("&{: 8.3f}  " * 9).format(*mean_errors.tolist()) + "\\\\")

    return mean_errors


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
        result.extend(evaluate(options))
        ws.append(result)

    if options.disable_median_scaling:
        ext='no_median'
    else:
        ext='median'
    ranges=""
    if options.eval_min_depth !=0.01 or options.eval_max_depth !=10:
        ranges += "_{}_{}".format(options.eval_min_depth,options.eval_max_depth)
    if options.eval_in_zone:
        ranges += "_in_zone"
    elif options.eval_out_zone:
        ranges += "_out_zone"
    assert options.eval_in_zone + options.eval_out_zone <=1
    if options.vis_epoch is not None:
        wb.save(os.path.join(options.load_weights_root, "evaluation_{}_{}{}.xlsx".format(options.vis_epoch,ext,ranges)))
    else:
        wb.save(os.path.join(options.load_weights_root, "evaluation_{}{}.xlsx".format(ext,ranges)))