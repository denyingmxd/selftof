from __future__ import absolute_import, division, print_function

import os, sys

sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import torch.nn.functional as F
import datasets
import networks

from tqdm import tqdm
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions

import cv2
from PIL import Image, ImageDraw
cv2.setNumThreads(0)
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

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    lg10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, lg10, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def prepare_model_for_test(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(encoder_dict, "module.")
    if opt.rgbd_encoder:
        encoder = networks.RGBD_Encoder(opt.num_layers, False, args=opt)
    else:
        encoder = networks.ResnetEncoder(opt.num_layers, False)

    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    decoder_dict = torch.load(decoder_path, map_location=torch.device('cpu'))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(decoder_dict, "module.")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, [0],
                                          PixelCoorModu=not opt.disable_pixel_coordinate_modulation,
                                          depth_repr=opt.depth_repr, args=opt)

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})

    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})

    encoder.cuda().eval()
    depth_decoder.cuda().eval()

    return encoder, depth_decoder, encoder_dict['height'], encoder_dict['width']


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    encoder, depth_decoder, thisH, thisW = prepare_model_for_test(opt)

    filenames = readlines('./splits/scannet_test_depth.txt')
    dataset = datasets.ScannetTestDepthDataset(
        opt.data_path,
        filenames,
        thisH, thisW,
        opt
    )
    dataloader = DataLoader(
        dataset, 1, shuffle=False,
        num_workers=4,pin_memory=True)

    print("-> Computing predictions with size {}x{}".format(thisH, thisW))
    print("-> Evaluating")



    gt_depths = []
    pred_disps = []
    colors = np.zeros((len(filenames),3, thisH, thisW))
    rect_data = []
    rect_mask = []
    hist_data = []
    tof_mask = []
    tof_depths = np.zeros((len(filenames),1, thisH, thisW))
    print("-> Computing predictions with size {}x{}".format(thisH, thisW))

    with torch.no_grad():
        for idx,data in enumerate(tqdm(dataloader)):
            input_color = data[("color", 0, 0)].cuda()
            norm_pix_coords = [data[("norm_pix_coords", s)].cuda() for s in opt.scales]

            gt_depth = data["depth_gt"][:, 0].numpy()
            gt_depths.append(gt_depth)

            if opt.vis_epoch is not None and opt.eval_do_save:
                colors[idx] = data[("color", 0, 0)][0].numpy()
                # colors_full[idx] = data[("color", 0, -1)][0].numpy()
            if 'tof' in opt.sparse_d_type:
                data[('tof_depth',0)] = data[('tof_depth',0)].cuda()
                data[('additional',0)]['mask'] = data[('additional',0)]['mask']

                rect_data.append(data[('additional',0)]['rect_data'])
                rect_mask.append(data[('additional',0)]['mask'])
                hist_data.append(data[('additional',0)]['hist_data'])
                tof_depths[idx] = data[('tof_depth',0)][0].detach().cpu().numpy()
                tof_mask_full = F.interpolate(data[('additional',0)]['tof_mask'].float(), size=gt_depth.shape[1:], mode='nearest').bool()
                tof_mask.append(tof_mask_full)


            # rgb_features,tof_features = encoder(input_color,data)
            # output = depth_decoder(rgb_features, norm_pix_coords, data, opt, tof_features)
            #
            # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            # pred_disp = pred_disp.cpu()[:, 0].numpy()

            # nearest neighbor
            pred_disp = 1./(data[('tof_depth',0)][0].detach().cpu().numpy()+1e-7)

            # # # #GF
            from some_codes.GF import upscale_depth
            pred_disp = upscale_depth(pred_disp[0], data[("color", 0, 0)][0].numpy().transpose(1,2,0))[0][None,::]


            pred_disps.append(pred_disp)

    gt_depths = np.concatenate(gt_depths)
    pred_disps = np.concatenate(pred_disps)
    if 'tof' in opt.sparse_d_type:
        tof_mask = np.concatenate(tof_mask)

    errors = []
    ratios = []


    for i in tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp



        mask = np.logical_and(gt_depth > opt.min_depth, gt_depth < opt.max_depth)

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

        mask_pred_depth[mask_pred_depth < opt.min_depth] = opt.min_depth
        mask_pred_depth[mask_pred_depth > opt.max_depth] = opt.max_depth

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

            if not os.path.exists(os.path.join(opt.load_weights_folder, 'scannet_depth_'+opt.model_name,impath)):
                os.makedirs(os.path.join(opt.load_weights_folder, 'scannet_depth_'+opt.model_name,impath))
            rgb_image.save(os.path.join(opt.load_weights_folder, 'scannet_depth_'+opt.model_name, f"{impath}_rgb.jpg"))
            pred_depth_image.save(os.path.join(opt.load_weights_folder,'scannet_depth_'+ opt.model_name , f"{impath}_pred_depth.jpg"))
            gt_depth_image.save(os.path.join(opt.load_weights_folder,'scannet_depth_'+ opt.model_name , f"{impath}_gt_depth.jpg"))
            tof_depth_image.save(os.path.join(opt.load_weights_folder,'scannet_depth_'+ opt.model_name , f"{impath}_tof_depth.jpg"))
            error_image.save(os.path.join(opt.load_weights_folder, 'scannet_depth_'+ opt.model_name, f"{impath}_error.jpg"))


    mean_errors = np.array(errors).mean(0)
    result_path = os.path.join(opt.load_weights_folder, "result_scannet_depth.txt")
    f = open(result_path, 'w+')

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)), file=f)

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 8).format(*mean_errors.tolist()) + "\\\\")

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"), file=f)
    print(("&{: 8.4f}  " * 8).format(*mean_errors.tolist()) + "\\\\", file=f)

    print("\n-> Done!")
    print("\n-> Done!", file=f)

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
        ext='scannet_depth_no_median'
    else:
        ext='scannet_depth_median'
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

