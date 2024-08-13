# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os.path
import random
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torch.distributed as dist
import json

from utils import *
from layers import *
from PIL import ImageDraw
import datasets
import networks
import random

# from IPython import embed

def seed_all(rank):
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class Trainer_DDP:
    def __init__(self, gpu, ngpus_per_node, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.opt.gpu = gpu
        self.opt.multigpu = True
        self.opt.rank = self.opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=self.opt.dist_backend, init_method=self.opt.dist_url,
                                world_size=self.opt.world_size, rank=self.opt.rank)
        seed_all(self.opt.rank)
        self.opt.batch_size = int(self.opt.batch_size / ngpus_per_node)
        self.opt.num_workers = int(self.opt.num_workers / ngpus_per_node)

        print(self.opt.gpu, self.opt.rank, self.opt.batch_size, self.opt.num_workers)
        torch.cuda.set_device(self.opt.gpu)

        find_unused_parameters = False
        if find_unused_parameters:
            torch.autograd.set_detect_anomaly(True)
        if self.opt.rgbd_encoder:
            self.models["encoder"] = networks.RGBD_Encoder(self.opt.num_layers, self.opt.weights_init == "pretrained",args=self.opt)
        else:
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"] = nn.SyncBatchNorm.convert_sync_batchnorm( self.models["encoder"])
        self.models["encoder"].to(self.device)
        self.models["encoder"] = torch.nn.parallel.DistributedDataParallel(self.models["encoder"], device_ids=[self.opt.gpu],
                                                                output_device=self.opt.gpu,
                                                                find_unused_parameters=find_unused_parameters,
                                                                           broadcast_buffers=False)
        self.parameters_to_train += list(self.models["encoder"].parameters())



        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].module.num_ch_enc, self.opt.scales,
                                                     PixelCoorModu=not self.opt.disable_pixel_coordinate_modulation,
                                                     depth_repr=self.opt.depth_repr,args=self.opt)
        self.models["depth"] = nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth"])
        self.models["depth"].to(self.device)
        self.models["depth"] = torch.nn.parallel.DistributedDataParallel(self.models["depth"], device_ids=[self.opt.gpu],
                                                                output_device=self.opt.gpu,
                                                                find_unused_parameters=find_unused_parameters,
                                                                         broadcast_buffers=False)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.opt.rgbd_pose_encoder:
            self.models["pose_encoder"] = networks.RGBD_Pose_Encoder(self.opt.num_layers,
                                                                self.opt.weights_init == "pretrained",
                                                                num_input_images=self.num_pose_frames,args=self.opt)
        else:
            self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers,
                                                                 self.opt.weights_init == "pretrained",
                                                                 num_input_images=self.num_pose_frames)

        self.models["pose_encoder"] = nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_encoder"])
        self.models["pose_encoder"].to(self.device)
        self.models["pose_encoder"] = torch.nn.parallel.DistributedDataParallel(self.models["pose_encoder"], device_ids=[self.opt.gpu],
                                                                output_device=self.opt.gpu,
                                                                find_unused_parameters=find_unused_parameters,
                                                                                broadcast_buffers=False)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].module.num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames - 1))
        self.models["pose"] = nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose"])
        self.models["pose"].to(self.device)
        self.models["pose"] = torch.nn.parallel.DistributedDataParallel(self.models["pose"], device_ids=[self.opt.gpu],
                                                                output_device=self.opt.gpu,
                                                                find_unused_parameters=find_unused_parameters,
                                                                        broadcast_buffers=False)
        self.parameters_to_train += list(self.models["pose"].parameters())


        self.opt.should_log = (self.opt.rank == 0)
        should_log = self.opt.should_log


        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        if should_log:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
            print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"nyu": datasets.NYUDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext,
            return_plane=not self.opt.disable_plane_regularization,
            num_plane_keysets=self.opt.num_plane_keysets,
            return_line=not self.opt.disable_line_regularization,
            num_line_keysets=self.opt.num_line_keysets,opt=self.opt)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=(train_sampler is None),
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,sampler=train_sampler)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext,
            return_plane=not self.opt.disable_plane_regularization,
            num_plane_keysets=self.opt.num_plane_keysets,
            return_line=not self.opt.disable_line_regularization,
            num_line_keysets=self.opt.num_line_keysets,opt=self.opt)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        if should_log:
            for mode in ["train", "val"]:
                if self.opt.model_name=='debug':
                    #remove existing log files
                    import shutil
                    if os.path.exists(os.path.join(self.log_path, mode)):
                        shutil.rmtree(os.path.join(self.log_path, mode))
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        # self.project_homo = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "de/log10", "da/a1", "da/a2", "da/a3"]

        if should_log:
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))

            self.save_opts()
        dist.barrier()
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        # self.val()
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.opt.should_log:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        run_step = 0
        loss_sum = 0.0
        full_times = []
        old_time=time.time()
        op_times = []
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader),
                                                 desc=f"Epoch: {self.epoch}/{self.opt.num_epochs}. Loop: Train",
                                                 total=len(self.train_loader)) if self.opt.should_log \
                                            else enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            op_times.append(duration)
            full_times.append(time.time() - old_time)
            old_time = time.time()
            run_step += 1
            loss_sum += losses["loss"].cpu().data
            if self.opt.should_log:
                # log less frequently after the first 2000 steps to save time & disk space
                early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
                late_phase = self.step % 3000 == 0
                print(np.mean(full_times),np.mean(op_times))
                if early_phase or late_phase:
                    # self.log_time(batch_idx, duration, loss_sum / run_step)

                    if "depth_gt" in inputs:
                        self.compute_depth_losses(inputs, outputs, losses)

                    self.log("train", inputs, outputs, losses)
            self.step += 1
            dist.barrier()
        self.model_lr_scheduler.step()

        # self.val()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if 'additional' not in key:
                inputs[key] = ipt.to(self.device)
            else:
                for k, v in ipt.items():
                    inputs[key][k] = v.to(self.device)

        norm_pix_coords = [inputs[("norm_pix_coords", s)] for s in self.opt.scales]

        rgb_features,tof_features = self.models["encoder"](inputs["color_aug", 0, 0],inputs)
        outputs = self.models["depth"](rgb_features, norm_pix_coords,inputs,self.opt,tof_features)


        outputs.update(self.predict_poses(inputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            half_source_frames = len(self.opt.frame_ids[1:]) // 2

            negative_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames:0:-1]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[negative_half[i + 1]], pose_feats[negative_half[i]]]
                if self.opt.rgbd_pose_encoder:
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1),inputs,[negative_half[i + 1],negative_half[i]])]
                else:
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))[0]]
                axisangle, translation = self.models["pose"](pose_inputs)

                outputs[("axisangle", negative_half[i + 1], negative_half[i])] = axisangle
                outputs[("translation", negative_half[i + 1], negative_half[i])] = translation

                # Invert the matrix if the frame id is negative
                if i == 0:
                    outputs[("cam_T_cam", 0, negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                else:
                    outputs[("cam_T_cam", negative_half[i], negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                    outputs[("cam_T_cam", 0, negative_half[i + 1])] = \
                        outputs[("cam_T_cam", 0, negative_half[i])] @ \
                        outputs[("cam_T_cam", negative_half[i], negative_half[i + 1])]

            positive_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames + 1:]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[positive_half[i]], pose_feats[positive_half[i + 1]]]
                if self.opt.rgbd_pose_encoder:
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1), inputs, [positive_half[i + 1], positive_half[i]])]
                else:
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))[0]]
                axisangle, translation = self.models["pose"](pose_inputs)

                outputs[("axisangle", positive_half[i], positive_half[i + 1])] = axisangle
                outputs[("translation", positive_half[i], positive_half[i + 1])] = translation

                if i == 0:
                    outputs[("cam_T_cam", 0, positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                else:
                    outputs[("cam_T_cam", positive_half[i], positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                    outputs[("cam_T_cam", 0, positive_half[i + 1])] = \
                        outputs[("cam_T_cam", 0, positive_half[i])] @ \
                        outputs[("cam_T_cam", positive_half[i], positive_half[i + 1])]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 1)

            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                outputs[("axisangle", 0, f_i)] = axisangle[:, i:i + 1]
                outputs[("translation", 0, f_i)] = translation[:, i:i + 1]
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

        return outputs


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            outputs[("cam_points", 0, scale)] = self.backproject_depth[scale](
                depth, inputs[("norm_pix_coords", scale)])

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                pix_coords = self.project_3d[scale](
                    outputs[("cam_points", 0, scale)], inputs[("K", scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords.permute(0, 2, 3, 1)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

        if self.opt.sparse_d_type == 'tof':
            depth = inputs['additional',0]['tof_mean']
            outputs[("cam_points_tof", 0, scale)] = self.backproject_depth[scale](
                depth, inputs[("norm_pix_coords", scale)])

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                pix_coords = self.project_3d[scale](
                    outputs[("cam_points_tof", 0, scale)], inputs[("K", scale)], T)

                outputs[("sample_tof", frame_id, scale)] = pix_coords.permute(0, 2, 3, 1)

                outputs[("color_tof", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, scale)],
                    outputs[("sample_tof", frame_id, scale)],
                    padding_mode="border", align_corners=True)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                outputs[("reprojection_losses", frame_id, scale)] = self.compute_reprojection_loss(pred, target)

                loss += outputs[("reprojection_losses", frame_id, scale)].mean()

                if self.opt.sparse_d_type == 'tof':
                    pred = outputs[("color_tof", frame_id, scale)]
                    outputs[("reprojection_losses_tof", frame_id, scale)] = self.compute_reprojection_loss(pred, target)


            if self.opt.disable_plane_smoothness:
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
            else:
                mean_coeff = outputs[("coeff", scale)].abs().mean(2, True).mean(3, True)

                norm_coeff = outputs[("coeff", scale)] / (mean_coeff + 1e-7)
                smooth_loss = get_smooth_loss(norm_coeff, color)

            loss += self.opt.smoothness_weight / (2 ** scale) * smooth_loss
            losses["smooth_loss/{}".format(scale)] = smooth_loss

            point3D = outputs[("cam_points", 0, scale)][:, :3, ...]
            mean_depth = outputs[("depth", 0, scale)].mean(2, True).mean(3)
            norm_point3D = point3D / (mean_depth + 1e-7)

            if not self.opt.disable_plane_regularization:
                plane_loss = get_plane_loss(inputs[("plane_keysets", 0, scale)], norm_point3D)
                loss += self.opt.plane_weight * plane_loss
                losses["plane_loss/{}".format(scale)] = plane_loss

            if not self.opt.disable_line_regularization:
                line_loss = get_line_loss(inputs[("line_keysets", 0, scale)], norm_point3D)
                loss += self.opt.line_weight * line_loss
                losses["line_loss/{}".format(scale)] = line_loss

            if self.opt.sparse_depth_loss_weight>0 and 'tof' in self.opt.sparse_d_type:
                mean_loss,std_loss,sparse_depth_loss = compute_sparse_depth_loss(inputs['additional',0]['tof_mean'], outputs[("depth", 0, scale)], inputs,outputs,self.opt)
                loss += self.opt.sparse_depth_loss_weight * sparse_depth_loss

                losses["sparse_depth_loss/{}".format(scale)] = sparse_depth_loss
                losses["mean_loss/{}".format(scale)] = mean_loss
                losses["std_loss/{}".format(scale)] = std_loss



            losses["loss/{}".format(scale)] = loss
            total_loss += loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [self.dataset.full_res_shape[1], self.dataset.full_res_shape[0]],
            mode="bilinear", align_corners=False), self.dataset.min_depth, self.dataset.max_depth)

        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, self.dataset.default_crop[2]:self.dataset.default_crop[3], \
        self.dataset.default_crop[0]:self.dataset.default_crop[1]] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=self.dataset.min_depth, max=self.dataset.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if ('global_ratio',0) in outputs.keys():
            writer.add_scalar("global_ratio", outputs[('global_ratio',0)][0][0][0][0], self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images

            writer.add_image(
                "gt_disp_0/{}".format(j),
                normalize_image(1 / (inputs["depth_gt"][j] + 0.01)), self.step)

            writer.add_image(
                "gt_depth_0/{}".format(j),
                normalize_image(inputs["depth_gt"][j]), self.step)


            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    rgb_image = inputs[("color", frame_id, s)][j].data
                    if self.opt.sparse_d_type == 'tof' and frame_id == 0:
                        rgb_image = (rgb_image.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                        rgb_image = Image.fromarray(rgb_image)
                        rect_data = inputs[('additional',0)]['rect_data'][j]
                        rect_mask = inputs[('additional',0)]['mask'][j]
                        draw = ImageDraw.Draw(rgb_image)
                        for abc, xxx in enumerate(rect_data[rect_mask == 1]):
                            a, b, c, d = xxx
                            draw.rectangle((b, a, d, c), outline=(255, 0, 0))
                        for abc, xxx in enumerate(rect_data[rect_mask == 0]):
                            a, b, c, d = xxx
                            draw.rectangle((b, a, d, c), outline=(0, 0, 0))
                        rgb_image = np.array(rgb_image).transpose(2, 0, 1)
                        rgb_image = torch.from_numpy(rgb_image).float() / 255



                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        rgb_image, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                        writer.add_image(
                            "reprojection_losses_{}_{}/{}".format(frame_id, s, j),
                            outputs[("reprojection_losses", frame_id, s)][j].data, self.step)
                        if self.opt.sparse_d_type == 'tof':

                            writer.add_image(
                                "reprojection_losses_tof_{}_{}/{}".format(frame_id, s, j),
                                outputs[("reprojection_losses_tof", frame_id, s)][j].data*inputs[('additional',0)]['my_mask'][j].data, self.step)
                            writer.add_image(
                                "reprojection_losses_cropped_{}_{}/{}".format(frame_id, s, j),
                                outputs[("reprojection_losses", frame_id, s)][j].data*inputs[('additional',0)]['my_mask'][j].data, self.step)

                writer.add_image(
                    "pred_disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                writer.add_image(
                    "pred_depth_{}/{}".format(s, j),
                    normalize_image(outputs[("depth", 0, s)][j]), self.step)

                if self.opt.sparse_d_type =='tof':
                    tof_mean = inputs['additional',0]['tof_mean'][j: j + 1].mean(1,True)
                    tof_mean = torch.clamp(F.interpolate(
                        tof_mean, [self.dataset.full_res_shape[1], self.dataset.full_res_shape[0]],
                        mode="nearest"), self.dataset.min_depth, self.dataset.max_depth)[0]
                    writer.add_image(
                        "tof_mean_{}/{}".format(s, j),
                        normalize_image(tof_mean), self.step)


                if  ('valid_area',0) in outputs.keys():
                    writer.add_image(
                        "pred_tof_seg_{}/{}".format(s, j),
                        outputs[('valid_area',0)][j], self.step)

                if ('scale_map',0) in outputs.keys():
                    writer.add_image(
                        "scale_map_{}/{}".format(s, j),
                        outputs[('scale_map',0)][j], self.step)


                if ('valid_con_mask',0) in outputs.keys():
                    writer.add_image(
                        "valid_con_mask_{}/{}".format(s, j),
                        outputs[('valid_con_mask',0)][j].float(), self.step)

                if ('ls_weight_map',0) in outputs.keys():
                    writer.add_image(
                        "ls_weight_map_{}/{}".format(s, j),
                        outputs[('ls_weight_map',0)][j], self.step)

                xx = inputs["depth_gt"][j]
                yy =  torch.clamp(F.interpolate(
                    outputs[("depth", 0, s)], [self.dataset.full_res_shape[1], self.dataset.full_res_shape[0]],
                    mode="bilinear", align_corners=False), self.dataset.min_depth, self.dataset.max_depth)[j]

                yy = torch.median(xx)/torch.median(yy)*yy
                writer.add_image(
                    "error_depth_{}/{}".format(s, j),
                    normalize_image(torch.abs(xx-yy)), self.step)



    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.module.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].module.state_dict()
            pretrained_dict = torch.load(path)
            # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(pretrained_dict, "module.")
            # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_dict, "module.")

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].module.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

#seed all for ddp but keep the speed


def train_ddp(gpu, ngpus_per_node, args):
    trainer = Trainer_DDP(gpu, ngpus_per_node, args)
    trainer.train()
    dist.destroy_process_group()