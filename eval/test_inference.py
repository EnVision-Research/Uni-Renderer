#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from easydict import EasyDict
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
import torchvision
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from glob import glob
import lpips
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.models.embeddings import (
    TextImageProjection, 
    TextImageTimeEmbedding, 
    TextTimeEmbedding, 
    TimestepEmbedding, 
    Timesteps, 
    GaussianFourierProjection, 
    ImageProjection,
    ImageTimeEmbedding,
    ImageHintTimeEmbedding,
    PositionNet
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from dataset.blendGen import BlenderGenDataset, BlenderGenDataset_old, BlenderGenDataset_3mod, BlenderGenDataset_3mod_demo, BlenderGenDataset_3mod_demo_validation
from dataset.objaverse import ObjaverseData, ObjaverseData_overfit_5
# from utils_metrics.compute_t import compute_t
from models.controlnet import ControlNetVAENoImgResOneCtlModel, _UnetDecControlModel, UNet2DConditionModel
from models.pipeline import StableDiffusionControl2BranchFtudecPipeline
from utils_metrics.inception import InceptionV3
from utils_metrics.calc_fid import calculate_frechet_distance, extract_features_from_samples
from utils_metrics.metrics_util import SegMetric, NormalMetric, calculate_miou_per_batch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
import nvdiffrast.torch as dr
sys.path.append('/hpc2hdd/home/zchen379/sd3/intrinsic-LRM')
from src.utils.material import Material
from src.utils.mesh import Mesh, compute_tangents
from src.utils import obj, render_utils, render
np.set_printoptions(threshold=sys.maxsize)

to_tensor = transforms.ToTensor()
    
GLCTX = [None] * torch.cuda.device_count()  # 存储每个 GPU 的上下文

def initialize_extension(gpu_id):
    global GLCTX
    if GLCTX[gpu_id] is None:
        print(f"Initializing extension module renderutils_plugin on GPU {gpu_id}...")
        torch.cuda.set_device(gpu_id)
        GLCTX[gpu_id] = dr.RasterizeCudaContext()
    return GLCTX[gpu_id]
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

def convert_to_white_bg(image, write_bg=True):
    alpha = image[:, :, 3:]
    if write_bg:
        return image[:, :, :3] * alpha + 1. * (1 - alpha)
    else:
        return image[:, :, :3] * alpha
    
def normalize(image):
    return image * 2 - 1.0


def process_material_and_mask(material, mask_image, device):
    batch_size, _, height, width = mask_image.shape
    
    # Initialize metallic and roughness images
    metallic_init = torch.zeros(batch_size, height, width, 1, dtype=torch.float64, device=device)
    roughness_init = torch.zeros(batch_size, height, width, 1, dtype=torch.float64, device=device)
    
    for i in range(batch_size):
        # Assign metallic and roughness values based on the material for each batch
        metallic_num, roughness_num = material[i][0]
        
        mask = mask_image[i, 0].bool()  # Extract the mask for the current batch
        
        metallic_init[i][mask] = metallic_num
        roughness_init[i][mask] = roughness_num

    # Convert mask to the same dtype as roughness_init and adjust the range
    # depth = depth.to(dtype=roughness_init.dtype) * 2 - 1.0
    
    # Scale metallic and roughness to the range (-1, 1)
    metallic = metallic_init * 2 - 1.0
    roughness = roughness_init * 2 - 1.0
    metallic = metallic.permute(0, 3, 1, 2)
    roughness = roughness.permute(0, 3, 1, 2)

    #zero_shape = torch.zeros_like(metallic)
    # Concatenate the metallic, roughness, and depth along the last dimension
    return_img = torch.cat((metallic, metallic, roughness), dim=1)
    
    return return_img, metallic, roughness

def collate_fn(batch):
    gpu_id = torch.cuda.current_device()  # 获取当前线程的 GPU ID
    glctx = initialize_extension(gpu_id)
    batch_size = len(batch['input_view_num'])

    # input_view_num = batch[0]["input_view_num"]
    # target_view_num = batch[0]["target_view_num"]
    iter_res = [512, 512]
    iter_spp = 1
    layers = 1

    # # Initialize lists for input and target data
    input_images, input_bg, input_alphas, input_depths, input_normals, input_albedos = [], [], [], [], [], []
    input_spec_light, input_diff_light, input_spec_albedo,input_diff_albedo = [], [], [], []
    input_w2cs, input_Ks, input_camera_pos, input_c2ws = [], [], [], []
    input_env, input_materials = [], []
    input_camera_embeddings = []    # camera_embedding_list

    # target_images, target_alphas, target_depths, target_normals, target_albedos = [], [], [], [], []
    # target_spec_light, target_diff_light, target_spec_albedo, target_diff_albedo = [], [], [], []
    # target_w2cs, target_Ks, target_camera_pos = [], [], []
    # target_env, target_materials = [], []

    for i in range(batch_size):
        obj_path = batch['obj_path'][i]

        mtl_path = os.path.splitext(batch['obj_path'][i])[0] + '.mtl'
        # try:
        with torch.no_grad():
            mesh_attributes = torch.load(obj_path, map_location=torch.device('cpu'))
            v_pos = mesh_attributes["v_pos"].cuda()
            v_nrm = mesh_attributes["v_nrm"].cuda()
            v_tex = mesh_attributes["v_tex"].cuda()
            v_tng = mesh_attributes["v_tng"].cuda()
            t_pos_idx = mesh_attributes["t_pos_idx"].cuda()
            t_nrm_idx = mesh_attributes["t_nrm_idx"].cuda()
            t_tex_idx = mesh_attributes["t_tex_idx"].cuda()
            t_tng_idx = mesh_attributes["t_tng_idx"].cuda()
            material = Material(mesh_attributes["mat_dict"])
            material = material.cuda()
            ref_mesh = Mesh(v_pos=v_pos, v_nrm=v_nrm, v_tex=v_tex, v_tng=v_tng, 
                            t_pos_idx=t_pos_idx, t_nrm_idx=t_nrm_idx, 
                            t_tex_idx=t_tex_idx, t_tng_idx=t_tng_idx, material=material)
            
        pose_list_sample = batch['pose_list'][0][i]  # mvp
        camera_pos_sample = batch['camera_pos'][0][i]  # campos, mv.inverse
        c2w_list_sample = batch['c2w_list'][0][i]  
        
        spec_env = batch['env_list'][0][0]
        spec_env_ = []
        for k in range(len(spec_env)):
            spec_env_.append(spec_env[k][i])
        diff_env = batch['env_list'][0][1][i]  
        
        env_list_sample = [spec_env_, diff_env]##batch['env_list'][0][i]  
        #material_list_sample = batch['material_list'][0][i]   
        metallic =  batch['material_list'][0][0][i]
        roughness = batch['material_list'][0][1][i]   
        material_list_sample = [metallic, roughness]
        camera_embeddings = batch['camera_embedding_list'][0][i] 


        sample_input_images, sample_input_alphas, sample_input_depths, sample_input_normals, sample_input_albedos = [], [], [], [], []
        sample_input_w2cs, sample_input_Ks, sample_input_camera_pos, sample_input_c2ws = [], [], [], []
        sample_input_camera_embeddings = []
        sample_input_bg = []
        sample_input_spec_light, sample_input_diff_light, sample_input_spec_albedo, sample_input_diff_albedo = [], [], [], []

        sample_target_images, sample_target_alphas, sample_target_depths, sample_target_normals, sample_target_albedos = [], [], [], [], []
        sample_target_w2cs, sample_target_Ks, sample_target_camera_pos = [], [], []
        sample_target_spec_light, sample_target_diff_light, sample_target_spec_albedo, sample_target_diff_albedo = [], [], [], []

        sample_input_env = []
        sample_input_materials = []
        sample_target_env = []
        sample_target_materials = []
        
        #for i in range(len(pose_list_sample)):
            
        mvp = pose_list_sample
        campos = camera_pos_sample

        env = env_list_sample
        materials = material_list_sample
        
        # breakpoint()
        with torch.no_grad():

            buffer_dict, bg = render.render_mesh(glctx, ref_mesh, mvp.cuda(), campos.cuda(), [env], None, None, 
                                            materials, iter_res, spp=iter_spp, num_layers=layers, msaa=True, 
                                            background=None, gt_render=True)
  
        image = buffer_dict['shaded'][0][...,:3]
        albedo = convert_to_white_bg(buffer_dict['albedo'][0])
        alpha = buffer_dict['mask'][0][:, :, 3:]  
        depth = convert_to_white_bg(buffer_dict['depth'][0])
        normal = convert_to_white_bg(buffer_dict['gb_normal'][0], write_bg=False)
        #breakpoint()
        spec_light = convert_to_white_bg(buffer_dict['spec_light'][0])
        diff_light = convert_to_white_bg(buffer_dict['diff_light'][0])

        #breakpoint()
        image = image.clamp(0., 1.)
        albedo = albedo.clamp(0., 1.)
        spec_light = spec_light.clamp(0., 1.)
        diff_light = diff_light.clamp(0., 1.)
    
        sample_input_images.append(image)
        sample_input_bg.append(bg[:, :, :, :3].squeeze(0))
        sample_input_albedos.append(albedo)
        sample_input_alphas.append(alpha)
        sample_input_depths.append(depth)
        sample_input_normals.append(normal)
        sample_input_spec_light.append(spec_light)
        sample_input_diff_light.append(diff_light)
        sample_input_materials.append(materials)

        input_images.append(torch.stack(sample_input_images, dim=0).permute(0, 3, 1, 2))
        input_bg.append(torch.stack(sample_input_bg, dim=0).permute(0, 3, 1, 2))
        input_albedos.append(torch.stack(sample_input_albedos, dim=0).permute(0, 3, 1, 2))
        input_alphas.append(torch.stack(sample_input_alphas, dim=0).permute(0, 3, 1, 2))
        input_depths.append(torch.stack(sample_input_depths, dim=0).permute(0, 3, 1, 2))
        input_normals.append(torch.stack(sample_input_normals, dim=0).permute(0, 3, 1, 2))
        input_spec_light.append(torch.stack(sample_input_spec_light, dim=0).permute(0, 3, 1, 2))
        input_diff_light.append(torch.stack(sample_input_diff_light, dim=0).permute(0, 3, 1, 2))
        # input_spec_albedo.append(torch.stack(sample_input_spec_albedo, dim=0).permute(0, 3, 1, 2))
        # input_diff_albedo.append(torch.stack(sample_input_diff_albedo, dim=0).permute(0, 3, 1, 2))
        # input_w2cs.append(torch.stack(sample_input_w2cs, dim=0))
        # input_camera_pos.append(torch.stack(sample_input_camera_pos, dim=0))
        # input_c2ws.append(torch.stack(sample_input_c2ws, dim=0))
        # input_camera_embeddings.append(torch.stack(sample_input_camera_embeddings, dim=0))
        # input_Ks.append(torch.stack(sample_input_Ks, dim=0))
        # input_env.append(sample_input_env)
        input_materials.append(sample_input_materials)

        del ref_mesh
        del material
        del mesh_attributes
        torch.cuda.empty_cache()

    data = {
        'input_images': torch.stack(input_images, dim=0).detach(),           # (batch_size, input_view_num, 3, H, W)
        'input_alphas': torch.stack(input_alphas, dim=0).detach(),           # (batch_size, input_view_num, 1, H, W) 
        'input_depths': torch.stack(input_depths, dim=0).detach(),  
        'input_normals': torch.stack(input_normals, dim=0).detach(), 
        'input_albedos': torch.stack(input_albedos, dim=0).detach(), 
        'input_spec_light': torch.stack(input_spec_light, dim=0).detach(), 
        'input_diff_light': torch.stack(input_diff_light, dim=0).detach(), 
        # 'input_spec_albedo': torch.stack(input_spec_albedo, dim=0), 
        # 'input_diff_albedo': torch.stack(input_diff_albedo, dim=0), 
        'input_materials': input_materials,
        'input_env': torch.stack(input_bg, dim=0).detach(),

    }

    return data

import matplotlib.pyplot as plt
def unnormalize(tensor, mean, std):
        """Revert the normalization of image data."""
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # in-place multiplication and addition
        return tensor
def save_tensor_as_image(tensor, filename, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Save a PyTorch tensor as an image file after unnormalizing."""
    tensor = unnormalize(tensor.clone(), mean, std)  # Clone to avoid changing the original tensor
    npimg = tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def vis_dataset(images, env_image, normal_image, albedo_image, metallic_image, roughness_image, depth_image,  spec_light_image, diff_light_image, mask_image):

    normal_grid = torchvision.utils.make_grid(normal_image.cpu())
    save_tensor_as_image(normal_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'normal_image_{1}.png'))

    # Save 'albedo' images
    albedo_grid = torchvision.utils.make_grid(albedo_image.cpu())
    save_tensor_as_image(albedo_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'albedo_image_{1}.png'))

    # Save 'rgb' images
    rgb_grid = torchvision.utils.make_grid(images.cpu())
    save_tensor_as_image(rgb_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'rgb_image_{1}.png'))

    env_grid = torchvision.utils.make_grid(env_image.cpu())
    save_tensor_as_image(env_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'env_image_{1}.png'))
    
    metallic_grid = torchvision.utils.make_grid(metallic_image.cpu())
    save_tensor_as_image(metallic_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'mel_image_{1}.png'))

    roughness_grid = torchvision.utils.make_grid(roughness_image.cpu())
    save_tensor_as_image(roughness_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'roughness_image_{1}.png'))
    
    # depth_grid = torchvision.utils.make_grid(depth_image.cpu())
    # save_tensor_as_image(depth_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'depth_image_{1}.png'))

    spec_light_grid = torchvision.utils.make_grid(spec_light_image.cpu())
    save_tensor_as_image(spec_light_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'spec_image_{1}.png'))

    diff_light_grid = torchvision.utils.make_grid(diff_light_image.cpu())
    save_tensor_as_image(diff_light_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'diff_image_{1}.png'))
    
    diff_light_grid = torchvision.utils.make_grid(mask_image.cpu())
    save_tensor_as_image(diff_light_grid, os.path.join('/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data', f'mask_image_{1}.png'))
    #breakpoint() 


def compute_t(len_t, num_timesteps, bs, device):
    all_t = torch.zeros(len_t, bs).to(device)
    idx = random.randint(0, len_t-1)
    all_t[idx] = torch.randint(0, num_timesteps, (bs,), device=device).long()

    for i in range(len_t):
        if i != idx:
            for j in range(bs):
                all_t[i,j] = random.choice([0, num_timesteps-1])
    #print(all_t.long())
    return all_t.long()


logger = get_logger(__name__)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def psnr(target, ref, scale=255):
    """
    Computes the Peak Signal-to-Noise Ratio between two images.
    
    Args:
    - target (numpy.ndarray): Target image (numpy array).
    - ref (numpy.ndarray): Reference image (numpy array) to which target is compared.
    - scale (int, optional): The maximum possible pixel value of the images. Default is 255 for 8-bit images.

    Returns:
    - float: The PSNR value in decibels (dB).
    """
    # First, calculate the Mean Squared Error (MSE) between the images
    mse = np.mean((target - ref) ** 2)
    
    # If MSE is zero, the PSNR is infinite (images are identical)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(scale / np.sqrt(mse))
    return psnr


def preprocess_mask(image_path, mask_path):
    image = np.array(Image.open(image_path).convert('RGB').resize((args.resolution, args.resolution)))
    mask_img = np.array(Image.open(mask_path).resize((args.resolution, args.resolution)))
    mask = np.array(mask_img)[:, :, 3] > 128
    mask = mask.astype(bool)
    mask = np.stack([mask]*3, axis=-1)
    masked_image = np.zeros_like(image)  # Start with an all-black image
    masked_image[mask] = image[mask] 
    return np.transpose(masked_image, [2, 0, 1])


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f'mask{i}.png')
        # plt.show()

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, 
    controldec, args, accelerator, weight_dtype, step, img_guidance_scale=0.0, compute_times=5, mask_guidance_scale=0.0, sam_predictor=None):
    logger.info("Running validation... ")

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    assert sam_predictor is not None

    def calc_lpips(pred, gt):
        pred = pred.permute(2, 0, 1)
        gt = gt.permute(2, 0, 1)
        return loss_fn_vgg(pred, gt).item()
    
    controlnet = accelerator.unwrap_model(controlnet)
    controldec = accelerator.unwrap_model(controldec)
    unet = accelerator.unwrap_model(unet)

    pipeline = StableDiffusionControl2BranchFtudecPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        controldec=controldec,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    
    # breakpoint()
    pipeline.scheduler_img = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_attr = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_material = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_albedo = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_normal = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_spec_light = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_diff_light = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_env = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = None
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # ---------- mask2image/image2mask/joint sampling ------------
    print("prediction type:", pipeline.scheduler_img.prediction_type)
    image_logs = []

    # Ensure that args.validation_image is a string path to the folder
    if isinstance(args.validation_image, list):
    # Check if the first element is a directory
        if os.path.isdir(args.validation_image[0]):
            image_folder = args.validation_image[0]
        else:
            # If it's a file, get the directory
            image_folder = os.path.dirname(args.validation_image[0])
    else:
        image_folder = args.validation_image  # Should be a string path

    if isinstance(args.validation_mask, list):
    # Check if the first element is a directory
        if os.path.isdir(args.validation_mask[0]):
            mask_folder = args.validation_mask[0]
        else:
            # If it's a file, get the directory
            mask_folder = os.path.dirname(args.validation_mask[0])
    else:
        mask_folder = args.validation_mask  # Should be a string path

    resolution = args.resolution  # The desired resolution

    # Get a list of all image files in the folder (supports .png, .jpg, .jpeg)
    image_extensions = ('*.png',)
    mask_extensions = ('*.png',)
    image_paths = []
    mask_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_folder, ext)))
        mask_paths.extend(glob(os.path.join(mask_folder, ext)))
        
    # Create output folders if they don't exist
    output_base_path = args.output_dir # The given path where you want to create output folders

    # Ensure the output_base_path exists
    os.makedirs(output_base_path, exist_ok=True)
    image_paths.sort()
    mask_paths.sort()       
    # Define the output folders relative to output_base_path
    output_folders = ["metallic", "roughness", "albedo", "normal", "specular", "diffuse"]
    for folder in output_folders:
        folder_path = os.path.join(output_base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    prompts = ' '
    # Loop over each image
    for image_path, mask_path in tqdm(zip(image_paths,mask_paths), desc="Processing images"):
        # Load and preprocess the image
        validation_img = Image.open(image_path).convert("RGB").resize((resolution, resolution))
        validation_mask = Image.open(mask_path).convert("L").resize((resolution, resolution))
        # image_array = np.array(validation_img)
        
        # height, width, _ = image_array.shape
        # center_x = width // 2
        # center_y = height // 2

        # input_point = np.array([[center_x, center_y]])
        # input_label = np.array([1])

        # # Predict the mask using SAM2
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        #     sam_predictor.set_image(image_array)
        #     masks, scores, _ = sam_predictor.predict(
        #         point_coords=input_point,
        #         point_labels=input_label,
        #     )
        #     #show_masks(image_array, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

        # # Convert mask back to PIL Image
        # validation_mask = Image.fromarray((masks[0] * 255).astype(np.uint8))
        #breakpoint()
        # Process the image through the pipeline
        # Convert mask_array to a PyTorch tensor and move to GPU
        mask_array = np.array(validation_mask) / 255.0  # Normalize mask to [0, 1]
        #mask_tensor = torch.from_numpy(mask_array).to(material.device)
        with torch.autocast("cuda"):
            metallic_list = []
            roughness_list = []
            albedo_list = []
            normal_list = []
            spec_light_list = []
            diff_light_list = []

            for _ in range(compute_times):
                material, normal, albedo, spec_light, diff_light, env_pred = pipeline.real_image2mask_3mod_albedo(
                    prompts,
                    validation_img, 
                    validation_mask,
                    guidance_scale=mask_guidance_scale,
                    height=resolution, width=resolution, num_inference_steps=20, generator=generator
                )
                # Process metallic and roughness using mask_tensor

                metallic = (material[0, :2].mean().cpu() + 1) * 127.5 * mask_array
                roughness = (material[0, 2:].mean().cpu() + 1) * 127.5 * mask_array
                #breakpoint()
                # Append results to lists, moving tensors to CPU and converting to NumPy arrays
                metallic_list.append(metallic.cpu().numpy())
                roughness_list.append(roughness.cpu().numpy())
            metallic_image = Image.fromarray(np.mean(metallic_list, axis=0).astype(np.uint8)) if compute_times > 1 else Image.fromarray(metallic_list[0].astype(np.uint8))
            roughness_image = Image.fromarray(np.mean(roughness_list, axis=0).astype(np.uint8)) if compute_times > 1 else Image.fromarray(roughness_list[0].astype(np.uint8))

                # # Collect other outputs
                # albedo_list.append(albedo.squeeze(0).permute(1, 2, 0).cpu().numpy())
                # normal_list.append(normal.squeeze(0).permute(1, 2, 0).cpu().numpy())
                # spec_light_list.append(spec_light.squeeze(0).permute(1, 2, 0).cpu().numpy())
                # diff_light_list.append(diff_light.squeeze(0).permute(1, 2, 0).cpu().numpy())

                # Average over compute_times if needed
            # metallic_avg = np.mean(metallic_list, axis=0).astype(np.uint8)
            # roughness_avg = np.mean(roughness_list, axis=0).astype(np.uint8)
            # albedo_avg = np.mean(albedo_list, axis=0)
            # normal_avg = np.mean(normal_list, axis=0)
            # spec_light_avg = np.mean(spec_light_list, axis=0)
            # diff_light_avg = np.mean(diff_light_list, axis=0)

            # Get the base filename without extension
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            # Save outputs to their respective folders
            # Save metallic
            # metallic_image = Image.fromarray(metallic_avg)
            metallic_image_path = os.path.join(output_base_path, 'metallic', f'metallic_{base_filename}.png')
            metallic_image.save(metallic_image_path)

            # Similarly for other outputs
            roughness_image_path = os.path.join(output_base_path, 'roughness', f'roughness_{base_filename}.png')
            roughness_image.save(roughness_image_path)

            albedo_image_path = os.path.join(output_base_path, 'albedo', f'albedo_{base_filename}.png')
            albedo[0].save(albedo_image_path)

            normal_image_path = os.path.join(output_base_path, 'normal', f'normal_{base_filename}.png')
            normal[0].save(normal_image_path)

            specular_image_path = os.path.join(output_base_path, 'specular', f'specular_{base_filename}.png')
            spec_light[0].save(specular_image_path)

            diffuse_image_path = os.path.join(output_base_path, 'diffuse', f'diffuse_{base_filename}.png')
            diff_light[0].save(diffuse_image_path)
        
                

    return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--step",
        type=str,
        default="10000",
        help=(
            "check-point step use for inference"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--miou",# for testing
        action="store_true",

    )
    parser.add_argument(
        "--test_batch_size", type=int, default=4, help="Batch size for the testing dataloader."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning mask be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_mask`s, or a single"
            " `--validation_mask` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=6,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    # if args.dataset_name is not None and args.train_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    # if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
    #     raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # if args.validation_prompt is not None and args.validation_image is None:
    #     raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    # if args.validation_prompt is None and args.validation_image is not None:
    #     raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    # if (
    #     args.validation_image is not None
    #     and args.validation_prompt is not None
    #     and len(args.validation_image) != 1
    #     and len(args.validation_prompt) != 1
    #     and len(args.validation_image) != len(args.validation_prompt)
    # ):
    #     raise ValueError(
    #         "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
    #         " or the same number of `--validation_prompt`s and `--validation_image`s"
    #     )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    checkpoint = "./sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint), device=accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )


    # from torch.utils.data import random_split
    # #train_dataset = BlenderGenDataset_old(root_dir=args.train_data_dir, mode='train', transform=transforms, resize=(256, 256))
    # root_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_highQuality_singleObj_OBJ_Mesh_final_Full_valid'
    # light_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/env_mipmap_gaint'
    # #train_dataset = BlenderGenDataset_3mod_demo(root_dir=args.train_data_dir, mode='train', resize=(args.resolution, args.resolution), random_flip=False, random_crop=False)
    # #train_dataset = ObjaverseData(root_dir=root_dir, light_dir=light_dir)
    # train_dataset= ObjaverseData_overfit_5(root_dir=root_dir, light_dir=light_dir)

    # # Calculate the sizes of train and test sets
    # #train_size = int(0.9995 * len(train_dataset))
    # #test_size = len(train_dataset) - train_size

    # #train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    # #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    # # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    logger.info("Loading existing pretrain weights")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    logger.info("Loading existing Unet, Controlnet, Controlnet Decoder weights")
    unet = UNet2DConditionModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="unet")
    controlnet = ControlNetVAENoImgResOneCtlModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controlnet")
    controldec = _UnetDecControlModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controldec")

    # vae.requires_grad_(False)
    # unet.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    # controlnet.train()
    # controldec.train()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    #unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    controlnet, controldec, unet = accelerator.prepare(controlnet, controldec, unet)
    image_logs = log_validation(
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        controldec,
        args,
        accelerator,
        weight_dtype,
        'final',
        sam_predictor=sam_predictor,
        img_guidance_scale=0, 
        mask_guidance_scale=0,
    )
    # image_logs = log_validation(
    #     vae,
    #     text_encoder,
    #     tokenizer,
    #     unet,
    #     controlnet,
    #     controldec,
    #     args,
    #     accelerator,
    #     weight_dtype,
    #     'final',
    #     img_guidance_scale=2.5, 
    #     mask_guidance_scale=7.5,
    # )


if __name__ == "__main__":
    args = parse_args()
    main(args)
