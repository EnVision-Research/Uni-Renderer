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
#import lpips
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
import sys
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

# from dataset.blendGen import BlenderGenDataset, BlenderGenDataset_old, BlenderGenDataset_3mod, BlenderGenDataset_3mod_demo, BlenderGenDataset_3mod_demo_validation
from dataset.objaverse import ObjaverseData, ObjaverseData_overfit_5
# from utils_metrics.compute_t import compute_t
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.controlnet import ControlNetVAENoImgResOneCtlModel, _UnetDecControlModel, UNet2DConditionModel
from models.pipeline import StableDiffusionControl2BranchFtudecPipeline
from utils_metrics.inception import InceptionV3
from utils_metrics.calc_fid import calculate_frechet_distance, extract_features_from_samples
from utils_metrics.metrics_util import SegMetric, NormalMetric, calculate_miou_per_batch
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



def collate_fn(batch, single_obj=False):
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
    if not single_obj:
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
    else:
        for i in range(batch_size):
            obj_path = batch['obj_path'][0]

            mtl_path = os.path.splitext(batch['obj_path'][0])[0] + '.mtl'
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
            
            spec_env = batch['env_list'][0][0] #[0][i]
            spec_env_ = []
            for k in range(len(spec_env)):
                spec_env_.append(spec_env[k][i])
            diff_env = batch['env_list'][0][1][i]


            env_list_sample = [spec_env_, diff_env]##batch['env_list'][0][i]  
            #material_list_sample = batch['material_list'][0][i]   
            metallic =  batch['material_list'][0][i][0]
            roughness = batch['material_list'][0][i][1]   
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
    # normal_image = normal_image.permute(0, 3, 1, 2)
    # albedo_image = albedo_image.permute(0, 3, 1, 2)
    # images = images.permute(0, 3, 1, 2)
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

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
def calc_psnr(pred_tensor, gt_tensor):
    gt_tensor = torch.from_numpy(gt_tensor).permute(2, 0, 1).float() / 255.0
    pred_tensor = torch.from_numpy(pred_tensor).permute(2, 0, 1).float() / 255.0

    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(pred_tensor, gt_tensor, data_range=1.0)
    return psnr_value.item()

def calc_ssim(pred_tensor, gt_tensor):
    gt_tensor = torch.from_numpy(gt_tensor).permute(2, 0, 1).float() / 255.0
    pred_tensor = torch.from_numpy(pred_tensor).permute(2, 0, 1).float() / 255.0

    ssim_value = structural_similarity_index_measure(
    pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0), data_range=1.0)
    return ssim_value.item()


def preprocess_mask(image_path, mask_path):
    image = np.array(Image.open(image_path).convert('RGB').resize((args.resolution, args.resolution)))
    mask_img = np.array(Image.open(mask_path).resize((args.resolution, args.resolution)))
    mask = np.array(mask_img)[:, :, 3] > 128
    mask = mask.astype(bool)
    mask = np.stack([mask]*3, axis=-1)
    masked_image = np.zeros_like(image)  # Start with an all-black image
    masked_image[mask] = image[mask] 
    return np.transpose(masked_image, [2, 0, 1])

from lpips_pytorch import LPIPS, lpips

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, 
    controldec, args, accelerator, weight_dtype, step, test_dataloader, img_guidance_scale=7.5, compute_times=1, mask_guidance_scale=7.5):
    logger.info("Running validation... ")

    def calc_lpips(pred, gt):
        pred = pred.permute(2, 0, 1)
        gt = gt.permute(2, 0, 1)
        # criterion = LPIPS(net_type='alex',version='0.1').cuda()
        # loss = criterion(x, y)

# functional call
        loss = lpips(pred, gt, net_type='vgg', version='0.1').cuda()
        return loss.item()
    
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

    # validation_img = Image.open(args.validation_image[0]).convert("RGB").resize((args.resolution, args.resolution))
    # material_image = np.load(args.validation_image[1])
    # normal_image = Image.open(args.validation_image[2]).convert("RGB").resize((args.resolution, args.resolution))
    # light_image = Image.open(
    #     "/hpc2hdd/home/zchen379/working/christmas_photo_studio_07_4k.png"
    #     ).convert("RGB").resize((args.resolution, args.resolution))

    # test_data_path = args.validation_image[0]
    # if test_data_path[-1] != '/':
    #     test_data_path += '/'

    #paths = glob(test_data_path+'*.png')
    # num_validation_images = len(paths)
    images_list, real_images, re_render_images, env_list, env_gt_list = [], [], [], [], []
    metallics, roughnesses, masks_list, normals, albedos, diff_lights, spec_lights = [], [], [], [], [], [], []
    metallics_gt, roughnesses_gt, masks_list, normals_gt, albedos_gt, diff_lights_gt, spec_lights_gt = [], [], [], [], [], [], []
    metallic_oos_scaled, roughness_oos_scaled = [], []
    progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)

    psnr_normal, psnr_albedo, psnr_rgb, psnr_rgb_rerend, psnr_diffuse, psnr_specular = [], [], [], [], [], []
    lpips_normal, lpips_albedo, lpips_rgb, lpips_rgb_rerend, lpips_diffuse, lpips_specular = [], [], [], [], [], []
    ssim_normal, ssim_albedo, ssim_rgb, ssim_rgb_rerend, ssim_diffuse, ssim_specular = [], [], [], [], [], []

    m_error_list, r_error_list = [], []
    for raw_batch in test_dataloader:
        batch = collate_fn(raw_batch)
        #breakpoint()
        images = batch['input_images'].squeeze(1)
        # depth = batch['input_depths'].squeeze(1)
        env_image = batch['input_env'].squeeze(1)

        #breakpoint()
        masks_image = batch['input_alphas'].squeeze(1) # norm done in process_material_and_mask
        masks_4_eval = batch['input_alphas'].squeeze(1)  
        masks_image = torch.cat([masks_image, masks_image, masks_image], dim=1)   
        # Attributes
        spec_light_image = batch['input_spec_light'].squeeze(1)
        diff_light_image = batch['input_diff_light'].squeeze(1)
        #breakpoint()
        materials = batch['input_materials']
        material_image, metallic_image, roughness_image = process_material_and_mask(materials, masks_image, images.device)
        normal_image = batch['input_normals'].squeeze(1)
        albedo_image = batch['input_albedos'].squeeze(1)
        # breakpoint()
        prompts = ' '


        ############ dataset test ############
        #vis_dataset(normalize(images), normalize(env), normalize(normal_image), normalize(albedo_image), metallic_image, roughness_image, depth_image, normalize(spec_light_image), normalize(diff_light_image), normalize(masks))

        #breakpoint()
        ############ dataset test ############

        with torch.autocast("cuda"):
            metallic_list = []
            roughness_list = []

            for _ in range(compute_times):
                material, normal, albedo, spec_light, diff_light, env_pred = pipeline.image2mask_3mod_albedo(
                    prompts,
                    images, 
                    masks_image,
                    guidance_scale=mask_guidance_scale,
                    height=args.resolution, width=args.resolution, num_inference_steps=10, generator=generator
                )

                #breakpoint()
                metallic, _, roughness = material[0].split()

                metallic_list.append(np.array(metallic))
                roughness_list.append(np.array(roughness))

            # metallic and roughness
            metallic = Image.fromarray(np.mean(np.array(metallic_list), axis=0).astype(np.uint8)) if compute_times > 1 else Image.fromarray(metallic_list[0])
            roughness = Image.fromarray(np.mean(np.array(roughness_list), axis=0).astype(np.uint8)) if compute_times > 1 else Image.fromarray(roughness_list[0])
            # breakpoint()
            binary_mask = (np.array(masks_4_eval[:, 0:1, :, :].detach().cpu()) > 0.5).squeeze(0).squeeze(0)     # [0, 1]
            not_binary_mask = (np.array(masks_4_eval[:, 0:1, :, :].detach().cpu()) < 0.5).squeeze(0).squeeze(0)   # [0, 1]

            metallic_pred_value = np.mean(np.array(metallic)[binary_mask]) / 255.   # [0, 255] -> [0, 1]
            roughness_pred_value = np.mean(np.array(roughness)[binary_mask]) / 255.   # [0, 255] -> [0, 1]
            metallic_gt_value = (np.mean(np.array(metallic_image.squeeze(0).squeeze(0).detach().cpu())[binary_mask]) + 1) / 2   # [-1, 1] -> [0, 1]
            roughness_gt_value = (np.mean(np.array(roughness_image.squeeze(0).squeeze(0).detach().cpu())[binary_mask]) + 1) / 2   # [-1, 1] -> [0, 1]
            
            m_error_list.append(math.sqrt((metallic_pred_value - metallic_gt_value)**2))
            r_error_list.append(math.sqrt((roughness_pred_value - roughness_gt_value)**2))

            # 两个值算psnr感觉没必要，直接算mean error吧
            # metallic_one_over_se = 1. / math.sqrt((metallic_pred_value - metallic_gt_value + 1e-5)**2)
            # roughness_one_over_se = 1. / math.sqrt((roughness_pred_value - roughness_gt_value + 1e-5)**2)

            # metallic_oos_scaled.append(math.log10(metallic_one_over_se) * 20.)
            # roughness_oos_scaled.append(math.log10(roughness_one_over_se) * 20.)

            metallic_3_channel = Image.merge("RGB", (metallic, metallic, metallic))
            roughness_3_channel = Image.merge("RGB", (roughness, roughness, roughness))
            # mask_3_channel = Image.merge("RGB", (mask, mask, mask))

            # normal
            normal_gt = ((normal_image.squeeze().permute(1, 2, 0) + 1) * 127.5).cpu().numpy().astype(np.uint8)
            normal_pred = np.array(normal[0])
            normal_pred[not_binary_mask] = 255
            psnr_normal.append(calc_psnr(normal_pred, normal_gt))
            ssim_normal.append(calc_ssim(normal_pred, normal_gt))
            
            normal_gt = torch.from_numpy(normal_gt).cuda()
            normal_pred = torch.from_numpy(normal_pred).cuda()
            lpips_normal.append(calc_lpips(normal_pred, normal_gt))

            # albedo
            albedo_gt = ((albedo_image.squeeze().permute(1, 2, 0) + 1) * 127.5).cpu().numpy().astype(np.uint8)
            albedo_pred = np.array(albedo[0])
            albedo_pred[not_binary_mask] = 255
            psnr_albedo.append(calc_psnr(albedo_pred, albedo_gt))
            ssim_albedo.append(calc_ssim(albedo_pred, albedo_gt))

            albedo_gt = torch.from_numpy(albedo_gt).cuda()
            albedo_pred = torch.from_numpy(albedo_pred).cuda()
            lpips_albedo.append(calc_lpips(albedo_pred, albedo_gt))


            # diffuse
            diffuse_gt = ((diff_light_image.squeeze().permute(1, 2, 0) + 1) * 127.5).cpu().numpy().astype(np.uint8)
            diffuse_pred = np.array(diff_light[0])
            diffuse_pred[not_binary_mask] = 255
            psnr_diffuse.append(calc_psnr(diffuse_pred, diffuse_gt))
            ssim_diffuse.append(calc_ssim(diffuse_pred, diffuse_gt))

            diffuse_gt = torch.from_numpy(diffuse_gt).cuda()
            diffuse_pred = torch.from_numpy(diffuse_pred).cuda()
            lpips_diffuse.append(calc_lpips(diffuse_pred, diffuse_gt))
            # breakpoint()

            # specular
            specular_gt = ((spec_light_image.squeeze().permute(1, 2, 0) + 1) * 127.5).cpu().numpy().astype(np.uint8)
            specular_pred = np.array(spec_light[0])
            specular_pred[not_binary_mask] = 255
            psnr_specular.append(calc_psnr(specular_pred, specular_gt))
            ssim_specular.append(calc_ssim(specular_pred, specular_gt))

            specular_gt = torch.from_numpy(specular_gt).cuda()
            specular_pred = torch.from_numpy(specular_pred).cuda()
            lpips_specular.append(calc_lpips(specular_pred, specular_gt))

            # breakpoint()
            rgb_image = pipeline.mask2image_3mod_albedo(
                [""],
                material_image,
                normal_image, 
                albedo_image,
                spec_light_image,
                diff_light_image,
                env_image,
                masks_image, 
                guidance_scale=img_guidance_scale,
                height=args.resolution, width=args.resolution, num_inference_steps=20, generator=generator
            )

            ################ re-render ###################
            # material, normal, albedo, spec_light, diff_light, env_pred
            # breakpoint()
            re_rendered_rgb_image = pipeline.mask2image_3mod_albedo(
                [""],
                to_tensor(material[0]), 
                to_tensor(normal[0]), 
                to_tensor(albedo[0]), 
                to_tensor(spec_light[0]), 
                to_tensor(diff_light[0]), 
                to_tensor(env_pred[0]),
                masks_image, 
                re_rendering=True,
                guidance_scale=img_guidance_scale,
                height=args.resolution, width=args.resolution, num_inference_steps=20, generator=generator
            )
            # breakpoint()
            ################ re-render ###################
            rgb_image = np.array(rgb_image[0])
            re_rendered_rgb_image = np.array(re_rendered_rgb_image[0])
            rgb_gt = (images.squeeze().permute(1, 2, 0)*255.).cpu().numpy().astype(np.uint8)
            rgb_pred = rgb_image
            # breakpoint()
            
            psnr_rgb.append(calc_psnr(rgb_pred, rgb_gt))
            ssim_rgb.append(calc_ssim(rgb_pred, rgb_gt))
            psnr_rgb_rerend.append(calc_psnr(re_rendered_rgb_image, rgb_gt))
            ssim_rgb_rerend.append(calc_ssim(re_rendered_rgb_image, rgb_gt))

            rgb_gt = torch.from_numpy(rgb_gt).cuda()
            rgb_pred = torch.from_numpy(rgb_pred).cuda()
            re_rendered_rgb_image = torch.from_numpy(re_rendered_rgb_image).cuda()
            lpips_rgb.append(calc_lpips(rgb_pred, rgb_gt))
            lpips_rgb_rerend.append(calc_lpips(re_rendered_rgb_image, rgb_gt))
            # rgb_image = Image.fromarray(rgb_image)
            
            progress_bar.update(1)
            breakpoint()


            ####### documentation ########
    save_img_dir = os.path.join(args.output_dir, 'score')
    os.makedirs(save_img_dir, exist_ok=True)
    psnr_normal_mean = sum(psnr_normal) / len(psnr_normal)
    psnr_albedo_mean = sum(psnr_albedo) / len(psnr_albedo)
    psnr_rgb_mean = sum(psnr_rgb) / len(psnr_rgb)
    psnr_rgb_rerend_mean = sum(psnr_rgb_rerend) / len(psnr_rgb_rerend)
    psnr_diffuse_mean = sum(psnr_diffuse) / len(psnr_diffuse)
    psnr_specular_mean = sum(psnr_specular) / len(psnr_specular)

    # Compute the mean of LPIPS metrics
    lpips_normal_mean = sum(lpips_normal) / len(lpips_normal)
    lpips_albedo_mean = sum(lpips_albedo) / len(lpips_albedo)
    lpips_rgb_mean = sum(lpips_rgb) / len(lpips_rgb)
    lpips_rgb_rerend_mean = sum(lpips_rgb_rerend) / len(lpips_rgb_rerend)
    lpips_diffuse_mean = sum(lpips_diffuse) / len(lpips_diffuse)
    lpips_specular_mean = sum(lpips_specular) / len(lpips_specular)

    ssim_normal_mean = sum(ssim_normal) / len(ssim_normal)
    ssim_albedo_mean = sum(ssim_albedo) / len(ssim_albedo)
    ssim_rgb_mean = sum(ssim_rgb) / len(ssim_rgb)
    ssim_rgb_rerend_mean = sum(ssim_rgb_rerend) / len(ssim_rgb_rerend)
    ssim_diffuse_mean = sum(ssim_diffuse) / len(ssim_diffuse)
    ssim_specular_mean = sum(ssim_specular) / len(ssim_specular)

    # Write the metrics along with the iteration number to a text file
    with open('validation_scores.txt', 'a') as f:
        f.write(f"Step {step}:\n")
        f.write(f"PSNR Normal: {psnr_normal_mean:.4f}\n")
        f.write(f"PSNR Albedo: {psnr_albedo_mean:.4f}\n")
        f.write(f"PSNR RGB: {psnr_rgb_mean:.4f}\n")
        f.write(f"PSNR RGB Rerendered: {psnr_rgb_rerend_mean:.4f}\n")
        f.write(f"PSNR Diffuse: {psnr_diffuse_mean:.4f}\n")
        f.write(f"PSNR Specular: {psnr_specular_mean:.4f}\n")
        f.write(f"LPIPS Normal: {lpips_normal_mean:.4f}\n")
        f.write(f"LPIPS Albedo: {lpips_albedo_mean:.4f}\n")
        f.write(f"LPIPS RGB: {lpips_rgb_mean:.4f}\n")
        f.write(f"LPIPS RGB Rerendered: {lpips_rgb_rerend_mean:.4f}\n")
        f.write(f"LPIPS Diffuse: {lpips_diffuse_mean:.4f}\n")
        f.write(f"LPIPS Specular: {lpips_specular_mean:.4f}\n")
        #Write SSIM metrics
        f.write(f"SSIM Normal: {ssim_normal_mean:.4f}\n")
        f.write(f"SSIM Albedo: {ssim_albedo_mean:.4f}\n")
        f.write(f"SSIM RGB: {ssim_rgb_mean:.4f}\n")
        f.write(f"SSIM RGB Rerendered: {ssim_rgb_rerend_mean:.4f}\n")
        f.write(f"SSIM Diffuse: {ssim_diffuse_mean:.4f}\n")
        f.write(f"SSIM Specular: {ssim_specular_mean:.4f}\n")
        f.write("\n")  # Add an empty line for readability

    logger.info("Validation metrics written to validation_scores.txt")

        


        # from torchvision.transforms import ToPILImage
        # to_pil = ToPILImage()
        # metallics.append(metallic_3_channel)
        # # breakpoint()
        # metallics_gt.append(to_pil(metallic_image[0].repeat(3,1,1) / 2 + 0.5))

        # roughnesses.append(roughness_3_channel)
        # roughnesses_gt.append(to_pil(roughness_image[0].repeat(3,1,1) / 2 + 0.5))
        
        # normals.append(normal[0])
        # normals_gt.append(to_pil(normal_image[0] / 2 + 0.5))
        
        # albedos.append(albedo[0])
        # albedos_gt.append(to_pil(albedo_image[0]))

        # spec_lights.append(spec_light[0])
        # spec_lights_gt.append(to_pil(spec_light_image[0]))

        # diff_lights.append(diff_light[0])
        # diff_lights_gt.append(to_pil(diff_light_image[0]))

        # env_list.append(env_pred[0])
        # env_gt_list.append(to_pil(env_image[0]))

        # real_images.append(to_pil(images[0]))
        # images_list.append(rgb_image)
        # re_render_images.append(re_rendered_rgb_image[0])
        
        #breakpoint()
    # # calc mean of metallic and roughness oos scaled value
    # metallic_oos_scaled = np.mean(np.array(metallic_oos_scaled))
    # roughness_oos_scaled = np.mean(np.array(roughness_oos_scaled))
    # # sum_oos_scaled = metallic_oos_scaled + roughness_oos_scaled

    # psnr_rgb, psnr_normal, psnr_albedo = np.mean(np.array(psnr_rgb)), np.mean(np.array(psnr_normal)), np.mean(np.array(psnr_albedo))
    # lpips_rgb, lpips_normal, lpips_albedo = np.mean(np.array(lpips_rgb)), np.mean(np.array(lpips_normal)), np.mean(np.array(lpips_albedo))
    
    # metallics = [validation_img] + metallics
    # roughnesses = [validation_img] + roughnesses
    # masks = [validation_img] + masks
    # normals = [validation_img] + normals
    # lights = [validation_img] + lights

    # images = [normal_image] + images

    # white_image = Image.new('RGB', (args.resolution, args.resolution), 'white')
    # joint_sample_images = [white_image] + joint_sample_images
    # joint_sample_masks = [white_image] + joint_sample_masks
    # joint_sample_normals = [white_image] + joint_sample_normals
    # plot_num = 20
    # log_dict = {
    #         "metallics": metallics[:plot_num], 
    #         "metallics_gt": metallics_gt[:plot_num], 
    #         "roughnesses": roughnesses[:plot_num],
    #         "roughnesses_gt": roughnesses_gt[:plot_num],
    #         "normals": normals[:plot_num],
    #         "normals_gt": normals_gt[:plot_num],
    #         "albedos": albedos[:plot_num],
    #         "albedos_gt": albedos_gt[:plot_num],
    #         "spec_lights": spec_lights[:plot_num],
    #         "spec_lights_gt": spec_lights_gt[:plot_num],
    #         "diff_lights": diff_lights[:plot_num],
    #         "diff_lights_gt": diff_lights_gt[:plot_num],
    #         "env_pred": env_list[:plot_num], 
    #         "env_gt": env_gt_list[:plot_num],
    #         "images": images_list[:plot_num],
    #         "real_images": real_images[:plot_num],
    #         "re_render_images": re_render_images[:plot_num]
    #         }


    # image_logs.append(log_dict)
    # #breakpoint()
    # save_img_dir = os.path.join(args.output_dir, 'imgs')
    # os.makedirs(save_img_dir, exist_ok=True)
    # # for tracker in accelerator.trackers:
    # #     if tracker.name == "tensorboard":
    # for log in image_logs:
    #     images = log["images"]
    #     formatted_images = []
    #     for image in images:
    #         formatted_images.append(np.asarray(image))
    #     formatted_images = np.stack(formatted_images)

    #     images = log["real_images"]
    #     formatted_real_images = []
    #     for image in images:
    #         formatted_real_images.append(np.asarray(image))
    #     formatted_real_images = np.stack(formatted_real_images)

    #     re_render_images = log["re_render_images"]
    #     formatted_re_render_images = []
    #     for image in re_render_images:
    #         formatted_re_render_images.append(np.asarray(image))
    #     formatted_re_render_images = np.stack(formatted_re_render_images)

    #     #print(123
    #     metallics = log["metallics"]
    #     formatted_metallics = []
    #     for metallic in metallics:
    #         formatted_metallics.append(np.asarray(metallic))
    #     formatted_metallics = np.stack(formatted_metallics)

    #     metallics_gt = log["metallics_gt"]
    #     formatted_metallics_gt = []
    #     for metallic in metallics_gt:
    #         formatted_metallics_gt.append(np.asarray(metallic))
    #     formatted_metallics_gt = np.stack(formatted_metallics_gt)

    #     roughnesses = log["roughnesses"]
    #     formatted_roughnesses = []
    #     for roughness in roughnesses:
    #         formatted_roughnesses.append(np.asarray(roughness))
    #     formatted_roughnesses = np.stack(formatted_roughnesses)

    #     roughnesses_gt = log["roughnesses_gt"]
    #     formatted_roughnesses_gt = []
    #     for roughness in roughnesses_gt:
    #         formatted_roughnesses_gt.append(np.asarray(roughness))
    #     formatted_roughnesses_gt = np.stack(formatted_roughnesses_gt)
        
    #     normals = log["normals"]
    #     formatted_normals= []
    #     for normal in normals:
    #         formatted_normals.append(np.asarray(normal))
    #     formatted_normals = np.stack(formatted_normals)

    #     normals_gt = log["normals_gt"]
    #     formatted_normals_gt = []
    #     for normal in normals_gt:
    #         formatted_normals_gt.append(np.asarray(normal))
    #     formatted_normals_gt = np.stack(formatted_normals_gt)

    #     albedos = log["albedos"]
    #     formatted_albedos= []
    #     for albedo in albedos:
    #         formatted_albedos.append(np.asarray(albedo))
    #     formatted_albedos = np.stack(formatted_albedos)

    #     albedos_gt = log["albedos_gt"]
    #     formatted_albedos_gt= []
    #     for albedo in albedos_gt:
    #         formatted_albedos_gt.append(np.asarray(albedo))
    #     formatted_albedos_gt = np.stack(formatted_albedos_gt)

    #     spec_lights = log["spec_lights"]
    #     formatted_spec_lights= []
    #     for light in spec_lights:
    #         formatted_spec_lights.append(np.asarray(light))
    #     formatted_spec_lights = np.stack(formatted_spec_lights)

    #     spec_lights_gt = log["spec_lights_gt"]
    #     formatted_spec_lights_gt = []
    #     for light in spec_lights_gt:
    #         formatted_spec_lights_gt.append(np.asarray(light))
    #     formatted_spec_lights_gt = np.stack(formatted_spec_lights_gt)

    #     diff_lights = log["diff_lights"]
    #     formatted_diff_lights= []
    #     for light in diff_lights:
    #         formatted_diff_lights.append(np.asarray(light))
    #     formatted_diff_lights = np.stack(formatted_diff_lights)

    #     diff_lights_gt = log["diff_lights_gt"]
    #     formatted_diff_lights_gt = []
    #     for light in diff_lights_gt:
    #         formatted_diff_lights_gt.append(np.asarray(light))
    #     formatted_diff_lights_gt = np.stack(formatted_diff_lights_gt)

    #     env_prediction = log["env_pred"]
    #     formatted_env = []
    #     for env_pred in env_prediction:
    #         formatted_env.append(np.asarray(env_pred))
    #     formatted_env = np.stack(formatted_env)

    #     env_groundtruth = log["env_gt"]
    #     formatted_env_gt = []
    #     for env_gts in env_groundtruth:
    #         formatted_env_gt.append(np.asarray(env_gts))
    #     formatted_env_gt = np.stack(formatted_env_gt)
        
    #     formatted_out = np.concatenate(
    #         [formatted_metallics, formatted_metallics_gt, formatted_roughnesses, formatted_roughnesses_gt, \
    #          formatted_normals, formatted_normals_gt, formatted_albedos, formatted_albedos_gt,  \
    #          formatted_spec_lights, formatted_spec_lights_gt,  formatted_diff_lights, formatted_diff_lights_gt, \
    #          formatted_env, formatted_env_gt, formatted_images, formatted_re_render_images, formatted_real_images], axis=1)

    #     #tracker.writer.add_images(f"validation", formatted_out, step, dataformats="NHWC")
        
    #     n,h,w,c = formatted_out.shape
    #     formatted_out = formatted_out.transpose(1,0,2,3).reshape(h,n*w, c)
    #     im = Image.fromarray(formatted_out)
    #     #breakpoint()
    #     im.save(f'{save_img_dir}/{str(args.step)}_cfg{mask_guidance_scale}_{img_guidance_scale}.png')
    #     #breakpoint()
                

    # return image_logs


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


    from torch.utils.data import random_split
    root_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_highQuality_singleObj_OBJ_Mesh_final_Full_valid'
    root_dir_new = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_OBJ_Mesh_valid'
    light_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/env_mipmap_gaint'
    #train_dataset = BlenderGenDataset_3mod_demo(root_dir=args.train_data_dir, mode='train', resize=(args.resolution, args.resolution), random_flip=False, random_crop=False)
    test_dataset = ObjaverseData_overfit_5(root_dir=root_dir, root_dir_new=root_dir_new, light_dir=light_dir)


    # Calculate the sizes of train and test sets
    #train_size = int(0.9995 * len(train_dataset))
    #test_size = len(train_dataset) - train_size

    #train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # import correct text encoder class
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

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    controldec.train()

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
        test_dataloader,
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
