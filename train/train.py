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

from dataset.objaverse import ObjaverseData, ObjaverseData_test

from models.controlnet import AttributeEncoderModel, AttributeDecoderModel, UNet2DConditionModel
from models.pipeline import UniRendererPipeline

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

import matplotlib.pyplot as plt

def save_tensor_as_image(tensor, filename):
    """Save a PyTorch tensor as an image file after scaling from (-1,1) to (0,255)."""
    tensor = tensor.clone().cpu()
    tensor = ((tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)  # Scale from (-1,1) to (0,255)
    npimg = tensor.numpy()
    
    if tensor.shape[0] == 1:  # Grayscale image
        npimg = npimg.squeeze(0)  # Remove channel dimension
        img = Image.fromarray(npimg, mode='L')  # 'L' mode for grayscale
    else:  # RGB image
        npimg = np.transpose(npimg, (1, 2, 0))  # Convert to HWC format
        img = Image.fromarray(npimg, mode='RGB')  # 'RGB' mode for color images
    
    img.save(filename)

def vis_dataset(images, env_image, normal_image, albedo_image, metallic_image, roughness_image, spec_light_image, diff_light_image, mask_image):
    batch_size = images.size(0)
    save_path = '/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/test_data'
    os.makedirs(save_path, exist_ok=True)

    for i in range(batch_size):
        # Save 'normal' images
        save_tensor_as_image(normal_image[i], os.path.join(save_path, f'normal_image_{i}.png'))

        # Save 'albedo' images
        save_tensor_as_image(albedo_image[i], os.path.join(save_path, f'albedo_image_{i}.png'))

        # Save 'rgb' images
        save_tensor_as_image(images[i], os.path.join(save_path, f'rgb_image_{i}.png'))

        # Save 'environment' images
        save_tensor_as_image(env_image[i], os.path.join(save_path, f'env_image_{i}.png'))

        # Save 'metallic' images
        save_tensor_as_image(metallic_image[i], os.path.join(save_path, f'metallic_image_{i}.png'))

        # Save 'roughness' images
        save_tensor_as_image(roughness_image[i], os.path.join(save_path, f'roughness_image_{i}.png'))

        # Save 'specular light' images
        save_tensor_as_image(spec_light_image[i], os.path.join(save_path, f'spec_image_{i}.png'))

        # Save 'diffuse light' images
        save_tensor_as_image(diff_light_image[i], os.path.join(save_path, f'diff_image_{i}.png'))

        # Save 'mask' images
        save_tensor_as_image(mask_image[i], os.path.join(save_path, f'mask_image_{i}.png'))


def collate_fn(batch, single_obj=True):
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
                
            mvp = pose_list_sample
            campos = camera_pos_sample

            env = env_list_sample
            materials = material_list_sample
            
            
            with torch.no_grad():

                buffer_dict, bg = render.render_mesh(glctx, ref_mesh, mvp.cuda(), campos.cuda(), [env], None, None, 
                                                materials, iter_res, spp=iter_spp, num_layers=layers, msaa=True, 
                                                background=None, gt_render=True)
    
            image = buffer_dict['shaded'][0][...,:3]
            albedo = convert_to_white_bg(buffer_dict['albedo'][0])
            alpha = buffer_dict['mask'][0][:, :, 3:]  
            depth = convert_to_white_bg(buffer_dict['depth'][0])
            normal = convert_to_white_bg(buffer_dict['gb_normal'][0], write_bg=False)
    
            spec_light = convert_to_white_bg(buffer_dict['spec_light'][0])
            diff_light = convert_to_white_bg(buffer_dict['diff_light'][0])

            
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
            
            
            with torch.no_grad():

                buffer_dict, bg = render.render_mesh(glctx, ref_mesh, mvp.cuda(), campos.cuda(), [env], None, None, 
                                                materials, iter_res, spp=iter_spp, num_layers=layers, msaa=True, 
                                                background=None, gt_render=True)
    
            image = buffer_dict['shaded'][0][...,:3]
            albedo = convert_to_white_bg(buffer_dict['albedo'][0])
            alpha = buffer_dict['mask'][0][:, :, 3:]  
            depth = convert_to_white_bg(buffer_dict['depth'][0])
            normal = convert_to_white_bg(buffer_dict['gb_normal'][0], write_bg=False)
            
            spec_light = convert_to_white_bg(buffer_dict['spec_light'][0])
            diff_light = convert_to_white_bg(buffer_dict['diff_light'][0])
            

            
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


def compute_t(len_t, num_timesteps, bs, device):
    assert len_t == 2
    all_t = torch.zeros(len_t, bs).to(device)
    idx = random.randint(0, 1)  # 0: rendering
    all_t[idx] = torch.randint(0, num_timesteps, (bs,), device=device).long()

    for i in range(len_t):
        if i != idx:
            for j in range(bs):
                all_t[i,j] = random.choice([0, num_timesteps-1])
    #print(all_t.long())
    return all_t.long(), bool(idx)


logger = get_logger(__name__)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def preprocess_mask(image_path, mask_path):
    image = np.array(Image.open(image_path).convert('RGB').resize((args.resolution, args.resolution)))
    mask_img = np.array(Image.open(mask_path).resize((args.resolution, args.resolution)))
    mask = np.array(mask_img)[:, :, 3] > 128
    mask = mask.astype(bool)
    mask = np.stack([mask]*3, axis=-1)
    masked_image = np.zeros_like(image)  # Start with an all-black image
    masked_image[mask] = image[mask] 
    return np.transpose(masked_image, [2, 0, 1])



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
        "--dataset_root_dir",
        type=str,
        default="path to the data stores all objaverse data",
        help="The directory where the objaverse data were stored",
    )
    parser.add_argument(
        "--dataset_env_dir",
        type=str,
        default="path to the environment data",
        help="The directory where the environment background data were stored",
    )
    parser.add_argument(
        "--dataset_filter_obj_json",
        type=str,
        default="the json path to the training objects filterd in objaverse",
        help="",
    )
    parser.add_argument(
        "--dataset_test_obj_json",
        type=str,
        default="the json path to the training objects filterd in objaverse",
        help="",
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
        default=2000,
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
        default=4,
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
    # parser.add_argument(
    #     "--train_data_dir",
    #     type=str,
    #     default=None,
    #     help=(
    #         "A folder containing the training data. Folder contents must follow the structure described in"
    #         " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
    #         " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
    #     ),
    # )
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
        "--high_metallic",
        type=bool,
        default=False,
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

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")


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

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)



    task_list = ['img', 'material_image', 'normal', 'light'] 
    from torch.utils.data import random_split

    #root_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_highQuality_singleObj_OBJ_Mesh_final_Full_valid'
    root_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_OBJ_Mesh_valid'
    env_dir = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/env_mipmap_gaint'

    filtered_json_path = '/hpc2hdd/home/zchen379/sd3/objaverse_data/Mesh_final_valid_texturemap.json' 
    test_json_path = '/hpc2hdd/home/zchen379/sd3/objaverse_data/test_ood_one.json'


    train_dataset = ObjaverseData(root_dir=root_dir, filtered_json=filtered_json_path, light_dir=env_dir, high_metallic=args.high_metallic)
    test_dataset = ObjaverseData_test(root_dir=root_dir, filtered_json=test_json_path, light_dir=env_dir, high_metallic=args.high_metallic)


    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load scheduler and models
    print(args.pretrained_model_name_or_path)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = AttributeEncoderModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controlnet")
        controldec = AttributeDecoderModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controldec")
    else:
        logger.info("Initializing controlnet and controldec weights from unet")
        controlnet = AttributeEncoderModel.from_unet(unet, len_t=len(task_list)-1) #zhifei exclude img as we dont pass it to controlnet
        controldec = AttributeDecoderModel.from_unet(unet, len_t=len(task_list)-1)
   

    ## model modification starts HERE ##
    controlnet.conv_in.weight = nn.Parameter(controlnet.conv_in.weight.repeat(1, 7, 1, 1) * 0.142) # material, mask, normal, albedo, specular, diff, env

    # replace config
    config_dict = {}
    for k,v in controlnet._internal_dict.items():
        config_dict[k] = v
    config_dict = EasyDict(config_dict)
    controlnet._internal_dict = config_dict

    controlnet.config["in_channels"] = 28 # 4 x 7

    # Replace the last layer to output 8 out_channels. 
    controldec.conv_out.weight = nn.Parameter(controldec.conv_out.weight.repeat(7, 1, 1, 1) * 0.142)
    controldec.conv_out.bias = nn.Parameter(controldec.conv_out.bias.repeat(7) * 0.142)
    
    config_dict = {}
    for k,v in controldec._internal_dict.items():
        config_dict[k] = v
    config_dict = EasyDict(config_dict)
    controlnet._internal_dict = config_dict
    controldec.config["out_channels"] = 28
    ## model modification ends HERE ##

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    
                    if model.__class__.__name__ == 'AttributeEncoderModel':
                        sub_dir = "controlnet"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    elif model.__class__.__name__ == 'AttributeDecoderModel':
                        sub_dir = "controldec"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    elif model.__class__.__name__ == 'UNet2DConditionModel':
                        sub_dir = "unet"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if model.__class__.__name__ == 'AttributeEncoderModel':
                    subfolder = "controlnet"
                    # load diffusers style into model
                    load_model = AttributeEncoderModel.from_pretrained(input_dir, subfolder=subfolder)
                
                elif model.__class__.__name__ == 'AttributeDecoderModel':
                    subfolder = "controldec"
                    load_model = AttributeDecoderModel.from_pretrained(input_dir, subfolder=subfolder)

                elif model.__class__.__name__ == 'UNet2DConditionModel':
                    subfolder = "unet"
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=subfolder)        

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    #unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    controldec.train()
    unet.train()

    # for _name, _module in unet.named_modules():
    #     if 'up_blocks' in _name: _module.requires_grad_()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
            controldec.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        controldec.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )
    if accelerator.unwrap_model(controldec).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controldec).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = [
        {'params': controlnet.parameters(), 'lr': args.learning_rate},
        {'params': controldec.parameters(), 'lr': args.learning_rate},
        {'params': unet.parameters(), 'lr': args.learning_rate}
    ]
    # params_to_optimize = list(controlnet.parameters()) + list(controldec.parameters()) # Modification
        
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Prepare everything with our `accelerator`.
    controlnet, controldec, unet, optimizer, train_dataloader = accelerator.prepare(
        controlnet, controldec, unet, optimizer, train_dataloader
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # TODO
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate
                param_group['betas'] = (args.adam_beta1, args.adam_beta2)

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    


    import time
    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, raw_batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, controldec, unet):
                # RGBA
                
                batch = collate_fn(raw_batch)
                
                images = normalize(batch['input_images'].squeeze(1))
                masks_copy = batch['input_alphas'].squeeze(1) # we need this copy [0, 1] for creating metallic and roughness 
                masks = normalize(batch['input_alphas'].squeeze(1)) # need to norm

                env = normalize(batch['input_env'].squeeze(1)) # permute(0, 3, 1, 2)[:, :3, :, :]

                # Attributes
                spec_light_image = normalize(batch['input_spec_light'].squeeze(1))
                diff_light_image = normalize(batch['input_diff_light'].squeeze(1))
                materials = batch['input_materials']
                material_image, metallic_image, roughness_image = process_material_and_mask(materials, masks_copy, images.device) #[metallic, roughness, mask]
                normal_image = batch['input_normals'].squeeze(1) # already [-1, 1]
                albedo_image = normalize(batch['input_albedos'].squeeze(1))
                prompts = ' '
                bsz = images.shape[0]

                #prepare time step for smaller tasks space
                timesteps, is_inv_rendering = compute_t(len_t=2, 
                                  num_timesteps=noise_scheduler.config.num_train_timesteps, 
                                  bs=bsz, 
                                  device=images.device)
                
                timesteps_img, timesteps_attribute = timesteps[0], timesteps[1]

                #### img ####
                latents_img = vae.encode(images.to(dtype=weight_dtype)).latent_dist.sample()
                latents_img = latents_img * vae.config.scaling_factor
                noise_img = torch.randn_like(latents_img) # 4*64*6
                noisy_latents_img = noise_scheduler.add_noise(latents_img, noise_img, timesteps_img)   # tianshuo
                #noisy_latents_img = latents_img

                #### material ####
                latents_material = vae.encode(material_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_material = latents_material * vae.config.scaling_factor

                #### mask ####
                mask_3_channel = torch.cat([masks, masks, masks], dim=1)
                latents_mask = vae.encode(mask_3_channel.to(dtype=weight_dtype)).latent_dist.sample()
                latents_mask = latents_mask * vae.config.scaling_factor

                #### env ####
                latents_env = vae.encode(env.to(dtype=weight_dtype)).latent_dist.sample()
                latents_env = latents_env * vae.config.scaling_factor
                
                ## slightly perturbed the env ##
                noise_aug = torch.randn_like(latents_env)
                train_noise_aug = 0.02
                latents_env = latents_env + train_noise_aug * noise_aug

                #### normal ####
                latents_normal = vae.encode(normal_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_normal = latents_normal * vae.config.scaling_factor

                #### albedo ####
                latents_albedo = vae.encode(albedo_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_albedo = latents_albedo * vae.config.scaling_factor

                 #### light ####
                latents_spec_light = vae.encode(spec_light_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_spec_light = latents_spec_light * vae.config.scaling_factor
                

                latents_diff_light = vae.encode(diff_light_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_diff_light = latents_diff_light * vae.config.scaling_factor

                #### concat all of them ####
                latents_attr = torch.cat((latents_material, latents_normal, latents_albedo, latents_spec_light, latents_diff_light, latents_env), dim=1)
                noise_attr = torch.randn_like(latents_attr)
                noisy_latents_attr_part = noise_scheduler.add_noise(latents_attr, noise_attr, timesteps_attribute)
                noisy_latents_attr = torch.cat((latents_mask, noisy_latents_attr_part), dim=1)

                # Get the text embedding for conditioning
                input_ids = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                    truncation=True
                ).input_ids.to(latents_img.device)
                input_ids = input_ids.repeat(bsz, 1) # zhifei repeat the text bs times
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Modification: raw samples from controlnet
                down_block_res_samples, mid_block_res_sample, raw_down_block_res_samples_ctlnet, raw_mid_block_sample_ctlnet = controlnet(
                    noisy_latents_img,
                    timesteps_attribute,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=noisy_latents_attr,
                    return_dict=False,
                )
               
                # Predict the noise residual
                img_pred, raw_down_block_res_samples_unet, raw_mid_block_sample_unet, _ = unet(
                    noisy_latents_img,
                    timesteps_img,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False
                )
                
                mask_pred = controldec(
                    sample = raw_mid_block_sample_ctlnet,
                    down_block_res_samples=raw_down_block_res_samples_ctlnet,
                    timestep = timesteps_attribute,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in raw_down_block_res_samples_unet
                    ],
                    mid_block_additional_residual=raw_mid_block_sample_unet.to(dtype=weight_dtype),
                    return_dict = False
                )
                mask_pred = mask_pred[:, 4:, :, :] # rule out the clean latent mask channel, starting from 4th ch 


                attr_target = latents_attr # for loss computation
                img_target = latents_img

                material_pred, albedo_pred, spec_pred = mask_pred[:, :4, :, :], mask_pred[:, 8:12, :, :], mask_pred[:, 12:16, :, :]
                

                ## Contrastive loss ##

                temperature = 0.1
                m_dis = F.cosine_similarity(material_pred[0].reshape(-1).float(), material_pred[1].reshape(-1).float(), dim=0) / temperature
                a_dis = F.cosine_similarity(albedo_pred[0].reshape(-1).float(), albedo_pred[1].reshape(-1).float(), dim=0) / temperature
                s_dis = F.cosine_similarity(spec_pred[0].reshape(-1).float(), spec_pred[1].reshape(-1).float(), dim=0) / temperature

                pos = torch.exp(a_dis)
                neg = pos + torch.exp(m_dis) + torch.exp(s_dis)
                contrastive_loss = - torch.log(pos / neg)

                loss_img = F.mse_loss(img_pred.float(), img_target.float(), reduction="mean")  
                loss_mask = F.mse_loss(mask_pred.float(), attr_target.float(), reduction="mean")

                loss = loss_img + loss_mask * 10. + contrastive_loss * 0.01

                # -------------------------------------------------------------- #
                # Consistency loss (only when is inverse rendering),             #
                # 1. re-random a new noise and new timestep for img_latents;     #
                # 2. add noise to img_latents and input;                         #
                # 3. give a clean mask in front of 'mask_pred' as input;         #
                # 4. calculate the loss of img_pred and the img_gt.              #
                # -------------------------------------------------------------- #
                
                if is_inv_rendering:
                    noise_img_c = torch.randn_like(latents_img) # 4*64*6
                    timesteps_img_c = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device).long()
                    timesteps_attribute_c = torch.zeros(bsz, ).to(images.device).long()
                    noisy_latents_img_c = noise_scheduler.add_noise(latents_img, noise_img_c, timesteps_img_c)   
                    noisy_latents_attr_c = torch.cat((latents_mask, mask_pred), dim=1)

                    down_block_res_samples, mid_block_res_sample, raw_down_block_res_samples_ctlnet, raw_mid_block_sample_ctlnet = controlnet(
                        noisy_latents_img_c,
                        timesteps_attribute_c,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=noisy_latents_attr_c,
                        return_dict=False,
                    )
                
                    # Predict the noise residual
                    img_pred_c, raw_down_block_res_samples_unet, raw_mid_block_sample_unet, _ = unet(
                        noisy_latents_img_c,
                        timesteps_img_c,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                        return_dict=False
                    )
                    loss_c = F.mse_loss(img_pred_c.float(), img_target.float(), reduction="mean") 
                    # masks
                    loss = loss_img + loss_mask + 0.8 * loss_c  

                


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(controlnet.parameters()) + list(controldec.parameters()) + list(unet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                  
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
                            global_step, 
                            test_dataloader,
                            mask_guidance_scale=0,
                            img_guidance_scale=0,
                        )
                        
            logs = {"loss": loss.detach().item(), "loss_mask": loss_mask.detach().item(), "lr": optimizer.param_groups[0]['lr'], 'betas': optimizer.param_groups[0]['betas']}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(os.path.join(args.output_dir, "controlnet"))

        controldec = accelerator.unwrap_model(controldec)
        controldec.save_pretrained(os.path.join(args.output_dir, "controldec"))

        controldec = accelerator.unwrap_model(unet)
        controldec.save_pretrained(os.path.join(args.output_dir, "unet"))


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
