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

from dataset.blendGen import BlenderGenDataset, BlenderGenDataset_old, BlenderGenDataset_3mod, BlenderGenDataset_3mod_old
# from utils_metrics.compute_t import compute_t
from models.controlnet import ControlNetVAENoImgResOneCtlModel, _UnetDecControlModel, UNet2DConditionModel
from models.pipeline import StableDiffusionControl2BranchFtudecPipeline
from utils_metrics.inception import InceptionV3
from utils_metrics.calc_fid import calculate_frechet_distance, extract_features_from_samples
from utils_metrics.metrics_util import SegMetric, NormalMetric, calculate_miou_per_batch

# if is_wandb_available():
#     import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")


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


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, 
    controldec, args, accelerator, weight_dtype, step, img_guidance_scale=7.5, mask_guidance_scale=7.5):
    logger.info("Running validation... ")
    
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
    pipeline.scheduler_normal = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler_light = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
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
    image_logs = []

    # validation_img = Image.open(args.validation_image[0]).convert("RGB").resize((args.resolution, args.resolution))
    # material_image = np.load(args.validation_image[1])
    # normal_image = Image.open(args.validation_image[2]).convert("RGB").resize((args.resolution, args.resolution))
    light_image = Image.open(
        "/hpc2hdd/home/zchen379/working/abandoned_factory_canteen_01_2k.png"
        ).convert("RGB").resize((args.resolution, args.resolution))

    test_data_path = args.validation_image[0]
    images, real_images = [], []
    metallics, roughnesses, masks, normals, lights = [], [], [], [], []
    for index in range(args.num_validation_images):
        validation_img = Image.open(
            os.path.join(test_data_path, f'rgb_{index}.png')
            ).convert("RGB").resize((args.resolution, args.resolution))
        material_image = np.load(os.path.join(test_data_path, f'material_{index}.npy'))
        normal_image = Image.open(
            os.path.join(test_data_path, f'normal_{index}.png')
            ).convert("RGB").resize((args.resolution, args.resolution))
        # light_image = Image.open(
        #     os.path.join(test_data_path, f'env_{index}.png')
        #     ).convert("RGB").resize((args.resolution, args.resolution))
        with torch.autocast("cuda"):
            material, normal, light = pipeline.image2mask_3mod(
                [""],
                validation_img, 
                guidance_scale=mask_guidance_scale,
                height=args.resolution, width=args.resolution, num_inference_steps=50, generator=generator
            )
            
            metallic, roughness, mask = material[0].split()
            metallic_3_channel = Image.merge("RGB", (metallic, metallic, metallic))
            roughness_3_channel = Image.merge("RGB", (roughness, roughness, roughness))
            mask_3_channel = Image.merge("RGB", (mask, mask, mask))

            rgb_image = pipeline.mask2image_3mod(
                [""],
                material_image,
                normal_image, 
                light_image,
                guidance_scale=img_guidance_scale,
                height=args.resolution, width=args.resolution, num_inference_steps=50, generator=generator
            )

        metallics.append(metallic_3_channel)
        roughnesses.append(roughness_3_channel)
        masks.append(mask_3_channel)
        normals.append(normal[0])
        lights.append(light[0])
        images.append(rgb_image[0])
        real_images.append(validation_img)

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
    log_dict = {
            "metallics": metallics, 
            "roughnesses": roughnesses,
            "masks": masks,
            "normals": normals,
            "lights": lights,
            "images": images,
            "real_images": real_images
            }
    # ---------- fid ------------
    # if args.fid:
    #     logging.info(f"Calculating FID score for generated images and masks ... ")
    #     logging.info(f"Generating {args.num_images_fid} samples...") 
    #     fid_img, _ = calculate_fid(pipeline, '', args.resolution, args.resolution, args.num_images_fid, 20, generator, accelerator, real_path=args.real_image_path, step=step, output_dir=args.output_dir, batch_size=args.fid_batch_size)
    
    # ---------- mIoU -----------
    # load test dataset: test images and mask gt
    # sampling masks for every test images
    # map rgb mask to label


    # calculate mIoU
    # if args.miou and step >= 5000:
    #     logging.info(f"Calculating mIoU score for image2mask... ")
    #     logging.info(f"  Instantaneous batch size = {args.test_batch_size}")
    #     logging.info(f"  Num batches each testing epoch = {len(test_dataloader)}")


    #     progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)

    #     for _ , batch in enumerate(test_dataloader):
    #         test_images, gt_labels = batch
    #         #print(test_images.shape)
    #         gt_masks = gt_labels["segmentation"]
    #         gt_normals = gt_labels['normal']
    #         rgb_masks, _, pred_normals= pipeline.image2mask(
    #             [""],
    #             test_images, # must be PIL.Image or Tensor
    #             height=args.resolution, width=args.resolution, num_inference_steps=50, generator=generator
    #         )
    #         pred_masks = torch.stack([torch.tensor(test_dataset.map_colors_to_index(m)) for m in rgb_masks]).to(gt_masks.device)
    #         #print(pred_normals.shape)
    #         #pred_normals = torch.stack(rbg_normals).to(gt_normals.device)
    #         #print(normals.shape, gt_normals.shape)
    #         segmetric.update_fun(pred_masks, gt_masks+1)
    #         normalmetric.update_fun(pred_normals, gt_normals)
    #         progress_bar.update(1)
    #     # overall_miou = miou_sum / batch_count if batch_count > 0 else 0


    #     # if args.fid:
    #     #     log_dict["fid_img"] = fid_img
    #     metrics_out = os.path.join(args.output_dir, 'metrics.txt')
    #     log_dict["miou"] = segmetric.score_fun()[0]
    #     log_dict["seg_acc"] = segmetric.score_fun()[1]
    #     log_dict["normal_acc"] = normalmetric.score_fun()
    #     with open(metrics_out, 'w') as f:   
    #         print("mIoU", log_dict["miou"], file=f)
    #         print("Acc: ", log_dict["seg_acc"], file=f)
    #         print("Normal Acc: ", log_dict["normal_acc"], file=f)

    image_logs.append(log_dict)
    
    save_img_dir = os.path.join(args.output_dir, 'imgs')
    os.makedirs(save_img_dir, exist_ok=True)
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)

                images = log["real_images"]
                formatted_real_images = []
                for image in images:
                    formatted_real_images.append(np.asarray(image))
                formatted_real_images = np.stack(formatted_real_images)

                #print(123
                metallics = log["metallics"]
                formatted_metallics = []
                for metallic in metallics:
                    formatted_metallics.append(np.asarray(metallic))
                formatted_metallics = np.stack(formatted_metallics)

                roughnesses = log["roughnesses"]
                formatted_roughnesses = []
                for roughness in roughnesses:
                    formatted_roughnesses.append(np.asarray(roughness))
                formatted_roughnesses = np.stack(formatted_roughnesses)

                masks = log["masks"]
                formatted_masks = []
                for mask in masks:
                    formatted_masks.append(np.asarray(mask))
                formatted_masks = np.stack(formatted_masks)
                
                normals = log["normals"]
                formatted_normals= []
                for normal in normals:
                    formatted_normals.append(np.asarray(normal))
                formatted_normals = np.stack(formatted_normals)

                lights = log["lights"]
                formatted_lights= []
                for light in lights:
                    formatted_lights.append(np.asarray(light))
                formatted_lights = np.stack(formatted_lights)

                formatted_out = np.concatenate(
                    [formatted_metallics, formatted_roughnesses, formatted_masks, 
                     formatted_normals, formatted_lights, formatted_images, formatted_real_images], axis=1)

                tracker.writer.add_images(f"validation", formatted_out, step, dataformats="NHWC")
                
                n,h,w,c = formatted_out.shape
                formatted_out = formatted_out.transpose(1,0,2,3).reshape(h,n*w, c)
                im = Image.fromarray(formatted_out)
                
                im.save(f'{save_img_dir}/{str(step)}_imcfg{img_guidance_scale}_attrcfg{mask_guidance_scale}.png')

                # if args.fid:
                #     tracker.writer.add_scalar("FID_image", log["fid_img"], step)
                #     # tracker.writer.add_scalar("FID/segmentation", fid_seg, step)
                # if args.miou and step >= 15000:
                #     tracker.writer.add_scalar("mIoU", log["miou"], step)
                #     tracker.writer.add_scalar("Acc", log["seg_acc"], step)
                #     for i in log["normal_acc"]:
                #         tracker.writer.add_scalars('Normal',, step)
                #     tracker.writer.add_scalar("Normal", log["normal_acc"], step)
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    

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

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

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

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)



    task_list = ['img', 'material_image', 'normal', 'light'] #TODO: depth has 1 channel, how to pass it to vae that requires 3 channel input

    #train_dataset = BlenderGenDataset_old(root_dir=args.train_data_dir, mode='train', transform=transforms, resize=(256, 256))
    train_dataset = BlenderGenDataset_3mod_old(root_dir=args.train_data_dir, resize=(args.resolution, args.resolution))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)


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
        controlnet = ControlNetVAENoImgResOneCtlModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controlnet")
        controldec = _UnetDecControlModel.from_pretrained(args.controlnet_model_name_or_path, subfolder="controldec")
    else:
        logger.info("Initializing controlnet and controldec weights from unet")
        controlnet = ControlNetVAENoImgResOneCtlModel.from_unet(unet, len_t=len(task_list)-1) #zhifei exclude img as we dont pass it to controlnet
        controldec = _UnetDecControlModel.from_unet(unet, len_t=len(task_list)-1)
   
    controlnet.conv_in.weight = nn.Parameter(controlnet.conv_in.weight.repeat(1, 3, 1, 1) * 0.33)
    #controlnet.time_embedding.linear_1.weight = nn.Parameter(controlnet.time_embedding.linear_1.weight.repeat(1, 3) * 0.33)

    # replace config
    config_dict = {}
    for k,v in controlnet._internal_dict.items():
        config_dict[k] = v
    config_dict = EasyDict(config_dict)
    controlnet._internal_dict = config_dict

    controlnet.config["in_channels"] = 12

    # Replace the last layer to output 8 out_channels. 
    controldec.conv_out.weight = nn.Parameter(controldec.conv_out.weight.repeat(3, 1, 1, 1) * 0.33)
    controldec.conv_out.bias = nn.Parameter(controldec.conv_out.bias.repeat(3) * 0.33)
    #controldec.time_embedding.linear_1.weight = nn.Parameter(controldec.time_embedding.linear_1.weight.repeat(1, 3) * 0.33)
    config_dict = {}
    for k,v in controldec._internal_dict.items():
        config_dict[k] = v
    config_dict = EasyDict(config_dict)
    controlnet._internal_dict = config_dict
    controldec.config["out_channels"] = 12


    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    
                    if model.__class__.__name__ == 'ControlNetVAENoImgResOneCtlModel':
                        sub_dir = "controlnet"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    elif model.__class__.__name__ == '_UnetDecControlModel':
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

                if model.__class__.__name__ == 'ControlNetVAENoImgResOneCtlModel':
                    subfolder = "controlnet"
                    # load diffusers style into model
                    load_model = ControlNetVAENoImgResOneCtlModel.from_pretrained(input_dir, subfolder=subfolder)
                
                elif model.__class__.__name__ == '_UnetDecControlModel':
                    subfolder = "controldec"
                    load_model = _UnetDecControlModel.from_pretrained(input_dir, subfolder=subfolder)

                elif model.__class__.__name__ == 'UNet2DConditionModel':
                    subfolder = "unet"
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=subfolder)        

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    controldec.train()

    for _name, _module in unet.named_modules():
        if 'up_blocks' in _name: _module.requires_grad_()

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

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    #     num_training_steps=args.max_train_steps * accelerator.num_processes,
    #     num_cycles=args.lr_num_cycles,
    #     power=args.lr_power,
    # )

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

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, controldec, unet):
                # RGBA
                images = batch['rgb']
                # Attributes
                light_image = batch['light']
                material_image = batch['material']
                normal_image = batch['normal']
                prompts = ' '

                bsz = images.shape[0]

                #prepare time step for smaller tasks space
                timesteps = compute_t(len_t=2, 
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


                #### normal ####
                latents_normal = vae.encode(normal_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_normal = latents_normal * vae.config.scaling_factor

                 #### light ####
                latents_light = vae.encode(light_image.to(dtype=weight_dtype)).latent_dist.sample()
                latents_light = latents_light * vae.config.scaling_factor
                
                latents_attr = torch.cat((latents_material, latents_normal, latents_light), dim=1)
                noise_attr = torch.randn_like(latents_attr)
                noisy_latents_attr = noise_scheduler.add_noise(latents_attr, noise_attr, timesteps_attribute)

                #### concat label ####
                #breakpoint()
                # timestep_label_concat = torch.cat((timesteps_metallic.unsqueeze(0), timesteps_roughness.unsqueeze(0), timesteps_normal.unsqueeze(0)), dim=0)
                #timestep_label_concat = torch.cat((timesteps_mater.unsqueeze(0), timesteps_normal.unsqueeze(0), timesteps_light.unsqueeze(0)), dim=0)
                #noise_concat = torch.cat((noise_material, noise_normal, noise_light), dim=1)
                #noisy_latents_concat = torch.cat((noisy_latents_material, noisy_latents_normal, noisy_latents_light), dim=1)
                
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

                img_target = noise_scheduler.get_velocity(latents_img, noise_img, timesteps_img)
                attr_target = noise_scheduler.get_velocity(latents_attr, noise_attr, timesteps_attribute) 


                loss_img = F.mse_loss(img_pred.float(), img_target.float(), reduction="mean")  # tianshuo
                loss_mask = F.mse_loss(mask_pred.float(), attr_target.float(), reduction="mean") 
                #loss_alignment = F.mse_loss(latents_mask_x_0.float(), img_target.float(), reduction='mean')
                #loss = loss_img + loss_mask + 0.1 * loss_alignment
                # breakpoint()
                loss = loss_img + loss_mask  # tianshuo
                #loss = loss_mask

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(controlnet.parameters()) + list(controldec.parameters()) + list(unet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                # lr_scheduler.step()
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
                            img_guidance_scale=0, 
                            mask_guidance_scale=0,
                        )
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
                            img_guidance_scale=2.5, 
                            mask_guidance_scale=10,
                        )
            # logs = {"loss": loss.detach().item(), "loss_img": loss_img.detach().item(), "loss_mask": loss_mask.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

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
