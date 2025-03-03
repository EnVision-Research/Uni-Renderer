
import imageio
# from pytorch_lightning import seed_everything
# from einops import rearrange, repeat


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
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.controlnet import ControlNetVAENoImgResOneCtlModel, _UnetDecControlModel, UNet2DConditionModel
from models.pipeline import StableDiffusionControl2BranchFtudecPipeline
from utils_metrics.inception import InceptionV3
from utils_metrics.calc_fid import calculate_frechet_distance, extract_features_from_samples
from utils_metrics.metrics_util import SegMetric, NormalMetric, calculate_miou_per_batch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import tempfile
from huggingface_hub import hf_hub_download
import gradio as gr

# if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
#     device0 = torch.device('cuda:0')
#     device1 = torch.device('cuda:1')
# else:
#     device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device1 = device0

# Define the cache directory for model files
# model_cache_dir = './ckpts/'
# os.makedirs(model_cache_dir, exist_ok=True)




###############################################################################
# Configuration.
###############################################################################



device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')

checkpoint = "./sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint), device=device)

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

def load_pipeline(pretrained_model_path, controlnet_model_path):
    """Load the diffusion pipeline and related models."""
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_path, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(pretrained_model_path, subfolder="text_encoder", revision=None)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,subfolder="tokenizer", revision=None, use_fast=False)
    unet = UNet2DConditionModel.from_pretrained(controlnet_model_path, subfolder="unet")
    controlnet = ControlNetVAENoImgResOneCtlModel.from_pretrained(controlnet_model_path, subfolder="controlnet")
    controldec = _UnetDecControlModel.from_pretrained(controlnet_model_path, subfolder="controldec")

    vae.to(device)
    unet.to(device)
    controlnet.to(device)
    controldec.to(device)

    return StableDiffusionControl2BranchFtudecPipeline.from_pretrained(
        pretrained_model_path,
        vae=vae, text_encoder=text_encoder, unet=unet,
        tokenizer=tokenizer, 
        controlnet=controlnet, controldec=controldec, 
        safety_checker=None
    )

# Load the model pipeline
pipeline = load_pipeline("/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4_x0", "/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/aa_even_more_data/checkpoint-354000")
pipeline.scheduler_img = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_attr = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_material = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_albedo = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_normal = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_spec_light = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_diff_light = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler_env = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

print('Loading Finished!')

click_points = []

def reset_clicks():
    """Reset stored click points."""
    global click_points
    click_points = []
    return "Clicks reset. Please select two points (top-left and bottom-right)."

def on_click(image, evt: gr.SelectData):
    """Capture click coordinates."""
    global click_points
    click_points.append((evt.index[0], evt.index[1]))  # Store (x, y) coordinates
    if len(click_points) == 2:
        return f"Selected Points: {click_points[0]} and {click_points[1]}"
    return "Click one more point."

def segment_image_with_sam(image):
    """Segment the selected area using SAM."""
    global click_points
    if len(click_points) != 2:
        return "Please select two points first."

    # Extract the coordinates
    (x_min, y_min), (x_max, y_max) = click_points

    # Use the SAM predictor to segment the selected area
    image_array = np.array(image)
    sam_predictor.set_image(image_array)

    # Predict the mask using the bounding box
    box = np.array([[x_min, y_min], [x_max, y_max]])
    masks, _, _ = sam_predictor.predict(box=box)

    # Convert the mask to a grayscale PIL image
    mask = Image.fromarray((masks[0] * 255).astype(np.uint8))
    
    # Reset click points for the next segmentation
    click_points = []

    return mask

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def run_diffusion_pipeline(image, mask, seed, num_steps):  # the mask here is already [0, 255]
    """Run the diffusion pipeline with the segmented mask."""

    metallic_list = []
    roughness_list = []
    compute_times = 5
    generator = torch.manual_seed(int(seed))
    prompts = " "
    mask_guidance_scale = 0.0
    mask_array = np.array(mask) / 255.0 
    resolution = 512

    for _ in range(compute_times):
        material, normal, albedo, specular, diffuse, env_pred = pipeline.real_image2mask_3mod_albedo(
            prompts,
            mask, 
            image,
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
        

    return metallic_image, roughness_image, normal, albedo, specular, diffuse





_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

Code: <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- Our demo can export a .obj mesh with vertex colors or a .glb mesh now. If you prefer to export a .obj mesh with a **texture map**, please refer to our <a href='https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#running-with-command-line' target='_blank'>Github Repo</a>.
- The 3D mesh generation results highly depend on the quality of generated multi-view images. Please try a different **seed value** if the result is unsatisfying (Default: 42).
'''

_CITE_ = r"""
If InstantMesh is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/InstantMesh?style=social)](https://github.com/TencentARC/InstantMesh)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""

# with gr.Blocks() as demo:
#     gr.Markdown(_HEADER_)
#     with gr.Row(variant="panel"):
#         with gr.Column():
#             with gr.Row():
#                 input_image = gr.Image(
#                     label="Upload Image",
#                     image_mode="RGBA",
#                     sources="upload",
#                     width=512,
#                     height=512,
#                     type="pil",
#                     elem_id="content_image",
#                 )
#                 processed_image = gr.Image(
#                     label="Processed Mask", 
#                     image_mode="L", 
#                     width=512,
#                     height=512,
#                     type="pil", 
#                     interactive=False
#                 )
#             with gr.Row():
#                 with gr.Group():
#                     # do_remove_background = gr.Checkbox(
#                     #     label="Remove Background", value=True
#                     # )
#                     sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

#                     sample_steps = gr.Slider(
#                         label="Sample Steps",
#                         minimum=30,
#                         maximum=75,
#                         value=75,
#                         step=5
#                     )

#             with gr.Row():
#                 submit = gr.Button("Generate", elem_id="generate", variant="primary")

#             with gr.Row(variant="panel"):
#                 gr.Examples(
#                     examples=[
#                         os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
#                     ],
#                     inputs=[input_image],
#                     label="Examples",
#                     examples_per_page=20
#                 )

#         with gr.Column():

#             with gr.Row():

#                 with gr.Column():
#                     mv_show_images = gr.Image(
#                         label="Generated Multi-views",
#                         type="pil",
#                         width=379,
#                         interactive=False
#                     )

#                 with gr.Column():
#                     output_video = gr.Video(
#                         label="video", format="mp4",
#                         width=379,
#                         autoplay=True,
#                         interactive=False
#                     )

#             with gr.Row():
#                 with gr.Tab("OBJ"):
#                     output_model_obj = gr.Model3D(
#                         label="Output Model (OBJ Format)",
#                         #width=768,
#                         interactive=False,
#                     )
#                     gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
#                 with gr.Tab("GLB"):
#                     output_model_glb = gr.Model3D(
#                         label="Output Model (GLB Format)",
#                         #width=768,
#                         interactive=False,
#                     )
#                     gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

#             with gr.Row():
#                 gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

#     gr.Markdown(_CITE_)
#     mv_images = gr.State()

#     submit.click(fn=check_input_image, inputs=[input_image]).success(
#         fn=preprocess,
#         inputs=[input_image, do_remove_background],
#         outputs=[processed_image],
#     ).success(
#         fn=generate_mvs,
#         inputs=[processed_image, sample_steps, sample_seed],
#         outputs=[mv_images, mv_show_images],
#     ).success(
#         fn=make3d,
#         inputs=[mv_images],
#         outputs=[output_video, output_model_obj, output_model_glb]
#     )
with gr.Blocks() as demo:
    gr.Markdown("# Combined SAM Segmentation and Diffusion Rendering with Seed Support")

    with gr.Row(variant="panel"):
        input_image = gr.Image(label="Upload Image", type="pil", interactive=True)
        #draw_rect = gr.Image(label="Draw Rectangle for Segmentation", type="pil")
        #input_image = gr.Image(label="Upload Image", type="pil").style(height=300)
        message = gr.Textbox(label="Message", interactive=False)
        segmented_mask = gr.Image(label="Segmented Mask", type="pil", interactive=False)


    with gr.Row():
        segmented_mask = gr.Image(label="Segmented Mask", type="pil", interactive=False)
        guidance_scale = gr.Slider(1.0, 20.0, step=0.5, value=7.5, label="Guidance Scale")
        num_steps = gr.Slider(10, 100, step=5, value=50, label="Inference Steps")
        sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

    with gr.Row():
        rendered_output = gr.Image(label="Rendered Image", interactive=False)
        metallic_output = gr.Image(label="Metallic Map", interactive=False)
        roughness_output = gr.Image(label="Roughness Map", interactive=False)
        normal_output = gr.Image(label="Normal Map", interactive=False)
        albedo_output = gr.Image(label="Albedo Map", interactive=False)
        specular_output = gr.Image(label="Specular Light", interactive=False)
        diffuse_output = gr.Image(label="Diffuse Light", interactive=False)

    with gr.Row():
        click_button = gr.Button("Reset Clicks")
        segment_button = gr.Button("Segment Selected Area")
        run_button = gr.Button("Run Diffusion Rendering")

    # Callbacks
    input_image.select(on_click, inputs=[input_image], outputs=[message])
    click_button.click(fn=reset_clicks, outputs=[message])
    segment_button.click(fn=segment_image_with_sam, inputs=[input_image], outputs=[segmented_mask])


    run_button.click(
        fn=run_diffusion_pipeline,
        inputs=[input_image, segmented_mask, guidance_scale, num_steps, sample_seed],
        outputs=[
            rendered_output, metallic_output, roughness_output, 
            normal_output, albedo_output, specular_output, diffuse_output
        ]
    )

if __name__ == "__main__":
    # Enable queueing with a maximum queue size of 10
    demo.queue(max_size=10)

    # Launch the demo on a specific IP address and port
    demo.launch(
        server_name="127.0.0.1",  # Makes it accessible externally
        server_port=5000  # Use the specified port
    )
