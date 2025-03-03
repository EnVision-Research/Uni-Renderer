import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from zf_alchemist.diffusion_4_perception.models.controlnet_old import ControlNetModel, ControlDecModel, UNet2DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionControl2BranchFtudecUni_only_seg_Pipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        controldec: ControlDecModel, 
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )


        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            controldec=controldec,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    # def prepare_extra_step_kwargs(self, generator, eta):
    #     # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    #     # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    #     # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    #     # and should be between [0, 1]

    #     accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    #     extra_step_kwargs = {}
    #     if accepts_eta:
    #         extra_step_kwargs["eta"] = eta

    #     # check if the scheduler accepts generator
    #     accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
    #     if accepts_generator:
    #         extra_step_kwargs["generator"] = generator
    #     return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler_img.init_noise_sigma
        return latents
    
    def prepare_latents_mask(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler_mask.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def image2mask(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):


        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # controldec= self.controldec._orig_mod if is_compiled_module(self.controldec) else self.controldec

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
                
        # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler_img, num_inference_steps, device)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler_mask, num_inference_steps, device)
        self._num_timesteps = len(timesteps)

        timesteps_img = torch.zeros_like(timesteps)

        # 6. Prepare latent variables
        if not isinstance(image, torch.Tensor): # hejing
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size*num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                guess_mode=guess_mode,
            )
        else:
            assert batch_size == 1
            batch_size = image.shape[0]

            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        
        weight_dtype=self.vae.dtype
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        latents_img = self.vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
        latents_img = latents_img * self.vae.config.scaling_factor

        num_channels_latents_mask = self.unet.config.in_channels * 3
        latents_mask = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_mask,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler_img.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        latent_model_input_img = torch.cat([latents_img] * 2) if self.do_classifier_free_guidance else latents_img
        latent_model_input_img = self.scheduler_img.scale_model_input(latent_model_input_img, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):
                t_mask = timesteps[i]
                t_img = timesteps_img[i]
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                
                latent_model_input_mask = torch.cat([latents_mask] * 2) if self.do_classifier_free_guidance else latents_mask
                latent_model_input_mask = self.scheduler_mask.scale_model_input(latent_model_input_mask, t_mask)

                # controlnet(s) inference

                control_model_input = latent_model_input_img
                controlnet_prompt_embeds = prompt_embeds
                cond_scale = controlnet_conditioning_scale

                down_block_res_samples, mid_block_res_sample, raw_down_block_res_samples_ctlnet, raw_mid_block_sample_ctlnet = self.controlnet(
                    control_model_input,
                    t_mask,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=latent_model_input_mask,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )

                # predict the noise residual
                _, raw_down_block_res_samples_unet, raw_mid_block_sample_unet, up_block_res_samples_unet = self.unet(
                    latent_model_input_img,
                    t_img,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ], # hejing
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )
                
                mask_pred = self.controldec(
                  sample = raw_mid_block_sample_ctlnet,
                  down_block_res_samples = raw_down_block_res_samples_ctlnet, 
                  timestep = t_mask,
                  encoder_hidden_states=prompt_embeds,
                  down_block_additional_residuals=raw_down_block_res_samples_unet,
                  #up_block_additional_residuals=up_block_res_samples_unet,
                  mid_block_additional_residual=raw_mid_block_sample_unet,
                  return_dict = False
                )
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    # img_pred_uncond, img_pred_text = img_pred.chunk(2)
                    # img_pred = img_pred_uncond + self.guidance_scale * (img_pred_text - img_pred_uncond)

                    mask_pred_uncond, mask_pred_text = mask_pred.chunk(2)
                    mask_pred = mask_pred_uncond + self.guidance_scale * (mask_pred_text - mask_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents_img = self.scheduler_img.step(img_pred, t, latents_img, return_dict=False)[0]
                latents_mask = self.scheduler_mask.step(mask_pred, t_mask, latents_mask, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler_img.order == 0):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            self.controldec.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            # image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            #     0
            # ]
            # image, has_nsfw_concept_img = self.run_safety_checker(image, device, prompt_embeds.dtype)
            C = 4
            latents_metallic, latents_roughness, latents_normal = torch.split(latents_mask / self.vae.config.scaling_factor, C, dim=1)
            
            mask_metallic = self.vae.decode(latents_metallic, return_dict=False, generator=generator)[0]
            mask_roughness = self.vae.decode(latents_roughness, return_dict=False, generator=generator)[0]
            mask_normal = self.vae.decode(latents_normal, return_dict=False, generator=generator)[0]

            mask_metallic, has_nsfw_concept_mask = self.run_safety_checker(mask_metallic, device, prompt_embeds.dtype)
            mask_roughness, _ = self.run_safety_checker(mask_roughness, device, prompt_embeds.dtype)
            mask_normal, _ = self.run_safety_checker(mask_normal, device, prompt_embeds.dtype)
        else:
            # image = latents_img
            # has_nsfw_concept_img = None

            # mask = latents_mask
            # has_nsfw_concept_mask = None
            raise ValueError

        # if has_nsfw_concept_img is None:
        #     do_denormalize_img = [True] * image.shape[0]
        # else:
        #     do_denormalize_img = [not has_nsfw for has_nsfw in has_nsfw_concept_img]
        do_denormalize_mask = [True] * mask_metallic.shape[0]
        # if has_nsfw_concept_mask is None:
        #     do_denormalize_mask = [True] * mask_metallic.shape[0]
        # else:
        #     do_denormalize_mask = [not has_nsfw for has_nsfw in has_nsfw_concept_mask]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize_img)
        mask_metallic = self.control_image_processor.postprocess(mask_metallic, output_type=output_type, do_denormalize=do_denormalize_mask)
        mask_roughness = self.control_image_processor.postprocess(mask_roughness, output_type=output_type, do_denormalize=do_denormalize_mask)
        mask_normal = self.control_image_processor.postprocess(mask_normal, output_type=output_type, do_denormalize=do_denormalize_mask)

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return mask_metallic, mask_roughness, mask_normal

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad()
    def mask2image(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):


        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # controldec= self.controldec._orig_mod if is_compiled_module(self.controldec) else self.controldec

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler_img, num_inference_steps, device)
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler_mask, num_inference_steps, device)
        self._num_timesteps = len(timesteps)

        timesteps_mask = torch.zeros_like(timesteps)

        # 6. Prepare latent variables
        mask = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size*num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            guess_mode=guess_mode,
        )
        latents_mask = self.vae.encode(mask).latent_dist.sample()
        latents_mask = latents_mask * self.vae.config.scaling_factor

        num_channels_latents_mask = self.unet.config.in_channels
        latents_img = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_mask,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler_img.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        latent_model_input_mask = torch.cat([latents_mask] * 2) if self.do_classifier_free_guidance else latents_mask
        latent_model_input_mask = self.scheduler_img.scale_model_input(latent_model_input_mask, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):
                t_img = timesteps[i]
                t_mask = timesteps_mask[i]
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                
                latent_model_input_img = torch.cat([latents_img] * 2) if self.do_classifier_free_guidance else latents_img
                latent_model_input_img = self.scheduler_mask.scale_model_input(latent_model_input_img, t_img)

                # controlnet(s) inference

                control_model_input = latent_model_input_img
                controlnet_prompt_embeds = prompt_embeds
                cond_scale = controlnet_conditioning_scale

                down_block_res_samples, mid_block_res_sample, _, _ = self.controlnet(
                    control_model_input,
                    t_mask,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=latent_model_input_mask,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )

                # predict the noise residual
                img_pred, _, _, _ = self.unet(
                    latent_model_input_img,
                    t_img,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )
                
                # mask_pred = self.controldec(
                #   sample = raw_mid_block_sample_ctlnet,
                #   down_block_res_samples = raw_down_block_res_samples_ctlnet, 
                #   timestep = t_mask,
                #   encoder_hidden_states=prompt_embeds,
                #   down_block_additional_residuals=raw_down_block_res_samples_unet,
                #   mid_block_additional_residual=raw_mid_block_sample_unet,
                #   return_dict = False
                # )
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    img_pred_uncond, img_pred_text = img_pred.chunk(2)
                    img_pred = img_pred_uncond + self.guidance_scale * (img_pred_text - img_pred_uncond)

                    # mask_pred_uncond, mask_pred_text = mask_pred.chunk(2)
                    # mask_pred = mask_pred_uncond + self.guidance_scale * (mask_pred_text - mask_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_img = self.scheduler_img.step(img_pred, t_img, latents_img, return_dict=False)[0]
                # latents_mask = self.scheduler_mask.step(mask_pred, t_mask, latents_mask, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler_img.order == 0):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            self.controldec.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept_img = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # mask = self.vae.decode(latents_mask / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            #     0
            # ]
            # mask, has_nsfw_concept_mask = self.run_safety_checker(mask, device, prompt_embeds.dtype)
        else:
            image = latents_img
            has_nsfw_concept_img = None

            # mask = latents_mask
            # has_nsfw_concept_mask = None

        if has_nsfw_concept_img is None:
            do_denormalize_img = [True] * image.shape[0]
        else:
            do_denormalize_img = [not has_nsfw for has_nsfw in has_nsfw_concept_img]
        
        # if has_nsfw_concept_mask is None:
        #     do_denormalize_mask = [True] * mask.shape[0]
        # else:
        #     do_denormalize_mask = [not has_nsfw for has_nsfw in has_nsfw_concept_mask]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize_img)
        # mask = self.control_image_processor.postprocess(mask, output_type=output_type, do_denormalize=do_denormalize_mask)

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return image

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad()
    def joint_sample(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        # controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # controldec= self.controldec._orig_mod if is_compiled_module(self.controldec) else self.controldec

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler_img, num_inference_steps, device)
        _, _ = retrieve_timesteps(self.scheduler_mask, num_inference_steps, device)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents_img = self.unet.config.in_channels
        latents_img = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_img,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        num_channels_latents_mask = self.unet.config.in_channels
        latents_mask = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_mask,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler_img.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input_img = torch.cat([latents_img] * 2) if self.do_classifier_free_guidance else latents_img
                latent_model_input_img = self.scheduler_img.scale_model_input(latent_model_input_img, t)

                latent_model_input_mask = torch.cat([latents_mask] * 2) if self.do_classifier_free_guidance else latents_mask
                latent_model_input_mask = self.scheduler_mask.scale_model_input(latent_model_input_mask, t)

                # controlnet(s) inference

                control_model_input = latent_model_input_img
                controlnet_prompt_embeds = prompt_embeds
                cond_scale = controlnet_conditioning_scale

                down_block_res_samples, mid_block_res_sample, raw_down_block_res_samples_ctlnet, raw_mid_block_sample_ctlnet = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=latent_model_input_mask,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )

                # predict the noise residual
                img_pred, raw_down_block_res_samples_unet, raw_mid_block_sample_unet, up_block_res_samples_unet = self.unet(
                    latent_model_input_img,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )
                
                mask_pred = self.controldec(
                  sample = raw_mid_block_sample_ctlnet,
                  down_block_res_samples = raw_down_block_res_samples_ctlnet, 
                  timestep = t,
                  encoder_hidden_states=prompt_embeds,
                  down_block_additional_residuals=raw_down_block_res_samples_unet,
                up_block_additional_residuals=up_block_res_samples_unet,
                  mid_block_additional_residual=raw_mid_block_sample_unet,
                  return_dict = False
                )
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    img_pred_uncond, img_pred_text = img_pred.chunk(2)
                    img_pred = img_pred_uncond + self.guidance_scale * (img_pred_text - img_pred_uncond)

                    mask_pred_uncond, mask_pred_text = mask_pred.chunk(2)
                    mask_pred = mask_pred_uncond + self.guidance_scale * (mask_pred_text - mask_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_img = self.scheduler_img.step(img_pred, t, latents_img, return_dict=False)[0]
                latents_mask = self.scheduler_mask.step(mask_pred, t, latents_mask, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler_img.order == 0):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            self.controldec.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept_img = self.run_safety_checker(image, device, prompt_embeds.dtype)

            mask = self.vae.decode(latents_mask / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            mask, has_nsfw_concept_mask = self.run_safety_checker(mask, device, prompt_embeds.dtype)
        else:
            image = latents_img
            has_nsfw_concept_img = None

            mask = latents_mask
            has_nsfw_concept_mask = None

        if has_nsfw_concept_img is None:
            do_denormalize_img = [True] * image.shape[0]
        else:
            do_denormalize_img = [not has_nsfw for has_nsfw in has_nsfw_concept_img]
        
        if has_nsfw_concept_mask is None:
            do_denormalize_mask = [True] * mask.shape[0]
        else:
            do_denormalize_mask = [not has_nsfw for has_nsfw in has_nsfw_concept_mask]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize_img)
        mask = self.control_image_processor.postprocess(mask, output_type=output_type, do_denormalize=do_denormalize_mask)

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return (image, mask)


class StableDiffusionControl2BranchFtudecPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        controldec: ControlDecModel, 
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )


        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            controldec=controldec,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    # def prepare_extra_step_kwargs(self, generator, eta):
    #     # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    #     # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    #     # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    #     # and should be between [0, 1]

    #     accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    #     extra_step_kwargs = {}
    #     if accepts_eta:
    #         extra_step_kwargs["eta"] = eta

    #     # check if the scheduler accepts generator
    #     accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
    #     if accepts_generator:
    #         extra_step_kwargs["generator"] = generator
    #     return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler_img.init_noise_sigma
        return latents
    
    def prepare_latents_mask(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler_mask.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @torch.no_grad()
    def image2mask(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 2.5, #zhifei
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):


        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # controldec= self.controldec._orig_mod if is_compiled_module(self.controldec) else self.controldec

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 4. Prepare timesteps
        #timesteps_img, num_inference_steps = retrieve_timesteps(self.scheduler_img, num_inference_steps, device)
        timesteps_mask, num_inference_steps = retrieve_timesteps(self.scheduler_mask, num_inference_steps, device)
        timesteps_normal, num_inference_steps = retrieve_timesteps(self.scheduler_normal, num_inference_steps, device)
        self._num_timesteps = len(timesteps_mask)

        timesteps_img = torch.zeros_like(timesteps_mask)

        # 6. Prepare latent variables
        if not isinstance(image, torch.Tensor): # hejing
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size*num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                guess_mode=guess_mode,
            )
        else:
            assert batch_size == 1
            batch_size = image.shape[0]

            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            
        weight_dtype= self.vae.dtype
        latents_img = self.vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
        latents_img = latents_img * self.vae.config.scaling_factor

        num_channels_latents_mask = self.unet.config.in_channels # zhifei
        latents_mask = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_mask,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        num_channels_latents_normal = self.unet.config.in_channels  # zhifei
        latents_normal = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_normal,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps_img) - num_inference_steps * self.scheduler_img.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        latent_model_input_img = torch.cat([latents_img] * 2) if self.do_classifier_free_guidance else latents_img
        latent_model_input_img = self.scheduler_img.scale_model_input(latent_model_input_img, 0)

        

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            #for i, (t_mask, t_normal) in enumerate(zip(timesteps_mask, timesteps_normal)):
            for i, (t_img, t_mask, t_normal) in enumerate(zip(timesteps_img, timesteps_mask, timesteps_normal)):
                #print(t_img, t_mask, t_normal)
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                
                latent_model_input_mask = torch.cat([latents_mask] * 2) if self.do_classifier_free_guidance else latents_mask
                latent_model_input_mask = self.scheduler_mask.scale_model_input(latent_model_input_mask, t_mask)

                latent_model_input_normal = torch.cat([latents_normal] * 2) if self.do_classifier_free_guidance else latents_normal
                latent_model_input_normal = self.scheduler_normal.scale_model_input(latent_model_input_normal, t_normal)

                latent_model_input_concat = torch.cat((latent_model_input_mask, latent_model_input_normal), dim=1) #zhifei

                # controlnet(s) inference

                control_model_input = latent_model_input_img
                controlnet_prompt_embeds = prompt_embeds
                cond_scale = controlnet_conditioning_scale
                #print(t_mask.shape, t_mask.view(1,1).shape)
                t_label = torch.cat((t_mask.view(1,1), t_normal.view(1,1)), dim=0) #zhifei [2,1]
                #print(t_label)
                down_block_res_samples, mid_block_res_sample, raw_down_block_res_samples_ctlnet, raw_mid_block_sample_ctlnet = self.controlnet(
                    control_model_input,
                    t_label,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=latent_model_input_concat, #zhifei
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )

                # predict the noise residual
                _, raw_down_block_res_samples_unet, raw_mid_block_sample_unet, _ = self.unet(
                    latent_model_input_img,
                    t_img,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ], # hejing
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )
                
                label_pred = self.controldec(
                  sample = raw_mid_block_sample_ctlnet,
                  down_block_res_samples = raw_down_block_res_samples_ctlnet, 
                  timestep = t_label,
                  encoder_hidden_states=prompt_embeds,
                  down_block_additional_residuals=raw_down_block_res_samples_unet,
                  mid_block_additional_residual=raw_mid_block_sample_unet,
                  return_dict = False
                )
                mask_pred, normal_pred = label_pred[:, :4, :, :], label_pred[:, 4:, :, :]
                # perform guidance
                if self.do_classifier_free_guidance:
                    # img_pred_uncond, img_pred_text = img_pred.chunk(2)
                    # img_pred = img_pred_uncond + self.guidance_scale * (img_pred_text - img_pred_uncond)

                    mask_pred_uncond, mask_pred_text = mask_pred.chunk(2)
                    mask_pred = mask_pred_uncond + self.guidance_scale * (mask_pred_text - mask_pred_uncond)

                    normal_pred_uncond, normal_pred_text = normal_pred.chunk(2)
                    normal_pred = normal_pred_uncond + self.guidance_scale * (normal_pred_text - normal_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents_img = self.scheduler_img.step(img_pred, t, latents_img, return_dict=False)[0]
                latents_mask = self.scheduler_mask.step(mask_pred, t_mask, latents_mask, return_dict=False)[0]
                latents_normal = self.scheduler_normal.step(normal_pred, t_normal, latents_normal, return_dict=False)[0]

                if i == len(timesteps_img) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler_img.order == 0):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            self.controldec.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":

            first_label_latents = latents_mask # zhifei
            second_label_latents = latents_normal

            label_1 = self.vae.decode(first_label_latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            label_2 = self.vae.decode(second_label_latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            mask_1, has_nsfw_concept_mask_1 = self.run_safety_checker(label_1, device, prompt_embeds.dtype)
            mask_2, has_nsfw_concept_mask_2 = self.run_safety_checker(label_2, device, prompt_embeds.dtype)
            normal_for_metrics = mask_2
        else:

            mask_1 = latents_mask
            mask_2 = latents_normal #zhifei
            has_nsfw_concept_mask_1 = None
            has_nsfw_concept_mask_2 = None
        
        if has_nsfw_concept_mask_1 is None:
            do_denormalize_mask_1 = [True] * mask_1.shape[0]
        else:
            do_denormalize_mask_1 = [not has_nsfw for has_nsfw in has_nsfw_concept_mask_1]#zhifei

        if has_nsfw_concept_mask_2 is None:  #zhifei
            do_denormalize_mask_2 = [True] * mask_2.shape[0]
        else:
            do_denormalize_mask_2 = [not has_nsfw for has_nsfw in has_nsfw_concept_mask_2]

        mask_1 = self.control_image_processor.postprocess(mask_1, output_type=output_type, do_denormalize=do_denormalize_mask_1)
        mask_2 = self.control_image_processor.postprocess(mask_2, output_type=output_type, do_denormalize=do_denormalize_mask_2)#zhifei


        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return (mask_1, mask_2, normal_for_metrics)

    @torch.no_grad()
    def joint_sample(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):


        # controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        # controldec= self.controldec._orig_mod if is_compiled_module(self.controldec) else self.controldec

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps_img, num_inference_steps = retrieve_timesteps(self.scheduler_img, num_inference_steps, device)
        timesteps_mask, num_inference_steps = retrieve_timesteps(self.scheduler_mask, num_inference_steps, device)
        timesteps_normal, num_inference_steps = retrieve_timesteps(self.scheduler_normal, num_inference_steps, device)
        self._num_timesteps = len(timesteps_img)

        # 6. Prepare latent variables
        num_channels_latents_img = self.unet.config.in_channels
        latents_img = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_img,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        num_channels_latents_mask = self.unet.config.in_channels #zhifei
        latents_mask = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_mask,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        num_channels_latents_normal = self.unet.config.in_channels #zhifei
        latents_normal = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_normal,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps_img) - num_inference_steps * self.scheduler_img.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t_img, t_mask, t_normal) in enumerate(zip(timesteps_img, timesteps_mask, timesteps_normal)):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input_img = torch.cat([latents_img] * 2) if self.do_classifier_free_guidance else latents_img
                latent_model_input_img = self.scheduler_img.scale_model_input(latent_model_input_img, t_img)

                latent_model_input_mask = torch.cat([latents_mask] * 2) if self.do_classifier_free_guidance else latents_mask
                latent_model_input_mask = self.scheduler_mask.scale_model_input(latent_model_input_mask, t_mask)

                latent_model_input_normal = torch.cat([latents_normal] * 2) if self.do_classifier_free_guidance else latents_normal
                latent_model_input_normal = self.scheduler_normal.scale_model_input(latent_model_input_normal, t_normal)

                # controlnet(s) inference

                control_model_input = latent_model_input_img
                controlnet_prompt_embeds = prompt_embeds
                cond_scale = controlnet_conditioning_scale

                t_label = torch.cat((t_mask.view(1,1), t_normal.view(1,1)), dim=0) #zhifei [2,1]
                latent_model_input_concat = torch.cat((latent_model_input_mask, latent_model_input_normal), dim=1) #zhifei
                #print()
                down_block_res_samples, mid_block_res_sample, raw_down_block_res_samples_ctlnet, raw_mid_block_sample_ctlnet = self.controlnet(
                    control_model_input,
                    t_label,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=latent_model_input_concat,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )
                img_pred, raw_down_block_res_samples_unet, raw_mid_block_sample_unet, _ = self.unet(
                    latent_model_input_img,
                    t_img,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )
                
                label_pred = self.controldec(
                  sample = raw_mid_block_sample_ctlnet,
                  down_block_res_samples = raw_down_block_res_samples_ctlnet, 
                  timestep = t_label,
                  encoder_hidden_states=prompt_embeds,
                  down_block_additional_residuals=raw_down_block_res_samples_unet,
                  mid_block_additional_residual=raw_mid_block_sample_unet,
                  return_dict = False
                )

                
                # perform guidance
                mask_pred, normal_pred = label_pred[:, :4, :, :], label_pred[:, 4:, :, :]
                # perform guidance
                if self.do_classifier_free_guidance:
                    img_pred_uncond, img_pred_text = img_pred.chunk(2)
                    img_pred = img_pred_uncond + self.guidance_scale * (img_pred_text - img_pred_uncond)

                    mask_pred_uncond, mask_pred_text = mask_pred.chunk(2)
                    mask_pred = mask_pred_uncond + self.guidance_scale * (mask_pred_text - mask_pred_uncond)

                    normal_pred_uncond, normal_pred_text = normal_pred.chunk(2)
                    normal_pred = normal_pred_uncond + self.guidance_scale * (normal_pred_text - normal_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_img = self.scheduler_img.step(img_pred, t_img, latents_img, return_dict=False)[0]
                latents_mask = self.scheduler_mask.step(mask_pred, t_mask, latents_mask, return_dict=False)[0]
                latents_normal = self.scheduler_normal.step(normal_pred, t_normal, latents_normal, return_dict=False)[0]

                if i == len(timesteps_img) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler_img.order == 0):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            self.controldec.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept_img = self.run_safety_checker(image, device, prompt_embeds.dtype)

            first_label_latents = latents_mask # zhifei
            second_label_latents = latents_normal

            label_1 = self.vae.decode(first_label_latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            label_2 = self.vae.decode(second_label_latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            mask_1, has_nsfw_concept_mask_1 = self.run_safety_checker(label_1, device, prompt_embeds.dtype)
            mask_2, has_nsfw_concept_mask_2 = self.run_safety_checker(label_2, device, prompt_embeds.dtype)
        else:
            image = latents_img
            has_nsfw_concept_img = None

            mask_1 = latents_mask
            mask_2 = latents_normal #zhifei
            has_nsfw_concept_mask_1 = None
            has_nsfw_concept_mask_2 = None

        if has_nsfw_concept_img is None:
            do_denormalize_img = [True] * image.shape[0]
        else:
            do_denormalize_img = [not has_nsfw for has_nsfw in has_nsfw_concept_img] #zhifei 
        
        if has_nsfw_concept_mask_1 is None:
            do_denormalize_mask_1 = [True] * mask_1.shape[0]
        else:
            do_denormalize_mask_1 = [not has_nsfw for has_nsfw in has_nsfw_concept_mask_1]#zhifei

        if has_nsfw_concept_mask_2 is None:  #zhifei
            do_denormalize_mask_2 = [True] * mask_2.shape[0]
        else:
            do_denormalize_mask_2 = [not has_nsfw for has_nsfw in has_nsfw_concept_mask_2]


        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize_img)
        mask_1 = self.control_image_processor.postprocess(mask_1, output_type=output_type, do_denormalize=do_denormalize_mask_1)
        mask_2 = self.control_image_processor.postprocess(mask_2, output_type=output_type, do_denormalize=do_denormalize_mask_2)#zhifei


        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return (image, mask_1, mask_2)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
