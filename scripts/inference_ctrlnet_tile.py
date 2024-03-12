import torch
import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
from torch import Generator
from safetensors.torch import load_file
from PIL import Image
from packaging import version
from huggingface_hub import HfApi
from pathlib import Path

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel, T2IAdapter, StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from model.unet_adapter import UNet2DConditionModel as UNet2DConditionModel_v2
from model.adapter import Adapter_XL
from pipeline.pipeline_sd_xl_adapter_controlnet_img2img import StableDiffusionXLAdapterControlnetI2IPipeline
from scripts.utils import str2float

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def inference_ctrlnet_tile(args):
    device = 'cuda'
    weight_dtype = torch.float16

    controlnet_condition_scale_list = str2float(args.controlnet_condition_scale_list)
    adapter_guidance_start_list = str2float(args.adapter_guidance_start_list)
    adapter_condition_scale_list = str2float(args.adapter_condition_scale_list)

    path = args.base_path
    path_sdxl = args.sdxl_path
    path_vae_sdxl = args.path_vae_sdxl
    adapter_path = args.adapter_checkpoint
    controlnet_path = args.controlnet_tile_path

    prompt = args.prompt
    if args.prompt_sd1_5 is None:
        prompt_sd1_5 = prompt
    else:
        prompt_sd1_5 = args.prompt_sd1_5

    if args.negative_prompt is None:
        negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    else:
        negative_prompt = args.negative_prompt

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # load controlnet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=weight_dtype
    )

    source_image = Image.open(args.input_image_path)
    # control_image = resize_for_condition_image(source_image, 512)
    input_image = source_image.convert("RGB")
    control_image = input_image.resize((args.width_sd1_5, args.height_sd1_5), resample=Image.LANCZOS)

    print('successfully load controlnet')
    # load adapter
    adapter = Adapter_XL()
    ckpt = torch.load(adapter_path)
    adapter.load_state_dict(ckpt)
    adapter.to(weight_dtype)
    print('successfully load adapter')
    # load SD1.5
    noise_scheduler_sd1_5 = DDPMScheduler.from_pretrained(
        path, subfolder="scheduler"
    )
    tokenizer_sd1_5 = CLIPTokenizer.from_pretrained(
        path, subfolder="tokenizer", revision=None, torch_dtype=weight_dtype
    )
    text_encoder_sd1_5 = CLIPTextModel.from_pretrained(
        path, subfolder="text_encoder", revision=None, torch_dtype=weight_dtype
    )
    vae_sd1_5 = AutoencoderKL.from_pretrained(
        path, subfolder="vae", revision=None, torch_dtype=weight_dtype
    )
    unet_sd1_5 = UNet2DConditionModel_v2.from_pretrained(
        path, subfolder="unet", revision=None, torch_dtype=weight_dtype
    )
    print('successfully load SD1.5')
    # load SDXL
    tokenizer_one = AutoTokenizer.from_pretrained(
        path_sdxl, subfolder="tokenizer", revision=None, use_fast=False, torch_dtype=weight_dtype
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        path_sdxl, subfolder="tokenizer_2", revision=None, use_fast=False, torch_dtype=weight_dtype
    )
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        path_sdxl, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        path_sdxl, None, subfolder="text_encoder_2"
    )
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(path_sdxl, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        path_sdxl, subfolder="text_encoder", revision=None, torch_dtype=weight_dtype
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        path_sdxl, subfolder="text_encoder_2", revision=None, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        path_vae_sdxl, revision=None, torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel_v2.from_pretrained(
        path_sdxl, subfolder="unet", revision=None, torch_dtype=weight_dtype
    )
    print('successfully load SDXL')

    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        unet_sd1_5.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    with torch.inference_mode():
        gen = Generator(device)
        gen.manual_seed(args.seed)
        pipe = StableDiffusionXLAdapterControlnetI2IPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=unet,
            scheduler=noise_scheduler,
            vae_sd1_5=vae_sd1_5,
            text_encoder_sd1_5=text_encoder_sd1_5,
            tokenizer_sd1_5=tokenizer_sd1_5,
            unet_sd1_5=unet_sd1_5,
            scheduler_sd1_5=noise_scheduler_sd1_5,
            adapter=adapter,
            controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler_sd1_5 = DPMSolverMultistepScheduler.from_config(pipe.scheduler_sd1_5.config)
        pipe.scheduler_sd1_5.config.timestep_spacing = "leading"
        pipe.unet.to(device=device, dtype=weight_dtype, memory_format=torch.channels_last)


        for i in range(args.iter_num):
            for controlnet_condition_scale in controlnet_condition_scale_list:
                for adapter_guidance_start in adapter_guidance_start_list:
                    for adapter_condition_scale in adapter_condition_scale_list:
                        img = \
                            pipe(prompt=prompt, negative_prompt=negative_prompt, prompt_sd1_5=prompt_sd1_5,
                                 width=args.width, height=args.height, height_sd1_5=args.height_sd1_5,
                                 width_sd1_5=args.width_sd1_5, source_img=control_image, image=control_image,
                                 num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                                 num_images_per_prompt=1, generator=gen,
                                 controlnet_conditioning_scale=controlnet_condition_scale,
                                 adapter_condition_scale=adapter_condition_scale,
                                 adapter_guidance_start=adapter_guidance_start).images[0]
                        img.save(
                            f"{args.save_path}/{prompt[:10]}_{i}_ccs_{controlnet_condition_scale:.2f}_ags_{adapter_guidance_start:.2f}_acs_{adapter_condition_scale:.2f}.png")

    print(f"results saved in {args.save_path}")

