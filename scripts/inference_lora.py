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
from safetensors.torch import load
from PIL import Image
from packaging import version
from diffusers import StableDiffusionXLPipeline
import pprint
import inspect

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel, \
    T2IAdapter
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from model.unet_adapter import UNet2DConditionModel
from pipeline.pipeline_sd_xl_adapter import StableDiffusionXLAdapterPipeline
from model.adapter import Adapter_XL
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


def load_lora(pipeline, lora_model_path, alpha):
    state_dict = load_file(lora_model_path)

    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:

        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue

        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER + '_')[-1].split('_')
            curr_layer = pipeline.text_encoder_sd1_5
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET + '_')[-1].split('_')
            curr_layer = pipeline.unet_sd1_5

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_' + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)



        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        # update visited list
        for item in pair_keys:
            visited.append(item)


def inference_lora(args):
    device = 'cuda'
    weight_dtype = torch.float16

    adapter_guidance_start_list = str2float(args.adapter_guidance_start_list)
    adapter_condition_scale_list = str2float(args.adapter_condition_scale_list)

    path = args.base_path
    path_sdxl = args.sdxl_path
    path_vae_sdxl = args.path_vae_sdxl
    adapter_path = args.adapter_checkpoint
    lora_model_path = args.lora_model_path

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

    # load adapter
    adapter = Adapter_XL()
    ckpt = torch.load(adapter_path)
    adapter.load_state_dict(ckpt)
    print('successfully load adapter')
    # load SD1.5
    sdpipline = StableDiffusionPipeline.from_single_file(f"{path}")
    unet_sd1_5 = UNet2DConditionModel.from_config(sdpipline.unet.config)
    unet_sd1_5.load_state_dict(sdpipline.unet.state_dict().copy())
    "return_hidden_states" in inspect.signature(sdpipline.unet.forward).parameters.keys()  # False
    "return_hidden_states" in inspect.signature(unet_sd1_5.forward).parameters.keys()  # True


    vae_sd1_5 = sdpipline.vae
    # <All keys matched successfully>
    #DPMSolverMultistepScheduler.load_state_dict
    noise_scheduler_sd1_5 = DDPMScheduler.from_config(sdpipline.scheduler.config)
    tokenizer_sd1_5 = sdpipline.tokenizer
    text_encoder_sd1_5  = sdpipline.text_encoder

    print('successfully load SD1.5')
    # load SDXL
    sdpipline2 = StableDiffusionXLPipeline.from_single_file(f"{path_sdxl}")
    unetXL = UNet2DConditionModel.from_config(sdpipline2.unet.config)
    unetXL.load_state_dict(sdpipline2.unet.state_dict().copy())
    "return_hidden_states" in inspect.signature(sdpipline2.unet.forward).parameters.keys()  # False
    "return_hidden_states" in inspect.signature(unetXL.forward).parameters.keys()  # True
        #unet = unet_sd1_5)
    tokenizer_one  = sdpipline2.tokenizer
    tokenizer_two  = sdpipline2.tokenizer_2
    text_encoder_cls_one =  sdpipline2.text_encoder
    text_encoder_cls_two =  sdpipline2.text_encoder_2
    noise_scheduler = sdpipline2.scheduler
    text_encoder_one =  sdpipline2.text_encoder
    text_encoder_two =  sdpipline2.text_encoder_2
    vae  = sdpipline2.vae

    print('successfully load SDXL')

    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unetXL.enable_xformers_memory_efficient_attention()
        unet_sd1_5.enable_xformers_memory_efficient_attention()

    with torch.inference_mode():
        gen = Generator("cuda")
        gen.manual_seed(args.seed)
        pipe = StableDiffusionXLAdapterPipeline(

            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=unetXL,
            scheduler=noise_scheduler,
            vae_sd1_5=vae_sd1_5,
            text_encoder_sd1_5=text_encoder_sd1_5,
            tokenizer_sd1_5=tokenizer_sd1_5,
            unet_sd1_5=unet_sd1_5,
            scheduler_sd1_5=noise_scheduler_sd1_5,
            adapter=adapter,
        )

        # load lora #unet_sd1_5
        load_lora(pipe, lora_model_path, 1)
        print('successfully load lora')

        pipe.to('cuda', weight_dtype)
        pipe.enable_model_cpu_offload()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler_sd1_5 = DPMSolverMultistepScheduler.from_config(pipe.scheduler_sd1_5.config)
        pipe.scheduler_sd1_5.config.timestep_spacing = "leading"
        for i in range(args.iter_num):
            for adapter_guidance_start in adapter_guidance_start_list:
                for adapter_condition_scale in adapter_condition_scale_list:
                    img = \
                        pipe(prompt=prompt, prompt_sd1_5=prompt_sd1_5, negative_prompt=negative_prompt, width=1024,
                             height=1024, height_sd1_5=512, width_sd1_5=512,
                             num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                             num_images_per_prompt=1, generator=gen,
                             adapter_guidance_start=adapter_guidance_start,
                             adapter_condition_scale=adapter_condition_scale).images[0]
                    img.save(
                        f"{args.save_path}/{prompt[:10]}_{i}_ags_{adapter_guidance_start:.2f}_acs_{adapter_condition_scale:.2f}.png")
    print(f"results saved in {args.save_path}")
