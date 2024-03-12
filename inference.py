import os
import datetime
import argparse

from scripts.inference_controlnet import inference_controlnet
from scripts.inference_lora import inference_lora
from scripts.inference_ctrlnet_tile import inference_ctrlnet_tile


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference setting for X-Adapter.")

    parser.add_argument(
        "--plugin_type",
        type=str, help='lora or controlnet', default="controlnet"
    )
    parser.add_argument(
        "--controlnet_condition_scale_list",
        nargs='+', help='controlnet_scale', default=[1.0, 2.0]
    )
    parser.add_argument(
        "--adapter_guidance_start_list",
        nargs='+', help='start of 2nd stage', default=[0.6, 0.65, 0.7, 0.75, 0.8]
    )
    parser.add_argument(
        "--adapter_condition_scale_list",
        nargs='+', help='X-Adapter scale', default=[0.8, 1.0, 1.2]
    )
    parser.add_argument(
        "--base_path",
        type=str, help='path to base model', default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--sdxl_path",
        type=str, help='path to SDXL', default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--path_vae_sdxl",
        type=str, help='path to SDXL vae', default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument(
        "--adapter_checkpoint",
        type=str, help='path to X-Adapter', default="./checkpoint/X-Adapter/X_Adapter_v1.bin"
    )
    parser.add_argument(
        "--condition_type",
        type=str, help='condition type', default="canny"
    )
    parser.add_argument(
        "--controlnet_canny_path",
        type=str, help='path to canny controlnet', default="lllyasviel/sd-controlnet-canny"
    )
    parser.add_argument(
        "--controlnet_depth_path",
        type=str, help='path to depth controlnet', default="lllyasviel/sd-controlnet-depth"
    )
    parser.add_argument(
        "--controlnet_tile_path",
        type=str, help='path to controlnet tile', default="lllyasviel/control_v11f1e_sd15_tile"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str, help='path to lora', default="./checkpoint/lora/MoXinV1.safetensors"
    )
    parser.add_argument(
        "--prompt",
        type=str, help='SDXL prompt', default=None, required=True
    )
    parser.add_argument(
        "--prompt_sd1_5",
        type=str, help='SD1.5 prompt', default=None
    )
    parser.add_argument(
        "--negative_prompt",
        type=str, default=None
    )
    parser.add_argument(
        "--iter_num",
        type=int, default=1
    )
    parser.add_argument(
        "--input_image_path",
        type=str, default="./controlnet_test_image/CuteCat.jpeg"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int, default=50
    )
    parser.add_argument(
        "--guidance_scale",
        type=float, default=7.5
    )
    parser.add_argument(
        "--seed",
        type=int, default=1674753452
    )
    parser.add_argument(
        "--width",
        type=int, default=1024
    )
    parser.add_argument(
        "--height",
        type=int, default=1024
    )
    parser.add_argument(
        "--height_sd1_5",
        type=int, default=512
    )
    parser.add_argument(
        "--width_sd1_5",
        type=int, default=512
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def run_inference(args):
    current_datetime = datetime.datetime.now()
    current_datetime = str(current_datetime).replace(":", "_")
    save_path = f"./result/{current_datetime}_lora" if args.plugin_type == "lora" else f"./result/{current_datetime}_controlnet"
    os.makedirs(save_path)
    args.save_path = save_path

    if args.plugin_type == "controlnet":
        inference_controlnet(args)
    elif args.plugin_type == "controlnet_tile":
        inference_ctrlnet_tile(args)
    elif args.plugin_type == "lora":
        inference_lora(args)
    else:
        raise NotImplementedError("not implemented yet")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
