# X-Adapter

This repository is the official implementation of [X-Adapter](https://arxiv.org/abs/2312.02238).

**[X-Adapter: Adding Universal Compatibility of Plugins for Upgraded Diffusion Model](https://arxiv.org/abs/2312.02238)**
<br/>
[Lingmin Ran](),
[Xiaodong Cun](https://vinthony.github.io/academic/),
[Jia-Wei Liu](https://jia-wei-liu.github.io/), 
[Rui Zhao](https://ruizhaocv.github.io/), 
[Song Zijie](), 
[Xintao Wang](https://xinntao.github.io/),
[Jussi Keppo](https://www.jussikeppo.com/), 
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://showlab.github.io/X-Adapter/)
[![arXiv](https://img.shields.io/badge/arXiv-2312.02238-b31b1b.svg)](https://arxiv.org/abs/2312.02238)

![Overview_v7](https://github.com/showlab/X-Adapter/assets/152716091/eb41c508-826c-404f-8223-09765765823b)

<em> X-Adapter enables plugins pretrained on the old version (e.g. SD1.5) directly work with the upgraded Model (e.g., SDXL) without further retraining.</em>

[//]: # (<p align="center">)

[//]: # (<img src="https://tuneavideo.github.io/assets/teaser.gif" width="1080px"/>  )

[//]: # (<br>)

[//]: # (<em>Given a video-text pair as input, our method, Tune-A-Video, fine-tunes a pre-trained text-to-image diffusion model for text-to-video generation.</em>)

[//]: # (</p>)

Thank [kijai](https://github.com/kijai) for creating a CumfyUI Warpper Node [here](https://github.com/kijai/ComfyUI-Diffusers-X-Adapter)

## News

- [17/02/2024] Inference code released

## Setup

### Requirements

```shell
conda create -n xadapter python=3.10
conda activate xadapter

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for high efficiency and low GPU cost.

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)). You can also use fine-tuned Stable Diffusion models trained on different styles (e.g., [Anything V4.0](https://huggingface.co/andite/anything-v4.0), [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion), etc.).

**[ControlNet]** [Controlnet](https://github.com/lllyasviel/ControlNet) is a method to control diffusion models with spatial conditions. You can download the ControlNet family [here](https://huggingface.co/lllyasviel/ControlNet).

**[LoRA]** [LoRA](https://arxiv.org/abs/2106.09685) is a lightweight adapter to fine-tune large-scale pretrained model. It is widely used for style or identity customization in diffusion models. You can download LoRA from the diffusion community (e.g., [civitai](https://civitai.com/)).

### Checkpoint

Models can be downloaded from our [Hugging Face page](https://huggingface.co/Lingmin-Ran/X-Adapter). Put the checkpoint in folder `./checkpoint/X-Adapter`.

## Usage

After preparing all checkpoints, we can run inference code using different plugins. You can refer to this [tutorial](https://www.reddit.com/r/StableDiffusion/comments/1asuyiw/xadapter/) to quickly get started with X-Adapter. 

### Controlnet Inference

Set `--controlnet_canny_path` or `--controlnet_depth_path` to ControlNet's path in the bash script. The default value is its Hugging Face model card. 

    sh ./bash_scripts/canny_controlnet_inference.sh
    sh ./bash_scripts/depth_controlnet_inference.sh

### LoRA Inference

Set `--lora_model_path` to LoRA's checkpoint in the bash script. In this example we use [MoXin](https://civitai.com/models/12597/moxin), and we put it in folder `./checkpoint/lora`.

    sh ./bash_scripts/lora_inference.sh

### Controlnet-Tile Inference

Set `--controlnet_tile_path` to ControlNet-tile's path in the bash script. The default value is its Hugging Face model card. 

    sh ./bash_scripts/controlnet_tile_inference.sh

## Cite
If you find X-Adapter useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{ran2023xadapter,
  title={X-Adapter: Adding Universal Compatibility of Plugins for Upgraded Diffusion Model},
  author={Lingmin Ran and Xiaodong Cun and Jia-Wei Liu and Rui Zhao and Song Zijie and Xintao Wang and Jussi Keppo and Mike Zheng Shou},
  journal={arXiv preprint arXiv:2312.02238},
  year={2023}
}
```
