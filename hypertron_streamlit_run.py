# Hypertron v2 (modified by @softology to work on Visions of Chaos and further modified by @multimodalart to run on MindsEye)
# Original file is located at https://colab.research.google.com/drive/1N4UNSbtNMd31N_gAT9rAm8ZzPh62Y5ud

"""
More info on flavors [here](https://i.ibb.co/hCdm3W4/flavors.png).
More info on prompt experiments [here](https://i.ibb.co/0FF7vNn/prompt-experiments.png).
The styles of made-up, not real, artists can be found [here](https://docs.google.com/spreadsheets/d/1nMq-TjBj3t6us-npLRoLFq0VtgpVwdXCKTcQgnxKgTQ/edit?usp=sharing).
Keywords cheatsheet can be found [here](https://imgur.com/a/SnSIQRu) (made by kingdomakrillic).
A short guide to prompt engineering can be found [here](https://docs.google.com/document/d/1qy5fdeThu7pIikulQuWpmKvYBiv9wMshIHcrBr-VldA/edit?usp=sharing).
"""

"""
Main_Libraries = True #@param {type:"boolean"}
Import_Libraries = True
Download_Video = False #@param {type:"boolean"}
Download_Super_Res = False #@param {type:"boolean"}
Download_Super_Slomo = False #@param {type:"boolean"}

if Main_Libraries == True:
  print('GPU:')
  !nvidia-smi --query-gpu=name,memory.total --format=cs

  print("\nDownloading CLIP...")
  !git clone https://github.com/openai/CLIP                  &> /dev/null
  
  print("Installing AI Python libraries...")
  !git clone https://github.com/CompVis/taming-transformers  &> /dev/null
  !pip install ftfy regex tqdm omegaconf pytorch-lightning   &> /dev/null
  !pip install kornia                                        &> /dev/null
  !pip install einops                                        &> /dev/null
  !pip install transformers                                  &> /dev/null
  !pip install torch_optimizer                               &> /dev/null

  !pip install noise                                         &> /dev/null
  !pip install gputil                                        &> /dev/null
  !pip install taming-transformers                           &> /dev/null
  
  #!git clone https://github.com/lessw2020/Ranger21.git       &> /dev/null
  #!cd Ranger21                                               &> /dev/null
  #!pip install -e .                                          &> /dev/null
  #!cd ..                                                     &> /dev/null

  !mkdir steps
#   %mkdir Init_Img

  print("Installing libraries for handling metadata...")
  !pip install stegano                                       &> /dev/null
  !apt install exempi                                        &> /dev/null
  !pip install python-xmp-toolkit                            &> /dev/null
  !pip install imgtag                                        &> /dev/null

  if Download_Video:
    print("Installing Python libraries for video creation...")
    !pip install imageio-ffmpeg &> /dev/null
    !pip install timm           &> /dev/null

  if Download_Super_Res:
    print("Installing Python libraries for super resolution...")
#     %cd /content/
    !git clone https://github.com/sberbank-ai/Real-ESRGAN /content/RealESRGAN &> /dev/null
#     %cd RealESRGAN
    !pip install -r requirements.txt &> /dev/null
    # download model weights
    # x2 
    #!gdown https://drive.google.com/uc?id=1pG2S3sYvSaO0V0B8QPOl1RapPHpUGOaV -O weights/RealESRGAN_x2.pth
    # x4
    !gdown https://drive.google.com/uc?id=1SGHdZAln4en65_NQeQY9UjchtkEF9f5F -O weights/RealESRGAN_x4.pth &> /dev/null
    # x8
    #!gdown https://drive.google.com/uc?id=1mT9ewx86PSrc43b-ax47l1E2UzR7Ln4j -O weights/RealESRGAN_x8.pth
#     %cd /content/


  if Download_Super_Slomo:
    !git clone -q --depth 1 https://github.com/avinashpaliwal/Super-SloMo.git &> /dev/null
    from os.path import exists
    def download_from_google_drive(file_id, file_name):
      # download a file from the Google Drive link
      !rm -f ./cookie
      !curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null
      confirm_text = !awk '/download/ {print $NF}' ./cookie
      confirm_text = confirm_text[0]
      !curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm={confirm_text}&id={file_id}" -o {file_name}  &> /dev/null
      
    pretrained_model = 'SuperSloMo.ckpt'
    if not exists(pretrained_model):
      download_from_google_drive('1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF', pretrained_model)

#   %mkdir png_processing

#   %mkdir templates
  !curl https://i.ibb.co/3kn9Qrv/flag.png -o templates/flag.png                       &> /dev/null
  !curl https://i.ibb.co/0BHqVyg/14135136623-3973d3f03c-z.jpg -o templates/planet.png &> /dev/null
  !curl https://i.ibb.co/52WMK2M/j7oocvu80qe11.png -o templates/map.png               &> /dev/null
  !curl https://i.ibb.co/3fg9Zkx/creature.png -o templates/creature.png               &> /dev/null
  !curl https://i.ibb.co/X3Mh2pP/human.jpg -o templates/human.png                     &> /dev/null
"""

import sys
import streamlit as st
import argparse
import math
from pathlib import Path
import sys
import pandas as pd
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from os.path import exists as path_exists

torch.cuda.empty_cache()
from torch import nn
import torch.optim as optim
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision.transforms as T
#from stqdm import stqdm

# from tqdm.notebook import tqdm
from CLIP import clip
import kornia.augmentation as K
import numpy as np
import subprocess
import imageio
from PIL import ImageFile, Image
import time

# ImageFile.LOAD_TRUNCATED_IMAGES = True
import hashlib
from PIL.PngImagePlugin import PngImageFile, PngInfo
import json
import IPython
from IPython.display import Markdown, display, Image, clear_output
import urllib.request
import random
from random import randint
from pathvalidate import sanitize_filename

sys.stdout.write("Imports ...\n")
sys.stdout.flush()

sys.path.append("./CLIP")
sys.path.append("./taming-transformers")


sys.stdout.write("Parsing arguments ...\n")
sys.stdout.flush()


def run_model(args2, status, stoutput, DefaultPaths):
    if args2.seed is not None:
        import torch

        sys.stdout.write(f"Setting seed to {args2.seed} ...\n")
        sys.stdout.flush()
        status.write(f"Setting seed to {args2.seed} ...\n")
        import numpy as np

        np.random.seed(args2.seed)
        import random

        random.seed(args2.seed)
        # next line forces deterministic random values, but causes other issues with resampling (uncomment to see)
        torch.manual_seed(args2.seed)
        torch.cuda.manual_seed(args2.seed)
        torch.cuda.manual_seed_all(args2.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    """
  from imgtag import ImgTag    # metadata 
  from libxmp import *         # metadata
  import libxmp                # metadata
  from stegano import lsb
  import gc
  import GPUtil as GPU
  """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    def noise_gen(shape, octaves=5):
        n, c, h, w = shape
        noise = torch.zeros([n, c, 1, 1])
        max_octaves = min(octaves, math.log(h) / math.log(2), math.log(w) / math.log(2))
        for i in reversed(range(max_octaves)):
            h_cur, w_cur = h // 2**i, w // 2**i
            noise = F.interpolate(
                noise, (h_cur, w_cur), mode="bicubic", align_corners=False
            )
            noise += torch.randn([n, c, h_cur, w_cur]) / 5
        return noise

    def sinc(x):
        return torch.where(
            x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([])
        )

    def lanczos(x, a):
        cond = torch.logical_and(-a < x, x < a)
        out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
        return out / out.sum()

    def ramp(ratio, width):
        n = math.ceil(width / ratio + 1)
        out = torch.empty([n])
        cur = 0
        for i in range(out.shape[0]):
            out[i] = cur
            cur += ratio
        return torch.cat([-out[1:].flip([0]), out])[1:-1]

    def resample(input, size, align_corners=True):
        n, c, h, w = input.shape
        dh, dw = size

        input = input.view([n * c, 1, h, w])

        if dh < h:
            kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
            pad_h = (kernel_h.shape[0] - 1) // 2
            input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
            input = F.conv2d(input, kernel_h[None, None, :, None])

        if dw < w:
            kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
            pad_w = (kernel_w.shape[0] - 1) // 2
            input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
            input = F.conv2d(input, kernel_w[None, None, None, :])

        input = input.view([n, c, h, w])
        return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)

    def lerp(a, b, f):
        return (a * (1.0 - f)) + (b * f)

    class ReplaceGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_forward, x_backward):
            ctx.shape = x_backward.shape
            return x_forward

        @staticmethod
        def backward(ctx, grad_in):
            return None, grad_in.sum_to_size(ctx.shape)

    replace_grad = ReplaceGrad.apply

    class ClampWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, min, max):
            ctx.min = min
            ctx.max = max
            ctx.save_for_backward(input)
            return input.clamp(min, max)

        @staticmethod
        def backward(ctx, grad_in):
            (input,) = ctx.saved_tensors
            return (
                grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
                None,
                None,
            )

    clamp_with_grad = ClampWithGrad.apply

    def vector_quantize(x, codebook):
        d = (
            x.pow(2).sum(dim=-1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * x @ codebook.T
        )
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return replace_grad(x_q, x)

    class Prompt(nn.Module):
        def __init__(self, embed, weight=1.0, stop=float("-inf")):
            super().__init__()
            self.register_buffer("embed", embed)
            self.register_buffer("weight", torch.as_tensor(weight))
            self.register_buffer("stop", torch.as_tensor(stop))

        def forward(self, input):
            input_normed = F.normalize(input.unsqueeze(1), dim=2)
            embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
            dists = (
                input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            )
            dists = dists * self.weight.sign()
            return (
                self.weight.abs()
                * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
            )

    # def parse_prompt(prompt):
    #    vals = prompt.rsplit(':', 2)
    #    vals = vals + ['', '1', '-inf'][len(vals):]
    #    return vals[0], float(vals[1]), float(vals[2])

    def parse_prompt(prompt):
        if prompt.startswith("http://") or prompt.startswith("https://"):
            vals = prompt.rsplit(":", 1)
            vals = [vals[0] + ":" + vals[1], *vals[2:]]
        else:
            vals = prompt.rsplit(":", 1)
        vals = vals + ["", "1", "-inf"][len(vals) :]
        return vals[0], float(vals[1]), float(vals[2])

    def one_sided_clip_loss(input, target, labels=None, logit_scale=100):
        input_normed = F.normalize(input, dim=-1)
        target_normed = F.normalize(target, dim=-1)
        logits = input_normed @ target_normed.T * logit_scale
        if labels is None:
            labels = torch.arange(len(input), device=logits.device)
        return F.cross_entropy(logits, labels)

    class EMATensor(nn.Module):
        """implmeneted by Katherine Crowson"""

        def __init__(self, tensor, decay):
            super().__init__()
            self.tensor = nn.Parameter(tensor)
            self.register_buffer("biased", torch.zeros_like(tensor))
            self.register_buffer("average", torch.zeros_like(tensor))
            self.decay = decay
            self.register_buffer("accum", torch.tensor(1.0))
            self.update()

        @torch.no_grad()
        def update(self):
            if not self.training:
                raise RuntimeError("update() should only be called during training")

            self.accum *= self.decay
            self.biased.mul_(self.decay)
            self.biased.add_((1 - self.decay) * self.tensor)
            self.average.copy_(self.biased)
            self.average.div_(1 - self.accum)

        def forward(self):
            if self.training:
                return self.tensor
            return self.average

    ############################################################################################
    ############################################################################################

    class MakeCutoutsCustom(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = nn.Sequential(
                K.RandomHorizontalFlip(p=Random_Horizontal_Flip),
                K.RandomSharpness(Random_Sharpness, p=Random_Sharpness_P),
                K.RandomGaussianBlur(
                    (Random_Gaussian_Blur),
                    (Random_Gaussian_Blur_W, Random_Gaussian_Blur_W),
                    p=Random_Gaussian_Blur_P,
                ),
                K.RandomGaussianNoise(p=Random_Gaussian_Noise_P),
                K.RandomElasticTransform(
                    kernel_size=(
                        Random_Elastic_Transform_Kernel_Size_W,
                        Random_Elastic_Transform_Kernel_Size_H,
                    ),
                    sigma=(Random_Elastic_Transform_Sigma),
                    p=Random_Elastic_Transform_P,
                ),
                K.RandomAffine(
                    degrees=Random_Affine_Degrees,
                    translate=Random_Affine_Translate,
                    p=Random_Affine_P,
                    padding_mode="border",
                ),
                K.RandomPerspective(Random_Perspective, p=Random_Perspective_P),
                K.ColorJitter(
                    hue=Color_Jitter_Hue,
                    saturation=Color_Jitter_Saturation,
                    p=Color_Jitter_P,
                ),
            )
            # K.RandomErasing((0.1, 0.7), (0.3, 1/0.4), same_on_batch=True, p=0.2),)

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []
            noise_fac = 0.1

            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):

                # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                randsize = (
                    torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )
                size_mult = randsize**self.cut_pow
                size = int(
                    min_size_width * (size_mult.clip(lower_bound, 1.0))
                )  # replace .5 with a result for 224 the default large size is .95
                # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)
            cutouts = clamp_with_grad(cutouts, 0, 1)

            # if args.use_augs:
            cutouts = self.augs(cutouts)
            if self.noise_fac:
                facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                    0, self.noise_fac
                )
                cutouts = cutouts + facs * torch.randn_like(cutouts)
            return cutouts

    class MakeCutoutsJuu(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.augs = nn.Sequential(
                # K.RandomGaussianNoise(mean=0.0, std=0.5, p=0.1),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomSharpness(0.3, p=0.4),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
                K.RandomPerspective(0.2, p=0.4),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
                K.RandomGrayscale(p=0.1),
            )
            self.noise_fac = 0.1

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(
                    torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            batch = self.augs(torch.cat(cutouts, dim=0))
            if self.noise_fac:
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
                batch = batch + facs * torch.randn_like(batch)
            return batch

    class MakeCutoutsMoth(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs, skip_augs=False):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.skip_augs = skip_augs
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomPerspective(distortion_scale=0.4, p=0.7),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.15),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )

        def forward(self, input):
            input = T.Pad(input.shape[2] // 4, fill=0)(input)
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)

            cutouts = []
            for ch in range(cutn):
                if ch > cutn - cutn // 4:
                    cutout = input.clone()
                else:
                    size = int(
                        max_size
                        * torch.zeros(
                            1,
                        )
                        .normal_(mean=0.8, std=0.3)
                        .clip(float(self.cut_size / max_size), 1.0)
                    )
                    offsetx = torch.randint(0, abs(sideX - size + 1), ())
                    offsety = torch.randint(0, abs(sideY - size + 1), ())
                    cutout = input[
                        :, :, offsety : offsety + size, offsetx : offsetx + size
                    ]

                if not self.skip_augs:
                    cutout = self.augs(cutout)
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
                del cutout

            cutouts = torch.cat(cutouts, dim=0)
            return cutouts

    class MakeCutoutsAaron(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.augs = augs
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []

            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):
                size = int(
                    min_size_width
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )  # replace .5 with a result for 224 the default large size is .95

                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)

            return clamp_with_grad(cutouts, 0, 1)

    class MakeCutoutsCumin(nn.Module):
        # from https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = nn.Sequential(
                # K.RandomHorizontalFlip(p=0.5),
                # K.RandomSharpness(0.3,p=0.4),
                # K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
                # K.RandomGaussianNoise(p=0.5),
                # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
                K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode="border"),
                K.RandomPerspective(0.7, p=0.7),
                K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
                K.RandomErasing((0.1, 0.4), (0.3, 1 / 0.3), same_on_batch=True, p=0.7),
            )

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []
            noise_fac = 0.1

            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):

                # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                randsize = (
                    torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )
                size_mult = randsize**self.cut_pow
                size = int(
                    min_size_width * (size_mult.clip(lower_bound, 1.0))
                )  # replace .5 with a result for 224 the default large size is .95
                # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)
            cutouts = clamp_with_grad(cutouts, 0, 1)

            # if args.use_augs:
            cutouts = self.augs(cutouts)
            if self.noise_fac:
                facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                    0, self.noise_fac
                )
                cutouts = cutouts + facs * torch.randn_like(cutouts)
            return cutouts

    class MakeCutoutsHolywater(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = nn.Sequential(
                # K.RandomGaussianNoise(mean=0.0, std=0.5, p=0.1),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomSharpness(0.3, p=0.4),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
                K.RandomPerspective(0.2, p=0.4),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
                K.RandomGrayscale(p=0.1),
            )

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []
            noise_fac = 0.1
            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):
                size = int(
                    torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
                )
                randsize = (
                    torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )
                size_mult = randsize**self.cut_pow * ii + size
                size1 = int(
                    (min_size_width) * (size_mult.clip(lower_bound, 1.0))
                )  # replace .5 with a result for 224 the default large size is .95
                size2 = int(
                    (min_size_width)
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.9, std=0.3)
                    .clip(lower_bound, 0.95)
                )  # replace .5 with a result for 224 the default large size is .95
                offsetx = torch.randint(0, sideX - size1 + 1, ())
                offsety = torch.randint(0, sideY - size2 + 1, ())
                cutout = input[
                    :, :, offsety : offsety + size2 + ii, offsetx : offsetx + size1 + ii
                ]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)
            cutouts = clamp_with_grad(cutouts, 0, 1)
            cutouts = self.augs(cutouts)
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                0, self.noise_fac
            )
            cutouts = cutouts + facs * torch.randn_like(cutouts)
            return cutouts

    class MakeCutoutsOldHolywater(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = nn.Sequential(
                # K.RandomHorizontalFlip(p=0.5),
                # K.RandomSharpness(0.3,p=0.4),
                # K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
                # K.RandomGaussianNoise(p=0.5),
                # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
                K.RandomAffine(
                    degrees=180, translate=0.5, p=0.2, padding_mode="border"
                ),
                K.RandomPerspective(0.6, p=0.9),
                K.ColorJitter(hue=0.03, saturation=0.01, p=0.1),
                K.RandomErasing((0.1, 0.7), (0.3, 1 / 0.4), same_on_batch=True, p=0.2),
            )

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []
            noise_fac = 0.1

            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):

                # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                randsize = (
                    torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )
                size_mult = randsize**self.cut_pow
                size = int(
                    min_size_width * (size_mult.clip(lower_bound, 1.0))
                )  # replace .5 with a result for 224 the default large size is .95
                # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)
            cutouts = clamp_with_grad(cutouts, 0, 1)

            # if args.use_augs:
            cutouts = self.augs(cutouts)
            if self.noise_fac:
                facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                    0, self.noise_fac
                )
                cutouts = cutouts + facs * torch.randn_like(cutouts)
            return cutouts

    class MakeCutoutsGinger(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = augs
            """
          nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
            K.RandomGaussianNoise(p=0.5),
            K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
            K.RandomPerspective(0.2,p=0.4, ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),)
  """

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []
            noise_fac = 0.1

            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):

                # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                randsize = (
                    torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )
                size_mult = randsize**self.cut_pow
                size = int(
                    min_size_width * (size_mult.clip(lower_bound, 1.0))
                )  # replace .5 with a result for 224 the default large size is .95
                # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)
            cutouts = clamp_with_grad(cutouts, 0, 1)

            # if args.use_augs:
            cutouts = self.augs(cutouts)
            if self.noise_fac:
                facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                    0, self.noise_fac
                )
                cutouts = cutouts + facs * torch.randn_like(cutouts)
            return cutouts

    class MakeCutoutsZynth(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                # K.RandomSolarize(0.01, 0.01, p=0.7),
                K.RandomSharpness(0.3, p=0.4),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
                K.RandomPerspective(0.2, p=0.4),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            )

        def set_cut_pow(self, cut_pow):
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            cutouts_full = []
            noise_fac = 0.1

            min_size_width = min(sideX, sideY)
            lower_bound = float(self.cut_size / min_size_width)

            for ii in range(self.cutn):

                # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                randsize = (
                    torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(lower_bound, 1.0)
                )
                size_mult = randsize**self.cut_pow
                size = int(
                    min_size_width * (size_mult.clip(lower_bound, 1.0))
                )  # replace .5 with a result for 224 the default large size is .95
                # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            cutouts = torch.cat(cutouts, dim=0)
            cutouts = clamp_with_grad(cutouts, 0, 1)

            # if args.use_augs:
            cutouts = self.augs(cutouts)
            if self.noise_fac:
                facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                    0, self.noise_fac
                )
                cutouts = cutouts + facs * torch.randn_like(cutouts)
            return cutouts

    class MakeCutoutsWyvern(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow, augs):
            super().__init__()
            self.cut_size = cut_size
            # tqdm.write(f"cut size: {self.cut_size}")
            self.cutn = cutn
            self.cut_pow = cut_pow
            self.noise_fac = 0.1
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
            self.augs = augs

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(
                    torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

    def load_vqgan_model(config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == "taming.models.vqgan.VQModel":
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        elif config.model.target == "taming.models.vqgan.GumbelVQ":
            model = vqgan.GumbelVQ(**config.model.params)
            # print(config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        else:
            raise ValueError(f"unknown model type: {config.model.target}")
        del model.loss
        return model

    import PIL

    def resize_image(image, out_size):
        ratio = image.size[0] / image.size[1]
        area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
        size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
        return image.resize(size, PIL.Image.LANCZOS)

    class GaussianBlur2d(nn.Module):
        def __init__(self, sigma, window=0, mode="reflect", value=0):
            super().__init__()
            self.mode = mode
            self.value = value
            if not window:
                window = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
            if sigma:
                kernel = torch.exp(
                    -((torch.arange(window) - window // 2) ** 2) / 2 / sigma**2
                )
                kernel /= kernel.sum()
            else:
                kernel = torch.ones([1])
            self.register_buffer("kernel", kernel)

        def forward(self, input):
            n, c, h, w = input.shape
            input = input.view([n * c, 1, h, w])
            start_pad = (self.kernel.shape[0] - 1) // 2
            end_pad = self.kernel.shape[0] // 2
            input = F.pad(
                input, (start_pad, end_pad, start_pad, end_pad), self.mode, self.value
            )
            input = F.conv2d(input, self.kernel[None, None, None, :])
            input = F.conv2d(input, self.kernel[None, None, :, None])
            return input.view([n, c, h, w])

    BUF_SIZE = 65536

    def get_digest(path, alg=hashlib.sha256):
        hash = alg()
        # print(path)
        with open(path, "rb") as fp:
            while True:
                data = fp.read(BUF_SIZE)
                if not data:
                    break
                hash.update(data)
        return b64encode(hash.digest()).decode("utf-8")

    flavordict = {
        "cumin": MakeCutoutsCumin,
        "holywater": MakeCutoutsHolywater,
        "old_holywater": MakeCutoutsOldHolywater,
        "ginger": MakeCutoutsGinger,
        "zynth": MakeCutoutsZynth,
        "wyvern": MakeCutoutsWyvern,
        "aaron": MakeCutoutsAaron,
        "moth": MakeCutoutsMoth,
        "juu": MakeCutoutsJuu,
        "custom": MakeCutoutsCustom,
    }

    @torch.jit.script
    def gelu_impl(x):
        """OpenAI's gelu implementation."""
        return (
            0.5
            * x
            * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
        )

    def gelu(x):
        return gelu_impl(x)

    class MSEDecayLoss(nn.Module):
        def __init__(self, init_weight, mse_decay_rate, mse_epoches, mse_quantize):
            super().__init__()

            self.init_weight = init_weight
            self.has_init_image = False
            self.mse_decay = init_weight / mse_epoches if init_weight else 0
            self.mse_decay_rate = mse_decay_rate
            self.mse_weight = init_weight
            self.mse_epoches = mse_epoches
            self.mse_quantize = mse_quantize

        @torch.no_grad()
        def set_target(self, z_tensor, model):
            z_tensor = z_tensor.detach().clone()
            if self.mse_quantize:
                z_tensor = vector_quantize(
                    z_tensor.movedim(1, 3), model.quantize.embedding.weight
                ).movedim(
                    3, 1
                )  # z.average
            self.z_orig = z_tensor

        def forward(self, i, z):
            if self.is_active(i):
                return F.mse_loss(z, self.z_orig) * self.mse_weight / 2
            return 0

        def is_active(self, i):
            if not self.init_weight:
                return False
            if i <= self.mse_decay_rate and not self.has_init_image:
                return False
            return True

        @torch.no_grad()
        def step(self, i):

            if (
                i % self.mse_decay_rate == 0
                and i != 0
                and i < self.mse_decay_rate * self.mse_epoches
            ):

                if (
                    self.mse_weight - self.mse_decay > 0
                    and self.mse_weight - self.mse_decay >= self.mse_decay
                ):
                    self.mse_weight -= self.mse_decay
                else:
                    self.mse_weight = 0
                # print(f"updated mse weight: {self.mse_weight}")

                return True

            return False

    class TVLoss(nn.Module):
        def forward(self, input):
            input = F.pad(input, (0, 1, 0, 1), "replicate")
            x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
            y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
            diff = x_diff**2 + y_diff**2 + 1e-8
            return diff.mean(dim=1).sqrt().mean()

    class MultiClipLoss(nn.Module):
        def __init__(
            self, clip_models, text_prompt, cutn, cut_pow=1.0, clip_weight=1.0
        ):
            super().__init__()

            # Load Clip
            self.perceptors = []
            for cm in clip_models:
                sys.stdout.write(f"Loading {cm[0]} ...\n")
                sys.stdout.flush()
                c = (
                    clip.load(cm[0], jit=False)[0]
                    .eval()
                    .requires_grad_(False)
                    .to(device)
                )
                self.perceptors.append(
                    {
                        "res": c.visual.input_resolution,
                        "perceptor": c,
                        "weight": cm[1],
                        "prompts": [],
                    }
                )
            self.perceptors.sort(key=lambda e: e["res"], reverse=True)

            # Make Cutouts
            self.max_cut_size = self.perceptors[0]["res"]
            # self.make_cuts = flavordict[flavor](self.max_cut_size, cutn, cut_pow)
            # cutouts = flavordict[flavor](self.max_cut_size, cutn, cut_pow=cut_pow, augs=args.augs)

            # Get Prompt Embedings
            # texts = [phrase.strip() for phrase in text_prompt.split("|")]
            # if text_prompt == ['']:
            #  texts = []
            texts = text_prompt
            self.pMs = []
            for prompt in texts:
                txt, weight, stop = parse_prompt(prompt)
                clip_token = clip.tokenize(txt).to(device)
                for p in self.perceptors:
                    embed = p["perceptor"].encode_text(clip_token).float()
                    embed_normed = F.normalize(embed.unsqueeze(0), dim=2)
                    p["prompts"].append(
                        {
                            "embed_normed": embed_normed,
                            "weight": torch.as_tensor(weight, device=device),
                            "stop": torch.as_tensor(stop, device=device),
                        }
                    )

            # Prep Augments
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            self.augs = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomSharpness(0.3, p=0.1),
                K.RandomAffine(
                    degrees=30, translate=0.1, p=0.8, padding_mode="border"
                ),  # padding_mode=2
                K.RandomPerspective(
                    0.2,
                    p=0.4,
                ),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
                K.RandomGrayscale(p=0.15),
            )
            self.noise_fac = 0.1

            self.clip_weight = clip_weight

        def prepare_cuts(self, img):
            cutouts = self.make_cuts(img)
            cutouts = self.augs(cutouts)
            if self.noise_fac:
                facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                    0, self.noise_fac
                )
                cutouts = cutouts + facs * torch.randn_like(cutouts)
            cutouts = self.normalize(cutouts)
            return cutouts

        def forward(self, i, img):
            cutouts = checkpoint(self.prepare_cuts, img)
            loss = []

            current_cuts = cutouts
            currentres = self.max_cut_size
            for p in self.perceptors:
                if currentres != p["res"]:
                    current_cuts = resample(cutouts, (p["res"], p["res"]))
                    currentres = p["res"]

                iii = p["perceptor"].encode_image(current_cuts).float()
                input_normed = F.normalize(iii.unsqueeze(1), dim=2)
                for prompt in p["prompts"]:
                    dists = (
                        input_normed.sub(prompt["embed_normed"])
                        .norm(dim=2)
                        .div(2)
                        .arcsin()
                        .pow(2)
                        .mul(2)
                    )
                    dists = dists * prompt["weight"].sign()
                    l = (
                        prompt["weight"].abs()
                        * replace_grad(
                            dists, torch.maximum(dists, prompt["stop"])
                        ).mean()
                    )
                    loss.append(l * p["weight"])

            return loss

    class ModelHost:
        def __init__(self, args):
            self.args = args
            self.model, self.perceptor = None, None
            self.make_cutouts = None
            self.alt_make_cutouts = None
            self.imageSize = None
            self.prompts = None
            self.opt = None
            self.normalize = None
            self.z, self.z_orig, self.z_min, self.z_max = None, None, None, None
            self.metadata = None
            self.mse_weight = 0
            self.normal_flip_optim = None
            self.usealtprompts = False

        def setup_metadata(self, seed):
            metadata = {k: v for k, v in vars(self.args).items()}
            del metadata["max_iterations"]
            del metadata["display_freq"]
            metadata["seed"] = seed
            if metadata["init_image"]:
                path = metadata["init_image"]
                digest = get_digest(path)
                metadata["init_image"] = (path, digest)
            if metadata["image_prompts"]:
                prompts = []
                for prompt in metadata["image_prompts"]:
                    path = prompt
                    digest = get_digest(path)
                    prompts.append((path, digest))
                metadata["image_prompts"] = prompts
            self.metadata = metadata

        def setup_model(self, x):
            i = x
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            """
      print('Using device:', device)
      if self.args.prompts:
          print('Using prompts:', self.args.prompts)
      if self.args.altprompts:
          print('Using alternate augment set prompts:', self.args.altprompts)
      if self.args.image_prompts:
          print('Using image prompts:', self.args.image_prompts)
      if args.seed is None:
          seed = torch.seed()
      else:
          seed = args.seed
      torch.manual_seed(seed)
      print('Using seed:', seed)
      """
            model = load_vqgan_model(
                f"{DefaultPaths.model_path}/{args.vqgan_model}.yaml",
                f"{DefaultPaths.model_path}/{args.vqgan_model}.ckpt",
            ).to(device)

            active_clips = (
                bool(self.args.clip_model2)
                + bool(self.args.clip_model3)
                + bool(self.args.clip_model4)
                + bool(self.args.clip_model5)
                + bool(self.args.clip_model6)
                + bool(self.args.clip_model7)
                + bool(self.args.clip_model8)
            )
            if active_clips != 0:
                clip_weight = round(1 / (active_clips + 1), 2)
                clip_models = []
                clip_models.append([self.args.clip_model, clip_weight])
                print(clip_models)
            else:
                clip_models = [[clip_model, 1.0]]

            if self.args.clip_model2:
                clip_models.append([self.args.clip_model2, clip_weight])
            if self.args.clip_model3:
                clip_models.append([self.args.clip_model3, clip_weight])
            if self.args.clip_model4:
                clip_models.append([self.args.clip_model4, clip_weight])
            if self.args.clip_model5:
                clip_models.append([self.args.clip_model5, clip_weight])
            if self.args.clip_model6:
                clip_models.append([self.args.clip_model6, clip_weight])
            if self.args.clip_model7:
                clip_models.append([self.args.clip_model7, clip_weight])
            if self.args.clip_model8:
                clip_models.append([self.args.clip_model8, clip_weight])

            clip_loss = MultiClipLoss(
                clip_models, self.args.prompts, cutn=self.args.cutn
            )

            # update_random(self.args.gen_seed, 'image generation')

            # [0].eval().requires_grad_(False)
            perceptor = (
                clip.load(args.clip_model, jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(device)
            )
            # [0].eval().requires_grad_(True)

            cut_size = perceptor.visual.input_resolution

            if self.args.is_gumbel:
                e_dim = model.quantize.embedding_dim
            else:
                e_dim = model.quantize.e_dim

            f = 2 ** (model.decoder.num_resolutions - 1)

            make_cutouts = flavordict[flavor](
                cut_size, args.mse_cutn, cut_pow=args.mse_cut_pow, augs=args.augs
            )

            # make_cutouts = MakeCutouts(cut_size, args.mse_cutn, cut_pow=args.mse_cut_pow,augs=args.augs)
            if args.altprompts:
                self.usealtprompts = True
                self.alt_make_cutouts = flavordict[flavor](
                    cut_size,
                    args.mse_cutn,
                    cut_pow=args.alt_mse_cut_pow,
                    augs=args.altaugs,
                )
                # self.alt_make_cutouts = MakeCutouts(cut_size, args.mse_cutn, cut_pow=args.alt_mse_cut_pow,augs=args.altaugs)

            if self.args.is_gumbel:
                n_toks = model.quantize.n_embed
            else:
                n_toks = model.quantize.n_e

            toksX, toksY = args.size[0] // f, args.size[1] // f
            sideX, sideY = toksX * f, toksY * f

            if self.args.is_gumbel:
                z_min = model.quantize.embed.weight.min(dim=0).values[
                    None, :, None, None
                ]
                z_max = model.quantize.embed.weight.max(dim=0).values[
                    None, :, None, None
                ]
            else:
                z_min = model.quantize.embedding.weight.min(dim=0).values[
                    None, :, None, None
                ]
                z_max = model.quantize.embedding.weight.max(dim=0).values[
                    None, :, None, None
                ]

            from PIL import Image
            import cv2

            # -------
            working_dir = self.args.folder_name

            if self.args.init_image != "":
                img_0 = cv2.imread(init_image)
                z, *_ = model.encode(
                    TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1
                )
            elif not os.path.isfile(f"{working_dir}/steps/{i:04d}.png"):
                one_hot = F.one_hot(
                    torch.randint(n_toks, [toksY * toksX], device=device), n_toks
                ).float()
                if self.args.is_gumbel:
                    z = one_hot @ model.quantize.embed.weight
                else:
                    z = one_hot @ model.quantize.embedding.weight
                z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
            else:
                if save_all_iterations:
                    img_0 = cv2.imread(
                        f"{working_dir}/steps/{i:04d}_{iterations_per_frame}.png"
                    )
                else:
                    # Hack to prevent colour inversion on every frame
                    img_temp = cv2.imread(f"{working_dir}/steps/{i}.png")
                    imageio.imwrite("inverted_temp.png", img_temp)
                    img_0 = cv2.imread("inverted_temp.png")
                center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
                trans_mat = np.float32([[1, 0, 10], [0, 1, 10]])
                rot_mat = cv2.getRotationMatrix2D(center, 10, 20)

                trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                transformation_matrix = np.matmul(rot_mat, trans_mat)

                img_0 = cv2.warpPerspective(
                    img_0,
                    transformation_matrix,
                    (img_0.shape[1], img_0.shape[0]),
                    borderMode=cv2.BORDER_WRAP,
                )
                z, *_ = model.encode(
                    TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1
                )

                def save_output(i, img, suffix="zoomed"):
                    filename = f"{working_dir}/steps/{i:04}{'_' + suffix if suffix else ''}.png"
                    imageio.imwrite(filename, np.array(img))

                save_output(i, img_0)
            # -------
            if args.init_image:
                pil_image = Image.open(args.init_image).convert("RGB")
                pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                z, *_ = model.encode(
                    TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
                )
            else:
                one_hot = F.one_hot(
                    torch.randint(n_toks, [toksY * toksX], device=device), n_toks
                ).float()
                if self.args.is_gumbel:
                    z = one_hot @ model.quantize.embed.weight
                else:
                    z = one_hot @ model.quantize.embedding.weight
                z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
            z = EMATensor(z, args.ema_val)

            if args.mse_with_zeros and not args.init_image:
                z_orig = torch.zeros_like(z.tensor)
            else:
                z_orig = z.tensor.clone()
            z.requires_grad_(True)
            # opt = optim.AdamW(z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
            if self.normal_flip_optim == True:
                if randint(1, 2) == 1:
                    opt = torch.optim.AdamW(
                        z.parameters(), lr=args.step_size, weight_decay=0.00000000
                    )
                    # opt = Ranger21(z.parameters(), lr=args.step_size, weight_decay=0.00000000)
                else:
                    opt = optim.DiffGrad(
                        z.parameters(), lr=args.step_size, weight_decay=0.00000000
                    )
            else:
                opt = torch.optim.AdamW(
                    z.parameters(), lr=args.step_size, weight_decay=0.00000000
                )

            self.cur_step_size = args.mse_step_size

            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            pMs = []
            altpMs = []

            for prompt in args.prompts:
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

            for prompt in args.altprompts:
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                altpMs.append(Prompt(embed, weight, stop).to(device))

            from PIL import Image

            for prompt in args.image_prompts:
                path, weight, stop = parse_prompt(prompt)
                img = resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
                batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
                embed = perceptor.encode_image(normalize(batch)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

            for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
                gen = torch.Generator().manual_seed(seed)
                embed = torch.empty([1, perceptor.visual.output_dim]).normal_(
                    generator=gen
                )
                pMs.append(Prompt(embed, weight).to(device))
                if self.usealtprompts:
                    altpMs.append(Prompt(embed, weight).to(device))

            self.model, self.perceptor = model, perceptor
            self.make_cutouts = make_cutouts
            self.imageSize = (sideX, sideY)
            self.prompts = pMs
            self.altprompts = altpMs
            self.opt = opt
            self.normalize = normalize
            self.z, self.z_orig, self.z_min, self.z_max = z, z_orig, z_min, z_max
            self.setup_metadata(args2.seed)
            self.mse_weight = self.args.init_weight

        def synth(self, z):
            if self.args.is_gumbel:
                z_q = vector_quantize(
                    z.movedim(1, 3), self.model.quantize.embed.weight
                ).movedim(3, 1)
            else:
                z_q = vector_quantize(
                    z.movedim(1, 3), self.model.quantize.embedding.weight
                ).movedim(3, 1)
            return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

        def add_metadata(self, path, i):
            imfile = PngImageFile(path)
            meta = PngInfo()
            step_meta = {"iterations": i}
            step_meta.update(self.metadata)
            # meta.add_itxt('vqgan-params', json.dumps(step_meta), zip=True)
            imfile.save(path, pnginfo=meta)
            # Hey you. This one's for Glooperpogger#7353 on Discord (Gloop has a gun), they are a nice snek

        @torch.no_grad()
        def checkin(self, i, losses, x):
            """
            losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
            if i < args.mse_end:
              tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
            else:
              tqdm.write(f'i: {i-args.mse_end} ({i}), loss: {sum(losses).item():g}, losses: {losses_str}')
            tqdm.write(f'cutn: {self.make_cutouts.cutn}, cut_pow: {self.make_cutouts.cut_pow}, step_size: {self.cur_step_size}')
            """
            out = self.synth(self.z.average)

            sys.stdout.flush()
            sys.stdout.write("Saving progress ...\n")
            sys.stdout.flush()

            batchpath = "./"
            TF.to_pil_image(out[0].cpu()).save(args2.image_file)
            if args2.frame_dir is not None:
                import os

                file_list = []
                for file in sorted(os.listdir(args2.frame_dir)):
                    if file.startswith("FRA"):
                        if file.endswith("PNG"):
                            if len(file) == 12:
                                file_list.append(file)
                if file_list:
                    last_name = file_list[-1]
                    count_value = int(last_name[3:8]) + 1
                    count_string = f"{count_value:05d}"
                else:
                    count_string = "00001"
                save_name = args2.frame_dir + "/FRA" + count_string + ".PNG"
                TF.to_pil_image(out[0].cpu()).save(save_name)

            sys.stdout.flush()
            sys.stdout.write("Progress saved\n")
            sys.stdout.flush()

        def unique_index(self, batchpath):
            i = 0
            while i < 10000:
                if os.path.isfile(batchpath + "/" + str(i) + ".png"):
                    i = i + 1
                else:
                    return batchpath + "/" + str(i) + ".png"

        def ascend_txt(self, i):
            out = self.synth(self.z.tensor)
            iii = self.perceptor.encode_image(
                self.normalize(self.make_cutouts(out))
            ).float()

            result = []
            if self.args.init_weight and self.mse_weight > 0:
                result.append(
                    F.mse_loss(self.z.tensor, self.z_orig) * self.mse_weight / 2
                )

            for prompt in self.prompts:
                result.append(prompt(iii))

            if self.usealtprompts:
                iii = self.perceptor.encode_image(
                    self.normalize(self.alt_make_cutouts(out))
                ).float()
                for prompt in self.altprompts:
                    result.append(prompt(iii))

            """      
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        im_path = 'progress.png'
        imageio.imwrite(im_path, np.array(img))
        self.add_metadata(im_path, i)
        """
            return result

        def train(self, i, x):
            self.opt.zero_grad()
            mse_decay = self.args.mse_decay
            mse_decay_rate = self.args.mse_decay_rate
            lossAll = self.ascend_txt(i)

            sys.stdout.write("Iteration {}".format(i) + "\n")
            sys.stdout.flush()

            """
        if i < args.mse_end and i % args.mse_display_freq == 0:
          self.checkin(i, lossAll, x)
        if i == args.mse_end:
          self.checkin(i,lossAll,x)
        if i > args.mse_end and (i-args.mse_end) % args.display_freq == 0:
          self.checkin(i, lossAll, x)
        """
            if i % args2.update == 0:
                self.checkin(i, lossAll, x)

            loss = sum(lossAll)
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                if (
                    self.mse_weight > 0
                    and self.args.init_weight
                    and i > 0
                    and i % mse_decay_rate == 0
                ):
                    if self.args.is_gumbel:
                        self.z_orig = vector_quantize(
                            self.z.average.movedim(1, 3),
                            self.model.quantize.embed.weight,
                        ).movedim(3, 1)
                    else:
                        self.z_orig = vector_quantize(
                            self.z.average.movedim(1, 3),
                            self.model.quantize.embedding.weight,
                        ).movedim(3, 1)
                    if self.mse_weight - mse_decay > 0:
                        self.mse_weight = self.mse_weight - mse_decay
                        # print(f"updated mse weight: {self.mse_weight}")
                    else:
                        self.mse_weight = 0
                        self.make_cutouts = flavordict[flavor](
                            self.perceptor.visual.input_resolution,
                            args.cutn,
                            cut_pow=args.cut_pow,
                            augs=args.augs,
                        )
                        if self.usealtprompts:
                            self.alt_make_cutouts = flavordict[flavor](
                                self.perceptor.visual.input_resolution,
                                args.cutn,
                                cut_pow=args.alt_cut_pow,
                                augs=args.altaugs,
                            )
                        self.z = EMATensor(self.z.average, args.ema_val)
                        self.new_step_size = args.step_size
                        self.opt = torch.optim.AdamW(
                            self.z.parameters(),
                            lr=args.step_size,
                            weight_decay=0.00000000,
                        )
                        # print(f"updated mse weight: {self.mse_weight}")
                if i > args.mse_end:
                    if (
                        args.step_size != args.final_step_size
                        and args.max_iterations > 0
                    ):
                        progress = (i - args.mse_end) / (args.max_iterations)
                        self.cur_step_size = lerp(step_size, final_step_size, progress)
                        for g in self.opt.param_groups:
                            g["lr"] = self.cur_step_size
                # self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

        def run(self, x):
            j = 0
            status.write("Starting the execution...")
            try:
                # pbar = tqdm(range(int(args.max_iterations + args.mse_end)))
                before_start_time = time.perf_counter()
                bar_container = status.container()
                iteration_counter = bar_container.empty()
                progress_bar = bar_container.progress(0)
                total_steps = int(args.max_iterations + args.mse_end) - 1
                for _ in range(total_steps):
                    if j == 0:
                        iteration_counter.empty()
                        imageLocation = stoutput.empty()
                    self.train(j, x)
                    imageLocation.image(Image.open(args2.image_file))
                    if j > 0 and j % args.mse_decay_rate == 0 and self.mse_weight > 0:
                        self.z = EMATensor(self.z.average, args.ema_val)
                        self.opt = torch.optim.AdamW(
                            self.z.parameters(),
                            lr=args.mse_step_size,
                            weight_decay=0.00000000,
                        )
                        # self.opt = optim.Adgarad(self.z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
                    if j >= total_steps:
                        # pbar.close()
                        break
                    self.z.update()
                    j += 1
                    time_past_seconds = time.perf_counter() - before_start_time
                    iterations_per_second = j / time_past_seconds
                    time_left = (total_steps - j) / iterations_per_second
                    percentage = round((j / (total_steps + 1)) * 100)

                    iteration_counter.write(
                        f"{percentage}% {j}/{total_steps+1} [{time.strftime('%M:%S', time.gmtime(time_past_seconds))}<{time.strftime('%M:%S', time.gmtime(time_left))}, {round(iterations_per_second,2)} it/s]"
                    )
                    progress_bar.progress(int(percentage))
                import shutil
                import os

                if not path_exists(DefaultPaths.output_path):
                    os.makedirs(DefaultPaths.output_path)
                save_filename = f"{DefaultPaths.output_path}/{sanitize_filename(args2.prompt)} [{args2.sub_model}] {args2.seed}.png"
                file_list = []
                if path_exists(save_filename):
                    for file in sorted(os.listdir(f"{DefaultPaths.output_path}/")):
                        if file.startswith(
                            f"{sanitize_filename(args2.prompt)} [{args2.sub_model}] {args2.seed}"
                        ):
                            file_list.append(file)
                    last_name = file_list[-1]
                    if last_name[-15:-10] == "batch":
                        count_value = int(last_name[-10:-4]) + 1
                        count_string = f"{count_value:05d}"
                        save_filename = f"{DefaultPaths.output_path}/{sanitize_filename(args2.prompt)} [{args2.sub_model}] {args2.seed}_batch {count_string}.png"
                    else:
                        save_filename = f"{DefaultPaths.output_path}/{sanitize_filename(args2.prompt)} [{args2.sub_model}] {args2.seed}_batch 00001.png"
                shutil.copyfile(
                    args2.image_file,
                    save_filename,
                )
                status.write("Done!")

            except KeyboardInterrupt:
                pass
            except st.script_runner.StopException as e:
                imageLocation.image(args2.image_file)
                torch.cuda.empty_cache()
                status.write("Done!")
                pass
            imageLocation.empty()
            return j

    def add_noise(img):

        # Getting the dimensions of the image
        row, col = img.shape

        # Randomly pick some pixels in the
        # image for coloring them white
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):

            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to white
            img[y_coord][x_coord] = 255

        # Randomly pick some pixels in
        # the image for coloring them black
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):

            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to black
            img[y_coord][x_coord] = 0

        return img

    import io
    import base64

    def image_to_data_url(img, ext):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=ext)
        img_byte_arr = img_byte_arr.getvalue()
        # ext = filename.split('.')[-1]
        prefix = f"data:image/{ext};base64,"
        return prefix + base64.b64encode(img_byte_arr).decode("utf-8")

    import torch
    import math

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def rand_perlin_2d(
        shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
    ):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])

        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
                ),
                dim=-1,
            )
            % 1
        )
        angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = (
            lambda slice1, slice2: gradients[
                slice1[0] : slice1[1], slice2[0] : slice2[1]
            ]
            .repeat_interleave(d[0], 0)
            .repeat_interleave(d[1], 1)
        )
        dot = lambda grad, shift: (
            torch.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[: shape[0], : shape[1]])
        return math.sqrt(2) * torch.lerp(
            torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
        )

    def rand_perlin_2d_octaves(desired_shape, octaves=1, persistence=0.5):
        shape = torch.tensor(desired_shape)
        shape = 2 ** torch.ceil(torch.log2(shape))
        shape = shape.type(torch.int)

        max_octaves = int(
            min(
                octaves,
                math.log(shape[0]) / math.log(2),
                math.log(shape[1]) / math.log(2),
            )
        )
        res = torch.floor(shape / 2**max_octaves).type(torch.int)

        noise = torch.zeros(list(shape))
        frequency = 1
        amplitude = 1
        for _ in range(max_octaves):
            noise += amplitude * rand_perlin_2d(
                shape, (frequency * res[0], frequency * res[1])
            )
            frequency *= 2
            amplitude *= persistence

        return noise[: desired_shape[0], : desired_shape[1]]

    def rand_perlin_rgb(desired_shape, amp=0.1, octaves=6):
        r = rand_perlin_2d_octaves(desired_shape, octaves)
        g = rand_perlin_2d_octaves(desired_shape, octaves)
        b = rand_perlin_2d_octaves(desired_shape, octaves)
        rgb = (torch.stack((r, g, b)) * amp + 1) * 0.5
        return rgb.unsqueeze(0).clip(0, 1).to(device)

    def pyramid_noise_gen(shape, octaves=5, decay=1.0):
        n, c, h, w = shape
        noise = torch.zeros([n, c, 1, 1])
        max_octaves = int(min(math.log(h) / math.log(2), math.log(w) / math.log(2)))
        if octaves is not None and 0 < octaves:
            max_octaves = min(octaves, max_octaves)
        for i in reversed(range(max_octaves)):
            h_cur, w_cur = h // 2**i, w // 2**i
            noise = F.interpolate(
                noise, (h_cur, w_cur), mode="bicubic", align_corners=False
            )
            noise += (torch.randn([n, c, h_cur, w_cur]) / max_octaves) * decay ** (
                max_octaves - (i + 1)
            )
        return noise

    def rand_z(model, toksX, toksY):
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

        return z

    def make_rand_init(
        mode,
        model,
        perlin_octaves,
        perlin_weight,
        pyramid_octaves,
        pyramid_decay,
        toksX,
        toksY,
        f,
    ):

        if mode == "VQGAN ZRand":
            return rand_z(model, toksX, toksY)
        elif mode == "Perlin Noise":
            rand_init = rand_perlin_rgb(
                (toksY * f, toksX * f), perlin_weight, perlin_octaves
            )
            z, *_ = model.encode(rand_init * 2 - 1)
            return z
        elif mode == "Pyramid Noise":
            rand_init = pyramid_noise_gen(
                (1, 3, toksY * f, toksX * f), pyramid_octaves, pyramid_decay
            ).to(device)
            rand_init = (rand_init * 0.5 + 0.5).clip(0, 1)
            z, *_ = model.encode(rand_init * 2 - 1)
            return z

    # Commented out IPython magic to ensure Python compatibility.
    # @title <font color="lightgreen" size="+3">←</font> <font size="+2">💠</font> Selection of models to download <font size="+2">💠</font>
    # @markdown By default, the notebook downloads the 16384 model from ImageNet. There are others like COCO, WikiArt 1024, WikiArt 16384, FacesHQ or S-FLCKR, which are heavy, and if you are not going to use them it would be pointless to download them, so if you want to use them, simply select the models to download. (by the way, COCO 1 Stage is a lighter COCO model. WikiArt 7 Mil is a lighter (and worst) WikiArt model.)
    # %cd /content/

    # import gdown
    import os

    imagenet_1024 = False  # @param {type:"boolean"}
    imagenet_16384 = True  # @param {type:"boolean"}
    gumbel_8192 = False  # @param {type:"boolean"}
    sber_gumbel = False  # @param {type:"boolean"}
    # imagenet_cin = False #@param {type:"boolean"}
    coco = False  # @param {type:"boolean"}
    coco_1stage = False  # @param {type:"boolean"}
    faceshq = False  # @param {type:"boolean"}
    wikiart_1024 = False  # @param {type:"boolean"}
    wikiart_16384 = False  # @param {type:"boolean"}
    wikiart_7mil = False  # @param {type:"boolean"}
    sflckr = False  # @param {type:"boolean"}

    ##@markdown Experimental models (won't probably work, if you know how to make them work, go ahead :D):
    # celebahq = False #@param {type:"boolean"}
    # ade20k = False #@param {type:"boolean"}
    # drin = False #@param {type:"boolean"}
    # gumbel = False #@param {type:"boolean"}
    # gumbel_8192 = False #@param {type:"boolean"}

    """
  if imagenet_1024:
    !curl -L -o vqgan_imagenet_f16_1024.yaml -C - 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 1024
    !curl -L -o vqgan_imagenet_f16_1024.ckpt -C - 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1'  #ImageNet 1024
  if imagenet_16384:
    !curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
    !curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
  if gumbel_8192:
    !curl -L -o gumbel_8192.yaml -C - 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #Gumbel 8192
    !curl -L -o gumbel_8192.ckpt -C - 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #Gumbel 8192
  #if imagenet_cin:
  #  !curl -L -o imagenet_cin.yaml -C - 'https://app.koofr.net/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e?path=%2F2021-04-03T19-39-50_cin_transformer%2Fconfigs%2F2021-04-03T19-39-50-project.yaml' #ImageNet (cIN)
  #  !curl -L -o imagenet_cin.ckpt -C - 'https://app.koofr.net/content/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e/files/get/last.ckpt?path=%2F2021-04-03T19-39-50_cin_transformer%2Fcheckpoints%2Flast.ckpt' #ImageNet (cIN)
  if sber_gumbel:
    models_folder = './'
    configs_folder = './'
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(configs_folder, exist_ok=True)
    models_storage = [
    {
        'id': '1WP6Li2Po8xYcQPGMpmaxIlI1yPB5lF5m',
        'name': 'sber_gumbel.ckpt',
    },
    ]
    configs_storage = [{
        'id': '1M7RvSoiuKBwpF-98sScKng0lsZnwFebR',
        'name': 'sber_gumbel.yaml',
    }]
    url_template = 'https://drive.google.com/uc?id={}'

    for item in models_storage:
        out_name = os.path.join(models_folder, item['name'])
        url = url_template.format(item['id'])
        gdown.download(url, out_name, quiet=True)
    for item in configs_storage:
        out_name = os.path.join(configs_folder, item['name'])
        url = url_template.format(item['id'])
        gdown.download(url, out_name, quiet=True)
  if coco:
    !curl -L -o coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml' #COCO
    !curl -L -o coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt' #COCO
  if faceshq:
    !curl -L -o faceshq.yaml -C - 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT' #FacesHQ
    !curl -L -o faceshq.ckpt -C - 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt' #FacesHQ
  if wikiart_1024:
    #I'm so sorry, I know this is exploiting, but there is no other way.
    !curl -L -o wikiart_1024.yaml -C - 'https://github.com/Eleiber/VQGAN-Mirrors/releases/download/0.0.1/wikiart_1024.yaml' #WikiArt 1024
    !curl -L -o wikiart_1024.ckpt -C - 'https://github.com/Eleiber/VQGAN-Mirrors/releases/download/0.0.1/wikiart_1024.ckpt' #WikiArt 1024
  if wikiart_16384: 
    !curl -L -o wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml' #WikiArt 16384
    !curl -L -o wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' #WikiArt 16384
  if sflckr:
    !curl -L -o sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' #S-FLCKR
    !curl -L -o sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' #S-FLCKR
  if wikiart_7mil:
    !curl -L -o wikiart_7mil.yaml -C - 'http://batbot.tv/ai/models/VQGAN/WikiArt_augmented_Steps_7mil_finetuned_1mil.yaml' #S-FLCKR
    !curl -L -o wikiart_7mil.ckpt -C - 'http://batbot.tv/ai/models/VQGAN/WikiArt_augmented_Steps_7mil_finetuned_1mil.ckpt' #S-FLCKR
  if coco_1stage:
    !curl -L -o coco_1stage.yaml -C - 'http://batbot.tv/ai/models/VQGAN/coco_first_stage.yaml' #S-FLCKR
    !curl -L -o coco_1stage.ckpt -C - 'http://batbot.tv/ai/models/VQGAN/coco_first_stage.ckpt' #S-FLCKR

  #None of these work, if you know how to make them work, go ahead. - Philipuss
  #if celebahq:
  #  !curl -L -o celebahq.yaml -C - 'https://app.koofr.net/content/links/6dddf083-40c8-470a-9360-a9dab2a94e96/files/get/2021-04-23T18-11-19-project.yaml?path=%2F2021-04-23T18-11-19_celebahq_transformer%2Fconfigs%2F2021-04-23T18-11-19-project.yaml&force' #celebahq
  #  !curl -L -o celebahq.ckpt -C - 'https://app.koofr.net/content/links/6dddf083-40c8-470a-9360-a9dab2a94e96/files/get/last.ckpt?path=%2F2021-04-23T18-11-19_celebahq_transformer%2Fcheckpoints%2Flast.ckpt' #celebahq
  #if ade20k:
  #  !curl -L -o ade20k.yaml -C - 'https://app.koofr.net/content/links/0f65c2cd-7102-4550-a2bd-07fd383aac9e/files/get/2020-11-20T21-45-44-project.yaml?path=%2F2020-11-20T21-45-44_ade20k_transformer%2Fconfigs%2F2020-11-20T21-45-44-project.yaml&force' #celebahq
  #  !curl -L -o ade20k.ckpt -C - 'https://app.koofr.net/content/links/0f65c2cd-7102-4550-a2bd-07fd383aac9e/files/get/last.ckpt?path=%2F2020-11-20T21-45-44_ade20k_transformer%2Fcheckpoints%2Flast.ckpt' #celebahq
  #if drin:
  #  !curl -L -o drin.yaml -C - 'https://app.koofr.net/content/links/028f1ba8-404d-42c4-a866-9a8a4eebb40c/files/get/2020-11-20T12-54-32-project.yaml?path=%2F2020-11-20T12-54-32_drin_transformer%2Fconfigs%2F2020-11-20T12-54-32-project.yaml&force' #celebahq
  #  !curl -L -o drin.ckpt -C - 'https://app.koofr.net/content/links/028f1ba8-404d-42c4-a866-9a8a4eebb40c/files/get/last.ckpt?path=%2F2020-11-20T12-54-32_drin_transformer%2Fcheckpoints%2Flast.ckpt' #celebahq

  """
    # Configure and run the model"""

    # Commented out IPython magic to ensure Python compatibility.
    # @title <font color="lightgreen" size="+3">←</font> <font size="+2">🏃‍♂️</font> **Configure & Run** <font size="+2">🏃‍♂️</font>

    import os
    import random
    import cv2

    # from google.colab import drive
    from PIL import Image
    from importlib import reload

    reload(PIL.TiffTags)
    # %cd /content/
    # @markdown >`prompts` is the list of prompts to give to the AI, separated by `|`. With more than one, it will attempt to mix them together. You can add weights to different parts of the prompt by adding a `p:x` at the end of a prompt (before a `|`) where `p` is the prompt and `x` is the weight.

    # prompts = "A fantasy landscape, by Greg Rutkowski. A lush mountain.:1 | Trending on ArtStation, unreal engine. 4K HD, realism.:0.63" #@param {type:"string"}

    prompts = args2.prompt

    width = args2.sizex  # @param {type:"number"}
    height = args2.sizey  # @param {type:"number"}

    sys.stdout.write(f"Loading {args2.vqgan_model} ...\n")
    sys.stdout.flush()
    status.write(f"Loading {args2.vqgan_model} ...\n")

    # model = "ImageNet 16384" #@param ['ImageNet 16384', 'ImageNet 1024', "Gumbel 8192", "Sber Gumbel", 'WikiArt 1024', 'WikiArt 16384', 'WikiArt 7mil', 'COCO-Stuff', 'COCO 1 Stage', 'FacesHQ', 'S-FLCKR']
    model = args2.vqgan_model

    if model == "Gumbel 8192" or model == "Sber Gumbel":
        is_gumbel = True
    else:
        is_gumbel = False

    ##@markdown The flavor effects the output greatly. Each has it's own characteristics and depending on what you choose, you'll get a widely different result with the same prompt and seed. Ginger is the default, nothing special. Cumin results more of a painting, while Holywater makes everythng super funky and/or colorful. Custom is a custom flavor, use the utilities above.
    #   Type "old_holywater" to use the old holywater flavor from Hypertron V1
    flavor = (
        args2.flavor
    )  #'ginger' #@param ["ginger", "cumin", "holywater", "zynth", "wyvern", "aaron", "moth", "juu", "custom"]
    template = (
        args2.template
    )  # @param ["none", "----------Parameter Tweaking----------", "Balanced", "Detailed", "Consistent Creativity", "Realistic", "Smooth", "Subtle MSE", "Hyper Fast Results", "----------Complete Overhaul----------", "flag", "planet", "creature", "human", "----------Sizes----------", "Size: Square", "Size: Landscape", "Size: Poster", "----------Prompt Modifiers----------", "Better - Fast", "Better - Slow", "Movie Poster", "Negative Prompt", "Better Quality"]
    ##@markdown To use initial or target images, upload it on the left in the file browser. You can also use previous outputs by putting its path below, e.g. `batch_01/0.png`. If your previous output is saved to drive, you can use the checkbox so you don't have to type the whole path.
    init = "default noise"  # @param ["default noise", "image", "random image", "salt and pepper noise", "salt and pepper noise on init image"]

    if args2.seed_image is None:
        init_image = ""  # args2.seed_image #""#@param {type:"string"}
    else:
        init_image = args2.seed_image  # ""#@param {type:"string"}

    if init == "random image":
        url = (
            "https://picsum.photos/"
            + str(width)
            + "/"
            + str(height)
            + "?blur="
            + str(random.randrange(5, 10))
        )
        urllib.request.urlretrieve(url, "Init_Img/Image.png")
        init_image = "Init_Img/Image.png"
    elif init == "random image clear":
        url = "https://source.unsplash.com/random/" + str(width) + "x" + str(height)
        urllib.request.urlretrieve(url, "Init_Img/Image.png")
        init_image = "Init_Img/Image.png"
    elif init == "random image clear 2":
        url = "https://loremflickr.com/" + str(width) + "/" + str(height)
        urllib.request.urlretrieve(url, "Init_Img/Image.png")
        init_image = "Init_Img/Image.png"
    elif init == "salt and pepper noise":
        urllib.request.urlretrieve(
            "https://i.stack.imgur.com/olrL8.png", "Init_Img/Image.png"
        )
        import cv2

        img = cv2.imread("Init_Img/Image.png", 0)
        cv2.imwrite("Init_Img/Image.png", add_noise(img))
        init_image = "Init_Img/Image.png"
    elif init == "salt and pepper noise on init image":
        img = cv2.imread(init_image, 0)
        cv2.imwrite("Init_Img/Image.png", add_noise(img))
        init_image = "Init_Img/Image.png"
    elif init == "perlin noise":
        # For some reason Colab started crashing from this
        import noise
        import numpy as np
        from PIL import Image

        shape = (width, height)
        scale = 100
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        seed = np.random.randint(0, 100000)
        world = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                world[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=seed,
                )
        Image.fromarray(prep_world(world)).convert("L").save("Init_Img/Image.png")
        init_image = "Init_Img/Image.png"
    elif init == "black and white":
        url = "https://www.random.org/bitmaps/?format=png&width=300&height=300&zoom=1"
        urllib.request.urlretrieve(url, "Init_Img/Image.png")
        init_image = "Init_Img/Image.png"

    seed = args2.seed  # @param {type:"number"}
    # @markdown >iterations excludes iterations spent during the mse phase, if it is being used. The total iterations will be more if `mse_decay_rate` is more than 0.
    iterations = args2.iterations  # @param {type:"number"}
    transparent_png = False  # @param {type:"boolean"}

    # @markdown <font size="+3">⚠</font> **ADVANCED SETTINGS** <font size="+3">⚠</font>
    # @markdown ---
    # @markdown ---

    # @markdown >If you want to make multiple images with different prompts, use this. Seperate different prompts for different images with a `~` (example: `prompt1~prompt1~prompt3`). Iter is the iterations you want each image to run for. If you use MSE, I'd type a pretty low number (about 10).
    multiple_prompt_batches = False  # @param {type:"boolean"}
    multiple_prompt_batches_iter = 300  # @param {type:"number"}

    # @markdown >`folder_name` is the name of the folder you want to output your result(s) to. Previous outputs will NOT be overwritten. By default, it will be saved to the colab's root folder, but the `save_to_drive` checkbox will save it to `MyDrive\VQGAN_Output` instead.
    folder_name = ""  # @param {type:"string"}
    save_to_drive = False  # @param {type:"boolean"}
    prompt_experiment = "None"  # @param ['None', 'Fever Dream', 'Philipuss’s Basement', 'Vivid Turmoil', 'Mad Dad', 'Platinum', 'Negative Energy']
    if prompt_experiment == "Fever Dream":
        prompts = "<|startoftext|>" + prompts + "<|endoftext|>"
    elif prompt_experiment == "Vivid Turmoil":
        prompts = prompts.replace(" ", "¡")
        prompts = "¬" + prompts + "®"
    elif prompt_experiment == "Mad Dad":
        prompts = prompts.replace(" ", "\\s+")
    elif prompt_experiment == "Platinum":
        prompts = "~!" + prompts + "!~"
        prompts = prompts.replace(" ", "</w>")
    elif prompt_experiment == "Philipuss’s Basement":
        prompts = "<|startoftext|>" + prompts
        prompts = prompts.replace(" ", "<|endoftext|><|startoftext|>")
    elif prompt_experiment == "Lowercase":
        prompts = prompts.lower()

    clip_model = (
        args2.clip_model_1
    )  # "ViT-B/32" #@param ["ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model2 = (
        args2.clip_model_2
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model3 = (
        args2.clip_model_3
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model4 = (
        args2.clip_model_4
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model5 = (
        args2.clip_model_5
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model6 = (
        args2.clip_model_6
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model7 = (
        args2.clip_model_7
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
    clip_model8 = (
        args2.clip_model_8
    )  #'None' #@param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]

    if clip_model2 == "None":
        clip_model2 = None
    if clip_model3 == "None":
        clip_model3 = None
    if clip_model4 == "None":
        clip_model4 = None
    if clip_model5 == "None":
        clip_model5 = None
    if clip_model6 == "None":
        clip_model6 = None
    if clip_model7 == "None":
        clip_model7 = None
    if clip_model8 == "None":
        clip_model8 = None
    # @markdown >Target images work like prompts, write the name of the image. You can add multiple target images by seperating them with a `|`.
    target_images = ""  # @param {type:"string"}

    # @markdown ><font size="+2">☢</font> Advanced values. Values of cut_pow below 1 prioritize structure over detail, and vice versa for above 1. Step_size affects how wild the change between iterations is, and if final_step_size is not 0, step_size will interpolate towards it over time.
    # @markdown >Cutn affects on 'Creativity': less cutout will lead to more random/creative results, sometimes barely readable, while higher values (90+) lead to very stable, photo-like outputs
    cutn = 130  # @param {type:"number"}
    cut_pow = 1  # @param {type:"number"}
    # @markdown >Step_size is like weirdness. Lower: more accurate/realistic, slower; Higher: less accurate/more funky, faster.
    step_size = 0.1  # @param {type:"number"}
    # @markdown >Start_step_size is a temporary step_size that will be active only in the first 10 iterations. It (sometimes) helps with speed. If it's set to 0, it won't be used.
    start_step_size = 0  # @param {type:"number"}
    # @markdown >Final_step_size is a goal step_size which the AI will try and reach. If set to 0, it won't be used.
    final_step_size = 0  # @param {type:"number"}
    if start_step_size <= 0:
        start_step_size = step_size
    if final_step_size <= 0:
        final_step_size = step_size

    # @markdown ---

    # @markdown >EMA maintains a moving average of trained parameters. The number below is the rate of decay (higher means slower).
    ema_val = 0.98  # @param {type:"number"}

    # @markdown >If you want to keep starting from the same point, set `gen_seed` to a positive number. `-1` will make it random every time.
    gen_seed = -1  # @param {type:'number'}

    init_image_in_drive = False  # @param {type:"boolean"}
    if init_image_in_drive and init_image:
        init_image = "/content/drive/MyDrive/VQGAN_Output/" + init_image

    images_interval = args2.update  # @param {type:"number"}

    # I think you should give "Free Thoughts on the Proceedings of the Continental Congress" a read, really funny and actually well-written, Hamilton presented it in a bad light IMO.

    batch_size = 1  # @param {type:"number"}

    # @markdown ---

    # @markdown <font size="+1">🔮</font> **MSE Regulization** <font size="+1">🔮</font>
    # Based off of this notebook: https://colab.research.google.com/drive/1gFn9u3oPOgsNzJWEFmdK-N9h_y65b8fj?usp=sharing - already in credits
    use_mse = args2.mse  # @param {type:"boolean"}
    mse_images_interval = images_interval
    mse_init_weight = 0.2  # @param {type:"number"}
    mse_decay_rate = 160  # @param {type:"number"}
    mse_epoches = 10  # @param {type:"number"}
    ##@param {type:"number"}

    # @markdown >Overwrites the usual values during the mse phase if included. If any value is 0, its normal counterpart is used instead.
    mse_with_zeros = True  # @param {type:"boolean"}
    mse_step_size = 0.87  # @param {type:"number"}
    mse_cutn = 42  # @param {type:"number"}
    mse_cut_pow = 0.75  # @param {type:"number"}

    # @markdown >normal_flip_optim flips between two optimizers during the normal (not MSE) phase. It can improve quality, but it's kind of experimental, use at your own risk.
    normal_flip_optim = True  # @param {type:"boolean"}
    ##@markdown >Adding some TV may make the image blurrier but also helps to get rid of noise. A good value to try might be 0.1.
    # tv_weight = 0.1 #@param {type:'number'}
    # @markdown ---

    # @markdown >`altprompts` is a set of prompts that take in a different augmentation pipeline, and can have their own cut_pow. At the moment, the default "alt augment" settings flip the picture cutouts upside down before evaluating. This can be good for optical illusion images. If either cut_pow value is 0, it will use the same value as the normal prompts.
    altprompts = ""  # @param {type:"string"}
    altprompt_mode = "flipped"
    ##@param ["normal" , "flipped", "sideways"]
    alt_cut_pow = 0  # @param {type:"number"}
    alt_mse_cut_pow = 0  # @param {type:"number"}
    # altprompt_type = "upside-down" #@param ['upside-down', 'as']

    ##@markdown ---
    ##@markdown <font size="+1">💫</font> **Zooming and Moving** <font size="+1">💫</font>
    zoom = False
    ##@param {type:"boolean"}
    zoom_speed = 100
    ##@param {type:"number"}
    zoom_frequency = 20
    ##@param {type:"number"}

    # @markdown ---
    # @markdown On an unrelated note, if you get any errors while running this, restart the runtime and run the first cell again. If that doesn't work either, message me on Discord (Philipuss#4066).

    model_names = {
        "vqgan_imagenet_f16_16384": "vqgan_imagenet_f16_16384",
        "ImageNet 1024": "vqgan_imagenet_f16_1024",
        "Gumbel 8192": "gumbel_8192",
        "Sber Gumbel": "sber_gumbel",
        "imagenet_cin": "imagenet_cin",
        "WikiArt 1024": "wikiart_1024",
        "WikiArt 16384": "wikiart_16384",
        "COCO-Stuff": "coco",
        "FacesHQ": "faceshq",
        "S-FLCKR": "sflckr",
        "WikiArt 7mil": "wikiart_7mil",
        "COCO 1 Stage": "coco_1stage",
    }

    if template == "Better - Fast":
        prompts = prompts + ". Detailed artwork. ArtStationHQ. unreal engine. 4K HD."
    elif template == "Better - Slow":
        prompts = (
            prompts
            + ". Detailed artwork. Trending on ArtStation. unreal engine. | Rendered in Maya. "
            + prompts
            + ". 4K HD."
        )
    elif template == "Movie Poster":
        prompts = prompts + ". Movie poster. Rendered in unreal engine. ArtStationHQ."
        width = 400
        height = 592
    elif template == "flag":
        prompts = (
            "A photo of a flag of the country "
            + prompts
            + " | Flag of "
            + prompts
            + ". White background."
        )
        # import cv2
        # img = cv2.imread('templates/flag.png', 0)
        # cv2.imwrite('templates/final_flag.png', add_noise(img))
        init_image = "templates/flag.png"
        transparent_png = True
    elif template == "planet":
        import cv2

        img = cv2.imread("templates/planet.png", 0)
        cv2.imwrite("templates/final_planet.png", add_noise(img))
        prompts = (
            "A photo of the planet "
            + prompts
            + ". Planet in the middle with black background. | The planet of "
            + prompts
            + ". Photo of a planet. Black background. Trending on ArtStation. | Colorful."
        )
        init_image = "templates/final_planet.png"
    elif template == "creature":
        # import cv2
        # img = cv2.imread('templates/planet.png', 0)
        # cv2.imwrite('templates/final_planet.png', add_noise(img))
        prompts = (
            "A photo of a creature with "
            + prompts
            + ". Animal in the middle with white background. | The creature has "
            + prompts
            + ". Photo of a creature/animal. White background. Detailed image of a creature. | White background."
        )
        init_image = "templates/creature.png"
        # transparent_png = True
    elif template == "Detailed":
        prompts = (
            prompts
            + ", by Puer Udger. Detailed artwork, trending on artstation. 4K HD, realism."
        )
        flavor = "cumin"
    elif template == "human":
        init_image = "/content/templates/human.png"
    elif template == "Realistic":
        cutn = 200
        step_size = 0.03
        cut_pow = 0.2
        flavor = "holywater"
    elif template == "Consistent Creativity":
        flavor = "cumin"
        cut_pow = 0.01
        cutn = 136
        step_size = 0.08
        mse_step_size = 0.41
        mse_cut_pow = 0.3
        ema_val = 0.99
        normal_flip_optim = False
    elif template == "Smooth":
        flavor = "wyvern"
        step_size = 0.10
        cutn = 120
        normal_flip_optim = False
        tv_weight = 10
    elif template == "Subtle MSE":
        mse_init_weight = 0.07
        mse_decay_rate = 130
        mse_step_size = 0.2
        mse_cutn = 100
        mse_cut_pow = 0.6
    elif template == "Balanced":
        cutn = 130
        cut_pow = 1
        step_size = 0.16
        final_step_size = 0
        ema_val = 0.98
        mse_init_weight = 0.2
        mse_decay_rate = 130
        mse_with_zeros = True
        mse_step_size = 0.9
        mse_cutn = 50
        mse_cut_pow = 0.8
        normal_flip_optim = True
    elif template == "Size: Square":
        width = 450
        height = 450
    elif template == "Size: Landscape":
        width = 480
        height = 336
    elif template == "Size: Poster":
        width = 336
        height = 480
    elif template == "Negative Prompt":
        prompts = prompts.replace(":", ":-")
        prompts = prompts.replace(":--", ":")
    elif template == "Hyper Fast Results":
        step_size = 1
        ema_val = 0.3
        cutn = 30
    elif template == "Better Quality":
        prompts = (
            prompts + ":1 | Watermark, blurry, cropped, confusing, cut, incoherent:-1"
        )

    mse_decay = 0

    if use_mse == False:
        mse_init_weight = 0.0
    else:
        mse_decay = mse_init_weight / mse_epoches

    if os.path.isdir("/content/drive") == False:
        if save_to_drive == True or init_image_in_drive == True:
            drive.mount("/content/drive")

    if seed == -1:
        seed = None
    if init_image == "None":
        init_image = None
    if target_images == "None" or not target_images:
        target_images = []
    else:
        target_images = target_images.split("|")
        target_images = [image.strip() for image in target_images]

    prompts = [phrase.strip() for phrase in prompts.split("|")]
    if prompts == [""]:
        prompts = []

    altprompts = [phrase.strip() for phrase in altprompts.split("|")]
    if altprompts == [""]:
        altprompts = []

    if mse_images_interval == 0:
        mse_images_interval = images_interval
    if mse_step_size == 0:
        mse_step_size = step_size
    if mse_cutn == 0:
        mse_cutn = cutn
    if mse_cut_pow == 0:
        mse_cut_pow = cut_pow
    if alt_cut_pow == 0:
        alt_cut_pow = cut_pow
    if alt_mse_cut_pow == 0:
        alt_mse_cut_pow = mse_cut_pow

    augs = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSharpness(0.3, p=0.4),
        K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
        # K.RandomGaussianNoise(p=0.5),
        # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
        K.RandomAffine(
            degrees=30, translate=0.1, p=0.8, padding_mode="border"
        ),  # padding_mode=2
        K.RandomPerspective(
            0.2,
            p=0.4,
        ),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.1),
    )

    if altprompt_mode == "normal":
        altaugs = nn.Sequential(
            K.RandomRotation(degrees=90.0, return_transform=True),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
            # K.RandomGaussianNoise(p=0.5),
            # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(
                degrees=30, translate=0.1, p=0.8, padding_mode="border"
            ),  # padding_mode=2
            K.RandomPerspective(
                0.2,
                p=0.4,
            ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.1),
        )
    elif altprompt_mode == "flipped":
        altaugs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomRotation(degrees=90.0),
            K.RandomVerticalFlip(p=1),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
            # K.RandomGaussianNoise(p=0.5),
            # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(
                degrees=30, translate=0.1, p=0.8, padding_mode="border"
            ),  # padding_mode=2
            K.RandomPerspective(
                0.2,
                p=0.4,
            ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.1),
        )
    elif altprompt_mode == "sideways":
        altaugs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomRotation(degrees=90.0),
            K.RandomVerticalFlip(p=1),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
            # K.RandomGaussianNoise(p=0.5),
            # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(
                degrees=30, translate=0.1, p=0.8, padding_mode="border"
            ),  # padding_mode=2
            K.RandomPerspective(
                0.2,
                p=0.4,
            ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.1),
        )

    if multiple_prompt_batches:
        prompts_all = str(prompts).split("~")
    else:
        prompts_all = prompts
        multiple_prompt_batches_iter = iterations

    if multiple_prompt_batches:
        mtpl_prmpts_btchs = len(prompts_all)
    else:
        mtpl_prmpts_btchs = 1

    # print(mtpl_prmpts_btchs)

    steps_path = "./"
    zoom_path = "./"

    path = "./"

    iterations = multiple_prompt_batches_iter

    for pr in range(0, mtpl_prmpts_btchs):
        # print(prompts_all[pr].replace('[\'', '').replace('\']', ''))
        if multiple_prompt_batches:
            prompts = prompts_all[pr].replace("['", "").replace("']", "")

        if zoom:
            mdf_iter = round(iterations / zoom_frequency)
        else:
            mdf_iter = 2
            zoom_frequency = iterations

        for iter in range(1, mdf_iter):
            if zoom:
                if iter != 0:
                    image = Image.open("progress.png")
                    area = (0, 0, width - zoom_speed, height - zoom_speed)
                    cropped_img = image.crop(area)
                    cropped_img.show()

                    new_image = cropped_img.resize((width, height))
                    new_image.save("zoom.png")
                    init_image = "zoom.png"

            args = argparse.Namespace(
                prompts=prompts,
                altprompts=altprompts,
                image_prompts=target_images,
                noise_prompt_seeds=[],
                noise_prompt_weights=[],
                size=[width, height],
                init_image=init_image,
                png=transparent_png,
                init_weight=mse_init_weight,
                vqgan_model=model_names[model],
                step_size=step_size,
                start_step_size=start_step_size,
                final_step_size=final_step_size,
                cutn=cutn,
                cut_pow=cut_pow,
                mse_cutn=mse_cutn,
                mse_cut_pow=mse_cut_pow,
                mse_step_size=mse_step_size,
                display_freq=images_interval,
                mse_display_freq=mse_images_interval,
                max_iterations=zoom_frequency,
                mse_end=0,
                seed=seed,
                folder_name=folder_name,
                save_to_drive=save_to_drive,
                mse_decay_rate=mse_decay_rate,
                mse_decay=mse_decay,
                mse_with_zeros=mse_with_zeros,
                normal_flip_optim=normal_flip_optim,
                ema_val=ema_val,
                augs=augs,
                altaugs=altaugs,
                alt_cut_pow=alt_cut_pow,
                alt_mse_cut_pow=alt_mse_cut_pow,
                is_gumbel=is_gumbel,
                clip_model=clip_model,
                clip_model2=clip_model2,
                clip_model3=clip_model3,
                clip_model4=clip_model4,
                clip_model5=clip_model5,
                clip_model6=clip_model6,
                clip_model7=clip_model7,
                clip_model8=clip_model8,
                gen_seed=gen_seed,
            )

            mh = ModelHost(args)
            x = 0

            for x in range(batch_size):
                mh.setup_model(x)
                last_iter = mh.run(x)
                x = x + 1

            if batch_size != 1:
                # clear_output()
                # print("===============================================================================")
                q = 0
                while q < batch_size:
                    display(Image("/content/" + folder_name + "/" + str(q) + ".png"))
                    # print("Image" + str(q) + '.png')
                    q += 1

        if zoom:
            files = os.listdir(steps_path)
            for index, file in enumerate(files):
                os.rename(
                    os.path.join(steps_path, file),
                    os.path.join(
                        steps_path,
                        "".join([str(index + 1 + zoom_frequency * iter), ".png"]),
                    ),
                )
                index = index + 1

            from pathlib import Path
            import shutil

            src_path = steps_path
            trg_path = zoom_path

            for src_file in range(1, mdf_iter):
                shutil.move(os.path.join(src_path, src_file), trg_path)
