import gc
import io
import math
import sys
import time 
from PIL import Image, ImageOps
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np
sys.path.append("./glid-3-xl")
from jack_guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from dalle_pytorch import DiscreteVAE, VQGanVAE

from einops import rearrange
from math import log2, sqrt

import argparse
import pickle

import shutil

import os
from os.path import exists as path_exists
sys.path.append("glid-3-xl/encoders")
from encoders.modules import BERTEmbedder

from CLIP import clip
from pathvalidate import sanitize_filename

torch.cuda.empty_cache()
def run_model(args, status, stoutput, DefaultPaths):
    global model, diffusion, ldm, bert, last_model, clip_model, clip_preprocess
    try:
        last_model
    except:
        last_model = ''

    print(args)
    def fetch(url_or_path):
        if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
            r = requests.get(url_or_path)
            r.raise_for_status()
            fd = io.BytesIO()
            fd.write(r.content)
            fd.seek(0)
            return fd
        return open(url_or_path, 'rb')

    class MakeCutouts(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.):
            super().__init__()

            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
            return torch.cat(cutouts)

    def spherical_dist_loss(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    def tv_loss(input):
        """L2 total variation loss, as in Mahendran et al."""
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        return (x_diff**2 + y_diff**2).mean([1, 2, 3])

    device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    print('Using device:', device)
    print(args.model_path)
    model_state_dict = torch.load(args.model_path, map_location='cpu')

    model_params = {
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '27',  # Modify this value to decrease the number of
                                    # timesteps.
        'image_size': 32,
        'learn_sigma': False,
        'noise_schedule': 'linear',
        'num_channels': 320,
        'num_heads': 8,
        'num_res_blocks': 2,
        'resblock_updown': False,
        'use_fp16': False,
        'use_scale_shift_norm': False,
        'clip_embed_dim': 768 if 'clip_proj.weight' in model_state_dict else None,
        'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
        'super_res_condition': True if 'external_block.0.0.weight' in model_state_dict else False,
    }

    if args.ddpm:
        model_params['timestep_respacing'] = 1000
    if args.ddim:
        if args.steps:
            model_params['timestep_respacing'] = 'ddim'+str(args.steps)
        else:
            model_params['timestep_respacing'] = 'ddim50'
    elif args.steps:
        model_params['timestep_respacing'] = str(args.steps)

    model_config = model_and_diffusion_defaults()
    model_config.update(model_params)

    if args.cpu:
        model_config['use_fp16'] = False

    # Load models
    if(last_model == args.model_path):
        try:
            model
            status.write(f"Loading {args.model_path} loaded.")
        except:
            status.write(f"Loading {args.model_path} ...\n")
            model, diffusion = create_model_and_diffusion(**model_config)
            model.load_state_dict(model_state_dict, strict=False)
            model.requires_grad_(args.clip_guidance).eval().to(device)
    else:
        #Yea I should make a function
        status.write(f"Loading {args.model_path} ...\n")
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(model_state_dict, strict=False)
        model.requires_grad_(args.clip_guidance).eval().to(device)
        

    if model_config['use_fp16']:
        model.convert_to_fp16()
    else:
        model.convert_to_fp32()

    def set_requires_grad(model, value):
        for param in model.parameters():
            param.requires_grad = value

    # vae
    try:
        ldm
        set_requires_grad(ldm, args.clip_guidance)    
    except:
        status.write(f"Loading {args.kl_path} ...\n")
        ldm = torch.load(args.kl_path, map_location="cpu")
        ldm.to(device)
        ldm.eval()
        ldm.requires_grad_(args.clip_guidance)
        set_requires_grad(ldm, args.clip_guidance)
    
    try:
        bert
        set_requires_grad(bert, False)
    except:    
        status.write(f"Loading {args.bert_path} ...\n")
        bert = BERTEmbedder(1280, 32)
        sd = torch.load(args.bert_path, map_location="cpu")
        bert.load_state_dict(sd)

        bert.to(device)
        bert.half().eval()
        set_requires_grad(bert, False)

    # clip
    try:
        clip_model
    except:
        clip_model, clip_preprocess = clip.load('ViT-L/14', device=device, jit=False)
        clip_model.eval().requires_grad_(False)
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    def do_run():
        if args.seed >= 0:
            torch.manual_seed(args.seed)

        # bert context
        
        text_emb = bert.encode([args.text]*args.batch_size).to(device).float()
        text_blank = bert.encode([args.negative]*args.batch_size).to(device).float()

        text = clip.tokenize([args.text]*args.batch_size, truncate=True).to(device)
        text_clip_blank = clip.tokenize([args.negative]*args.batch_size, truncate=True).to(device)


        # clip context
        text_emb_clip = clip_model.encode_text(text)
        text_emb_clip_blank = clip_model.encode_text(text_clip_blank)

        make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn)

        text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)

        image_embed = None

        # image context
        if args.edit:
            if args.edit.endswith('.npy'):
                with open(args.edit, 'rb') as f:
                    im = np.load(f)
                    im = torch.from_numpy(im).unsqueeze(0).to(device)

                    input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

                    y = args.edit_y//8
                    x = args.edit_x//8

                    ycrop = y + im.shape[2] - input_image.shape[2]
                    xcrop = x + im.shape[3] - input_image.shape[3]

                    ycrop = ycrop if ycrop > 0 else 0
                    xcrop = xcrop if xcrop > 0 else 0

                    input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

                    input_image_pil = ldm.decode(input_image)
                    input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

                    input_image *= 0.18215
            else:
                w = args.edit_width if args.edit_width else args.width
                h = args.edit_height if args.edit_height else args.height

                input_image_pil = Image.open(fetch(args.edit)).convert('RGB')
                input_image_pil = ImageOps.fit(input_image_pil, (w, h))

                input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

                im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
                im = 2*im-1
                im = ldm.encode(im).sample()

                y = args.edit_y//8
                x = args.edit_x//8

                input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

                ycrop = y + im.shape[2] - input_image.shape[2]
                xcrop = x + im.shape[3] - input_image.shape[3]

                ycrop = ycrop if ycrop > 0 else 0
                xcrop = xcrop if xcrop > 0 else 0

                input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

                input_image_pil = ldm.decode(input_image)
                input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

                input_image *= 0.18215

            if args.mask:
                mask_image = Image.open(fetch(args.mask)).convert('L')
                mask_image = mask_image.resize((args.width//8,args.height//8), Image.ANTIALIAS)
                mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)
            else:
                print('draw the area for inpainting, then close the window')
                app = QApplication(sys.argv)
                d = Draw(args.width, args.height, input_image_pil)
                app.exec_()
                mask_image = d.getCanvas().convert('L').point( lambda p: 255 if p < 1 else 0 )
                mask_image.save('mask.png')
                mask_image = mask_image.resize((args.width//8,args.height//8), Image.ANTIALIAS)
                mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

            mask1 = (mask > 0.5)
            mask1 = mask1.float()

            input_image *= mask1

            image_embed = torch.cat(args.batch_size*2*[input_image], dim=0).float()
        elif model_params['image_condition']:
            # using inpaint model but no image is provided
            image_embed = torch.zeros(args.batch_size*2, 4, args.height//8, args.width//8, device=device)

        kwargs = {
            "context": torch.cat([text_emb, text_blank], dim=0).float(),
            "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float() if model_params['clip_embed_dim'] else None,
            "image_embed": image_embed
        }

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        cur_t = None

        def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
            with torch.enable_grad():
                x = x[:args.batch_size].detach().requires_grad_()

                n = x.shape[0]

                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

                kw = {
                    'context': context[:args.batch_size],
                    'clip_embed': clip_embed[:args.batch_size] if model_params['clip_embed_dim'] else None,
                    'image_embed': image_embed[:args.batch_size] if image_embed is not None else None
                }

                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs=kw)

                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)

                x_in /= 0.18215

                x_img = ldm.decode(x_in)

                clip_in = normalize(make_cutouts(x_img.add(1).div(2)))
                clip_embeds = clip_model.encode_image(clip_in).float()
                dists = spherical_dist_loss(clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0))
                dists = dists.view([args.cutn, n, -1])

                losses = dists.sum(2).mean(0)

                loss = losses.sum() * args.clip_guidance_scale

                return -torch.autograd.grad(loss, x)[0]
    
        if args.ddpm:
            sample_fn = diffusion.ddpm_sample_loop_progressive
        elif args.ddim:
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.plms_sample_loop_progressive

        def save_sample(i, sample, clip_score=False):
            for k, image in enumerate(sample['pred_xstart'][:args.batch_size]):
                image /= 0.18215
                im = image.unsqueeze(0)
                out = ldm.decode(im)

                out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
                out.save(f'{k}-{args.image_file}')
                imageLocationInternal.append(f'{k}-{args.image_file}')
                if clip_score:
                    image_emb = clip_model.encode_image(clip_preprocess(out).unsqueeze(0).to(device))
                    image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)

                    similarity = torch.nn.functional.cosine_similarity(image_emb_norm, text_emb_norm, dim=-1)

                    final_filename = f'output/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.png'
                    #os.rename(filename, final_filename)

                    npy_final = f'output_npy/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.npy'
                    #os.rename(npy_filename, npy_final)
        if args.init_image:
            init = Image.open(args.init_image).convert('RGB')
            init = init.resize((int(args.width),  int(args.height)), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).clamp(0,1)
            h = ldm.encode(init * 2 - 1).sample() *  0.18215
            init = torch.cat(args.batch_size*2*[h], dim=0)
        else:
            init = None
        print(init)
        #image_display = Output()
        for i in range(args.num_batches):
            cur_t = diffusion.num_timesteps - 1
            total_steps = cur_t
            
            status.write("Starting the execution...")
            samples = sample_fn(
                model_fn,
                (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
                clip_denoised=False,
                model_kwargs=kwargs,
                cond_fn=cond_fn if args.clip_guidance else None,
                device=device,
                progress=True,
                init_image=init,
                skip_timesteps=args.skip_timesteps if init is not None else 0,
            )
            itt = 0
            before_start_time = time.perf_counter()
            bar_container = status.container()
            iteration_counter = bar_container.empty()
            progress_bar = bar_container.progress(0)            
            for j, sample in enumerate(samples):
                if(itt==0):
                    iteration_counter.empty()
                    imageLocation = stoutput.empty()
                    
                    #for _ in range(args.batch_size):
                    #    imageLocationInternal.append(stoutput.empty())
                cur_t -= 1
                
                if j % 5 == 0 and j != diffusion.num_timesteps - 1:
                    imageLocationInternal = []
                    #sample.save(args.image_file)
                    save_sample(i, sample)
                
                imageLocation.image(imageLocationInternal)
                itt += 1
                time_past_seconds = time.perf_counter() - before_start_time
                iterations_per_second = itt / time_past_seconds
                time_left = (total_steps - itt) / iterations_per_second
                percentage = round((itt / (total_steps + 1)) * 100)
                iteration_counter.write(
                    f"{percentage}% {itt}/{total_steps+1} [{time.strftime('%M:%S', time.gmtime(time_past_seconds))}<{time.strftime('%M:%S', time.gmtime(time_left))}, {round(iterations_per_second,2)} it/s]"
                )
                progress_bar.progress(int(percentage))
            
            #save_sample(i, sample, args.clip_score)
            if not path_exists(DefaultPaths.output_path):
                os.makedirs(DefaultPaths.output_path)
            save_filename = f"{DefaultPaths.output_path}/{sanitize_filename(args.text)} [GLID-3 XL] {args.seed}.png"
            
            for k in range(args.batch_size):
                shutil.copyfile(
                    f'{k}-{args.image_file}',
                    f'{save_filename[ : -4]}-{k}.png',
                )
            imageLocation.empty()
            status.write("Done!")
    gc.collect()
    do_run()
    last_model = args.model_path