"""Load Stable Diffusion v1.4 with BrushNet, and turn a photograph and a mask
into the tensors the sampler needs."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from .brushnet import BrushNetModel, UNet2DConditionModel, StableDiffusionPowerPaintBrushNetPipeline


def load_inpainting_pipeline(unet_path, device="cuda", dtype=torch.float16):
    """SD-1.4 with the U-Net taken from `unet_path` (a personalized checkpoint,
    or the pretrained model itself) and the public random-mask BrushNet attached.
    Both branches of PIAR are built this way and differ only in the U-Net."""
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", torch_dtype=dtype)
    brushnet = BrushNetModel.from_pretrained(C.WEIGHTS["brushnet"], torch_dtype=dtype)
    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        C.WEIGHTS["sd14"], unet=unet, brushnet=brushnet,
        safety_checker=None, requires_safety_checker=False, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_text2img_pipeline(unet_path, device="cuda", dtype=torch.float16):
    """Plain SD-1.4 text-to-image pipeline for the free-form baselines, with the
    U-Net swapped for a personalized one when `unet_path` is a checkpoint."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel as PlainUNet
    kw = {}
    if unet_path != C.WEIGHTS["sd14"]:
        kw["unet"] = PlainUNet.from_pretrained(unet_path, subfolder="unet", torch_dtype=dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        C.WEIGHTS["sd14"], safety_checker=None, requires_safety_checker=False,
        torch_dtype=dtype, **kw)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@torch.no_grad()
def encode_prompt(pipe, text, device="cuda", dtype=torch.float16):
    ids = pipe.tokenizer(text, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=77).input_ids.to(device)
    return pipe.text_encoder(ids)[0].to(dtype)


def masked_input(photo_path, mask_path, size=512):
    """The attacker's view: the photograph with the masked region set to black,
    plus the mask itself (white = masked)."""
    img = np.array(Image.open(photo_path).convert("RGB").resize((size, size)))
    m = np.array(Image.open(mask_path).convert("L").resize((size, size)))
    img[m > 128] = 0
    return Image.fromarray(img), Image.open(mask_path).convert("RGB").resize((size, size))


@torch.no_grad()
def prepare_condition(pipe, masked_pil, mask_pil, device="cuda", dtype=torch.float16, size=512, seed=None):
    """BrushNet conditioning [1, 5, 64, 64]: VAE latents of the masked image,
    followed by one channel that is 1 on the known region. The VAE latent is a
    sample; `seed` makes it deterministic."""
    image = pipe.image_processor.preprocess(masked_pil, height=size, width=size).to(device=device, dtype=dtype)
    mask = pipe.image_processor.preprocess(mask_pil, height=size, width=size)
    masked = (mask.sum(dim=1, keepdim=True) > 0).to(dtype=dtype).to(device)     # 1 = masked
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    latents = pipe.vae.encode(image).latent_dist.sample(generator=gen) * pipe.vae.config.scaling_factor
    masked = F.interpolate(masked, size=latents.shape[-2:], mode="nearest")
    return torch.cat([latents, 1.0 - masked], dim=1)
