"""Masked-image reconstruction with contrastive guidance (Section 4).

    eps_hat = eps_p + (w - 1) * (eps_p - eps_0)                      (Eq. 2)

eps_p and eps_0 are the noise predictions of the personalized and the pretrained
branch under the same latent, masked input, mask and prompt. w = 0 is Base
Inpainting, w = 1 is Personalized Inpainting, w = 3 is PIAR.
"""
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

import config as C


def predict_noise(pipe, scheduler, latents, t, cond, embeds):
    """One BrushNet -> U-Net pass of one branch."""
    x = scheduler.scale_model_input(latents, t)
    down, mid, up = pipe.brushnet(x, t, encoder_hidden_states=embeds, brushnet_cond=cond,
                                  conditioning_scale=1.0, guess_mode=False, return_dict=False)
    return pipe.unet(x, t, encoder_hidden_states=embeds, down_block_add_samples=down,
                     mid_block_add_sample=mid, up_block_add_samples=up, return_dict=False)[0]


@torch.no_grad()
def reconstruct(pipe_personalized, pipe_pretrained, cond, embeds_p, embeds_0, w, n, seed,
                steps=C.DDIM_STEPS, batch_size=64):
    """n candidate reconstructions for one masked input at guidance weight w.

    Candidate k is drawn from its own generator seeded with seed + k, so the
    first m of n candidates are exactly what an m-candidate run would produce.
    Only the branches the formula needs are evaluated: w = 0 uses the pretrained
    branch alone and w = 1 the personalized one alone, which is identical to
    evaluating both and cancelling."""
    device, dtype = cond.device, cond.dtype
    need_p, need_0 = w != 0, w != 1
    out = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        latents = torch.cat([
            torch.randn((1, 4, 64, 64), device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(seed + start + k))
            for k in range(b)])
        sched = DDIMScheduler.from_config(pipe_personalized.scheduler.config)
        sched.set_timesteps(steps, device=device)
        latents = latents * sched.init_noise_sigma
        c = cond.expand(b, *cond.shape[1:]).contiguous()
        e_p = embeds_p.expand(b, *embeds_p.shape[1:]).contiguous()
        e_0 = embeds_0.expand(b, *embeds_0.shape[1:]).contiguous()
        for t in sched.timesteps:
            eps_p = predict_noise(pipe_personalized, sched, latents, t, c, e_p) if need_p else None
            eps_0 = predict_noise(pipe_pretrained, sched, latents, t, c, e_0) if need_0 else None
            if not need_0:
                eps = eps_p
            elif not need_p:
                eps = eps_0
            else:
                eps = eps_p + (w - 1.0) * (eps_p - eps_0)
            latents = sched.step(eps, t, latents).prev_sample
        vae = pipe_personalized.vae
        img = vae.decode(latents / vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
        out.extend(Image.fromarray((x * 255).astype(np.uint8)) for x in img)
    return out
