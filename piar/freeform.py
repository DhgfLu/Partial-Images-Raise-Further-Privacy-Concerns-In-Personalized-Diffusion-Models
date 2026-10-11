"""The free-form baselines: CFG and FineXtract (Appendix A.4).

Both prompt the personalized model with its training prompt and differ in the
unconditional branch:
    CFG         eps = eps_p("") + s  * (eps_p(c) - eps_p(""))
    FineXtract  eps = eps_0("") + w' * (eps_p(c) - eps_0("")) + k * eps_0("")
with s = 3, w' = 3 and k = -0.02.
"""
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

import config as C

METHODS = ("cfg", "finextract")


@torch.no_grad()
def extract(pipe_personalized, pipe_pretrained, prompt_embeds, empty_p, empty_0, method, n, seed,
            steps=C.DDIM_STEPS, batch_size=64):
    """n free-form generations from the personalized model. `pipe_pretrained`
    is only used by FineXtract and may be None for CFG."""
    assert method in METHODS
    cross = method == "finextract"
    scale = C.FINEXTRACT_W if cross else C.CFG_SCALE
    k = C.FINEXTRACT_K if cross else 0.0
    unet_uncond = (pipe_pretrained if cross else pipe_personalized).unet
    empty = empty_0 if cross else empty_p
    device, dtype = prompt_embeds.device, prompt_embeds.dtype
    out = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        sched = DDIMScheduler.from_config(pipe_personalized.scheduler.config)
        sched.set_timesteps(steps, device=device)
        latents = torch.cat([
            torch.randn((1, 4, 64, 64), device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(seed + start + i))
            for i in range(b)]) * sched.init_noise_sigma
        e_c = prompt_embeds.expand(b, *prompt_embeds.shape[1:]).contiguous()
        e_u = empty.expand(b, *empty.shape[1:]).contiguous()
        for t in sched.timesteps:
            x = sched.scale_model_input(latents, t)
            eps_c = pipe_personalized.unet(x, t, encoder_hidden_states=e_c).sample
            eps_u = unet_uncond(x, t, encoder_hidden_states=e_u).sample
            eps = eps_u + scale * (eps_c - eps_u) + k * eps_u
            latents = sched.step(eps, t, latents).prev_sample
        vae = pipe_personalized.vae
        img = vae.decode(latents / vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
        out.extend(Image.fromarray((x * 255).astype(np.uint8)) for x in img)
    return out
