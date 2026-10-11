#!/usr/bin/env python
"""Train one DreamBooth victim checkpoint per subject (Appendix A.2).

    python train.py --dataset celebahq --run 1 --gpu 0

Stable Diffusion v1.4, U-Net only, 200 steps per fine-tuning image, learning
rate 2e-6, batch size 1, no prior preservation, 8-bit Adam, fp16, seed
42 + 1000 * run. Each checkpoint is pruned to its fp16 U-Net after training;
the text encoder and VAE are never changed and are read from the base model.
"""
import argparse
import os
import shutil
import subprocess
import sys

import config as C
from piar import manifest as M

TRAINER = f"{C.ROOT}/piar/train_dreambooth.py"


def command(entry, run, out_dir):
    steps = C.STEPS_PER_IMAGE * len(entry["train_images"])
    return [sys.executable, "-m", "accelerate.commands.launch", "--num_processes=1", TRAINER,
            "--pretrained_model_name_or_path", C.WEIGHTS["sd14"],
            "--instance_data_dirs", M.train_dir(entry),
            "--instance_prompts", entry["prompt"],
            "--output_dir", out_dir,
            "--resolution", "512", "--center_crop",
            "--train_batch_size", "1", "--gradient_accumulation_steps", "1",
            "--gradient_checkpointing",
            "--learning_rate", C.LEARNING_RATE, "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
            "--max_train_steps", str(steps), "--checkpointing_steps", str(steps + 1),
            "--use_8bit_adam", "--mixed_precision", "fp16",
            "--seed", str(C.train_seed(run))]


def prune(ckpt_dir):
    """Keep only unet/, in fp16."""
    import torch
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(ckpt_dir, subfolder="unet", torch_dtype=torch.float16)
    tmp = f"{ckpt_dir}/_unet_fp16"
    unet.save_pretrained(tmp)
    for entry in os.listdir(ckpt_dir):
        if entry != "_unet_fp16":
            p = f"{ckpt_dir}/{entry}"
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.rename(tmp, f"{ckpt_dir}/unet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=list(C.DATASETS) + ["dreambooth_heldout"])
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--subjects", default=None, help="comma-separated subset; default all")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = M.subjects(args.dataset)
    if args.subjects:
        want = set(args.subjects.split(","))
        entries = [e for e in entries if e["subject"] in want]
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(f"{C.OUTPUTS}/logs", exist_ok=True)

    for e in entries:
        out = M.checkpoint_dir(e, args.run)
        if M.is_trained(e, args.run):
            print(f"{M.checkpoint_name(e, args.run)}: already trained")
            continue
        cmd = command(e, args.run, out)
        print(f"{M.checkpoint_name(e, args.run)}: {len(e['train_images'])} images, "
              f"{C.STEPS_PER_IMAGE * len(e['train_images'])} steps, seed {C.train_seed(args.run)}")
        if args.dry_run:
            continue
        log = f"{C.OUTPUTS}/logs/train_{M.checkpoint_name(e, args.run)}.log"
        with open(log, "a") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if rc != 0:
            raise SystemExit(f"training failed for {e['subject']}, see {log}")
        prune(out)


if __name__ == "__main__":
    main()
