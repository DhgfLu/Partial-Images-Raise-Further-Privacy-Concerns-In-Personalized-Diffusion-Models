#!/usr/bin/env python
"""Free-form extraction baselines: CFG and FineXtract.

    python extract.py --dataset celebahq --run 1

Generates 50 * N0 images per checkpoint and method from the training prompt
(FineXtract's budget), to <OUTPUTS>/run<r>/freeform/<checkpoint>/<method>/gen_*.png.
Both methods use the same seed, so they share initial noise.
"""
import argparse
import os
import time

import torch

import config as C
from piar import manifest as M
from piar.freeform import METHODS, extract
from piar.pipeline import encode_prompt, load_text2img_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--subjects", default=None, help="comma-separated subset; default all")
    ap.add_argument("--method", default="both", choices=list(METHODS) + ["both"])
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=C.DDIM_STEPS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    entries = M.subjects(args.dataset)
    if args.subjects:
        want = set(args.subjects.split(","))
        entries = [e for e in entries if e["subject"] in want]
    methods = list(METHODS) if args.method == "both" else [args.method]
    total = sum(C.FREEFORM_PER_IMAGE * len(e["train_images"]) for e in entries) * len(methods)
    print(f"{args.dataset} run {args.run}: {len(entries)} subjects, {methods}, {total:,} images")
    if args.dry_run:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pretrained = load_text2img_pipeline(C.WEIGHTS["sd14"], device, dtype) if "finextract" in methods else None
    empty_0 = encode_prompt(pretrained, "", device, dtype) if pretrained else None
    t0 = time.time()
    for e in entries:
        ck = M.checkpoint_name(e, args.run)
        n = C.FREEFORM_PER_IMAGE * len(e["train_images"])
        todo = [m for m in methods if len(M.generations(M.freeform_dir(args.run, ck, m))) < n]
        if not todo:
            continue
        if not M.is_trained(e, args.run):
            raise SystemExit(f"checkpoint {ck} is not trained; run train.py first")
        personalized = load_text2img_pipeline(M.checkpoint_dir(e, args.run), device, dtype)
        prompt = encode_prompt(personalized, e["prompt"], device, dtype)
        empty_p = encode_prompt(personalized, "", device, dtype)
        seed = C.freeform_seed(ck, e["subject"], args.run)
        for m in todo:
            out = M.freeform_dir(args.run, ck, m)
            os.makedirs(out, exist_ok=True)
            images = extract(personalized, pretrained, prompt, empty_p, empty_0, m, n, seed,
                             args.steps, args.batch_size)
            for k, im in enumerate(images):
                im.save(f"{out}/gen_{k:04d}.png")
            print(f"  {ck} {m}: {n} images  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
