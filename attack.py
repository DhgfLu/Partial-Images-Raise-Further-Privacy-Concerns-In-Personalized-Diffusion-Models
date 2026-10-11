#!/usr/bin/env python
"""Masked-image reconstruction: Base Inpainting, Personalized Inpainting and PIAR.

    python attack.py --dataset celebahq --run 1                    # main protocol, small tier
    python attack.py --dataset celebahq --run 1 --tier all --n 50  # every tier, budget-sweep pools
    python attack.py --dataset celebahq --run 1 --target wrong     # checkpoint-specificity control
    python attack.py --dataset celebahq --target heldout --tier all

--target train    every fine-tuning photograph, one mask each (fixed rotation)
--target wrong    the same units, but the next subject's checkpoint supplies the
                  personalized branch (Table 2)
--target heldout  the held-out photograph under all ten masks of the tier

Candidates go to <OUTPUTS>/run<r>/<kind>/<tier>/<checkpoint>/w<w>/<subject>__<photo>__<mask>/gen_*.png.
"""
import argparse
import os
import time

import torch

import config as C
from piar import manifest as M
from piar.guidance import reconstruct
from piar.pipeline import encode_prompt, load_inpainting_pipeline, masked_input, prepare_condition


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS) + ["dreambooth_heldout"])
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--subjects", default=None, help="comma-separated subset; default all")
    ap.add_argument("--tier", default=C.MAIN_TIER, help="mask tier, or 'all'")
    ap.add_argument("--target", default="train", choices=["train", "wrong", "heldout"])
    ap.add_argument("--w", default=None,
                    help="guidance weights, comma-separated; default 0,1,3, or 1,3 for --target wrong "
                         "(Base Inpainting never uses the personalized checkpoint)")
    ap.add_argument("--n", type=int, default=C.N_MASKED, help="candidates per masked input")
    ap.add_argument("--prompt", default="",
                    help="attack prompt: empty (blind, the paper's default), any text, or "
                         "'training' for each subject's own training prompt")
    ap.add_argument("--kind", default=None,
                    help="output folder name; default masked / masked_wrong / heldout by --target")
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
    tiers = list(C.MASK_TIERS) if args.tier == "all" else [args.tier]
    ws = [float(x) for x in (args.w or ("1,3" if args.target == "wrong" else "0,1,3")).split(",")]
    kind = args.kind or {"train": "masked", "wrong": "masked_wrong", "heldout": "heldout"}[args.target]
    pairs_of = M.heldout_pairs if args.target == "heldout" else M.photo_mask_pairs

    jobs = []                                  # (entry, checkpoint entry, tier, photo, mask)
    for e in entries:
        if args.target == "heldout" and "heldout_image" not in e:
            raise SystemExit(f"{args.dataset} has no held-out photographs in its manifest; "
                             "for DreamBooth use --dataset dreambooth_heldout")
        ck_entry = M.wrong_checkpoint_entry(args.dataset, e) if args.target == "wrong" else e
        for tier in tiers:
            for photo, mask in pairs_of(e, tier):
                jobs.append((e, ck_entry, tier, photo, mask))
    print(f"{args.dataset} run {args.run}: {len(entries)} subjects, {len(jobs)} masked inputs, "
          f"w={ws}, N={args.n}, kind={kind}, prompt={args.prompt!r}")
    if args.dry_run:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pretrained = load_inpainting_pipeline(C.WEIGHTS["sd14"], device, dtype)
    embeds_0 = {}                              # prompt text -> pretrained-branch embedding
    personalized, loaded = None, None
    t0, done = time.time(), 0
    for e, ck_entry, tier, photo, mask in jobs:
        prompt = e["prompt"] if args.prompt == "training" else args.prompt
        ck = M.checkpoint_name(ck_entry, args.run)
        if loaded != ck:
            if not M.is_trained(ck_entry, args.run):
                raise SystemExit(f"checkpoint {ck} is not trained; run train.py first")
            if personalized is not None:
                del personalized
                if device == "cuda":
                    torch.cuda.empty_cache()
            personalized = load_inpainting_pipeline(M.checkpoint_dir(ck_entry, args.run), device, dtype)
            loaded, embeds_p = ck, {}
        if prompt not in embeds_p:
            embeds_p[prompt] = encode_prompt(personalized, prompt, device, dtype)
        if prompt not in embeds_0:
            embeds_0[prompt] = encode_prompt(pretrained, prompt, device, dtype)
        masked_pil, mask_pil = masked_input(photo, M.mask_path(tier, mask))
        cond = None
        for w in ws:
            group = f"{M.masked_dir(args.run, kind, tier, ck)}/w{w:g}/{M.group_name(e['subject'], photo, mask)}"
            if len(M.generations(group)) >= args.n:
                done += 1
                continue
            os.makedirs(group, exist_ok=True)
            masked_pil.save(f"{group}/masked_input.png")
            if cond is None:
                cond = prepare_condition(personalized, masked_pil, mask_pil, device, dtype,
                                         seed=C.seed_for(e["subject"], mask, args.run))
            images = reconstruct(personalized, pretrained, cond, embeds_p[prompt], embeds_0[prompt], w, args.n,
                                 C.seed_for(e["subject"], mask, args.run), args.steps, args.batch_size)
            for k, im in enumerate(images):
                im.save(f"{group}/gen_{k:03d}.png")
            done += 1
        print(f"  [{done}/{len(jobs) * len(ws)}] {e['subject']} {tier} {os.path.basename(photo)} {mask}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
