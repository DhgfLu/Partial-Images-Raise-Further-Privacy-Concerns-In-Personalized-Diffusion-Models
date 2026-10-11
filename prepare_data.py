#!/usr/bin/env python
"""Build the study tree from the shipped image lists and your raw downloads.

    python prepare_data.py

Reads data/celebahq/identities.json, data/dreambooth/subjects.json and
data/customconcept101/concepts.json, copies the listed images from the raw
datasets named in config.RAW into config.DATA, and writes the manifests the
other scripts read (config.DATA/manifests/<dataset>.json).

    <DATA>/celebahq/<identity>/            six fine-tuning images
    <DATA>/celebahq/<identity>_heldout/    the held-out photograph
    <DATA>/dreambooth/<subject>/           all images (used for fine-tuning)
    <DATA>/dreambooth_heldout/<subject>/   all but the held-out image
    <DATA>/dreambooth_heldout/<subject>_heldout/
    <DATA>/customconcept101/<concept>/     six fine-tuning images, centre-cropped to square
    <DATA>/customconcept101/<concept>_heldout/
"""
import argparse
import json
import os
import shutil

from PIL import Image

import config as C

LISTS = f"{C.ROOT}/data"


def load_list(dataset):
    fname = {"celebahq": "identities.json", "dreambooth": "subjects.json",
             "customconcept101": "concepts.json"}[dataset]
    with open(f"{LISTS}/{dataset}/{fname}") as fh:
        d = json.load(fh)
    return d["identities"] if dataset == "celebahq" else d["subjects"] if dataset == "dreambooth" else d["concepts"]


def copy_image(src, dst, square=False):
    """Byte copy; or, for CustomConcept101, a centre crop to the shorter side
    re-encoded at JPEG quality 95, exactly as the paper's copies were made."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if not square:
        shutil.copyfile(src, dst)
        return
    img = Image.open(src).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img.crop((left, top, left + s, top + s)).save(dst, quality=95)


def build(dataset, out):
    raw = C.RAW[dataset]
    entries, manifest = load_list(dataset), []
    heldout_manifest = []                     # dreambooth only: the retrained held-out study
    for i, e in enumerate(entries, 1):
        if dataset == "celebahq":
            subject, cls = str(e["identity"]), "person"
            train, held = e["fine_tuning_images"], e["held_out_image"]
            src = f"{raw}/CelebA-HQ-img"
            ck = f"f{i:02d}_1c"
        elif dataset == "dreambooth":
            subject, cls = e["subject"], e["class_noun"]
            train, held = e["images"], e["held_out_image"]
            src = f"{raw}/{subject}"
            ck = f"db{i:02d}_1c"
        else:
            subject, cls = e["concept"], e["class_noun"]
            train, held = e["fine_tuning_images"], e["held_out_image"]
            src = f"{raw}/{subject}"
            ck = f"cc{i:02d}_1c"
        square = dataset == "customconcept101"
        for x in train:
            copy_image(f"{src}/{x}", f"{out}/{dataset}/{subject}/{x}", square)
        entry = {"index": i, "subject": subject, "checkpoint": ck, "class": cls,
                 "prompt": f"a photo of sks1 {cls}",
                 "train_dir": f"{dataset}/{subject}", "train_images": sorted(train)}
        if dataset == "dreambooth":
            # every image is used for fine-tuning; the held-out study retrains
            # each subject on all images but one, as a separate dataset
            for x in train:
                if x != held:
                    copy_image(f"{src}/{x}", f"{out}/dreambooth_heldout/{subject}/{x}")
            copy_image(f"{src}/{held}", f"{out}/dreambooth_heldout/{subject}_heldout/{held}")
            heldout_manifest.append({
                "index": i, "subject": subject, "checkpoint": f"dbho{i:02d}_1c", "class": cls,
                "prompt": f"a photo of sks1 {cls}",
                "train_dir": f"dreambooth_heldout/{subject}",
                "train_images": sorted(x for x in train if x != held),
                "heldout_image": f"dreambooth_heldout/{subject}_heldout/{held}"})
        else:
            copy_image(f"{src}/{held}", f"{out}/{dataset}/{subject}_heldout/{held}", square)
            entry["heldout_image"] = f"{dataset}/{subject}_heldout/{held}"
        manifest.append(entry)

    os.makedirs(f"{out}/manifests", exist_ok=True)
    with open(f"{out}/manifests/{dataset}.json", "w") as fh:
        json.dump({"dataset": dataset, "subjects": manifest}, fh, indent=1)
    if heldout_manifest:
        with open(f"{out}/manifests/dreambooth_heldout.json", "w") as fh:
            json.dump({"dataset": "dreambooth_heldout", "subjects": heldout_manifest}, fh, indent=1)
    n_img = sum(len(e["train_images"]) for e in manifest)
    print(f"{dataset:18s} {len(manifest)} subjects, {n_img} fine-tuning images, "
          f"{len(manifest)} held-out photographs -> {out}/{dataset}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(C.DATASETS))
    ap.add_argument("--out", default=C.DATA, help="study tree to write (config.DATA)")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        if not os.path.isdir(C.RAW[ds]):
            raise SystemExit(f"raw {ds} not found at {C.RAW[ds]}; set config.RAW or PIAR_RAW_* env")
        build(ds, args.out)


if __name__ == "__main__":
    main()
