"""Study manifests: which subject trains on which images, and where files live.

A manifest (data/manifests/<dataset>.json, written by prepare_data.py) is
    {"dataset": ..., "subjects": [entry, ...]}
and an entry is
    {"index": 1, "subject": "7613", "checkpoint": "f01_1c", "class": "person",
     "prompt": "a photo of sks1 person",
     "train_dir": "celebahq/7613", "train_images": [...],
     "heldout_image": "celebahq/7613_heldout/7553.jpg"}
with paths relative to config.DATA.
"""
import json
import os

import config as C

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def load(dataset):
    with open(f"{C.DATA}/manifests/{dataset}.json") as fh:
        return json.load(fh)


def subjects(dataset):
    return load(dataset)["subjects"]


def find(dataset, subject):
    for e in subjects(dataset):
        if e["subject"] == subject:
            return e
    raise KeyError(f"{subject!r} is not a subject of {dataset}")


# ------------------------------------------------------------ checkpoints --
def checkpoint_name(entry, run=0):
    """Run 0 keeps the bare name; replicate runs carry a suffix, so no run can
    overwrite another's checkpoint."""
    return entry["checkpoint"] + (f"_r{run}" if run else "")


def checkpoint_dir(entry, run=0):
    return f"{C.OUTPUTS}/checkpoints/{checkpoint_name(entry, run)}"


def is_trained(entry, run=0):
    u = f"{checkpoint_dir(entry, run)}/unet"
    return os.path.isdir(u) and any(f.endswith((".safetensors", ".bin")) for f in os.listdir(u))


def wrong_checkpoint_entry(dataset, entry):
    """Checkpoint-specificity control: the next subject's checkpoint in a fixed
    cyclic order, which never trained on this subject."""
    subs = subjects(dataset)
    other = subs[entry["index"] % len(subs)]          # index is 1-based; last wraps to first
    assert other["subject"] != entry["subject"]
    return other


# ----------------------------------------------------------------- images --
def train_dir(entry):
    return f"{C.DATA}/{entry['train_dir']}"


def train_photos(entry):
    return [f"{train_dir(entry)}/{x}" for x in sorted(entry["train_images"])]


def heldout_photo(entry):
    return f"{C.DATA}/{entry['heldout_image']}"


def masks(tier):
    """Mask stems of a tier, sorted."""
    d = f"{C.MASKS}/{tier}"
    return [f[:-4] for f in sorted(os.listdir(d)) if f.endswith(".png")]


def mask_path(tier, stem):
    return f"{C.MASKS}/{tier}/{stem}.png"


def photo_mask_pairs(entry, tier):
    """The (photograph, mask) units of the main protocol: every fine-tuning
    photograph with one mask assigned by the fixed rotation."""
    photos = train_photos(entry)
    ms = masks(tier)
    return [(p, ms[C.mask_for(entry["index"], i, len(photos), len(ms))])
            for i, p in enumerate(photos)]


def heldout_pairs(entry, tier):
    """The held-out protocol: one held-out photograph with all ten masks."""
    p = heldout_photo(entry)
    return [(p, m) for m in masks(tier)]


# ------------------------------------------------------------- locations --
def group_name(subject, photo_path, mask_stem):
    stem = os.path.splitext(os.path.basename(photo_path))[0]
    return f"{subject}__{stem}__{mask_stem}"


def masked_dir(run, kind, tier, checkpoint):
    """kind: masked | masked_wrong | heldout | ablation_blind | ablation_prompted."""
    return f"{C.OUTPUTS}/run{run}/{kind}/{tier}/{checkpoint}"


def freeform_dir(run, checkpoint, method):
    return f"{C.OUTPUTS}/run{run}/freeform/{checkpoint}/{method}"


def results_path(run, kind, dataset, tier=None):
    name = f"{kind}_{dataset}" + (f"_{tier}" if tier else "") + ".jsonl"
    return f"{C.RESULTS}/run{run}/{name}"


def generations(directory, prefix="gen_"):
    if not os.path.isdir(directory):
        return []
    return [f"{directory}/{f}" for f in sorted(os.listdir(directory))
            if f.startswith(prefix) and f.endswith(".png")]
