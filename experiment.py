#!/usr/bin/env python
"""The paper's experiments.

    python experiment.py main               # Tables 1-3 and 5: one run
    python experiment.py main --runs 1-5    # the paper's mean ± SD over five runs
    python experiment.py sweep              # Tables 6-8, 12-14: run 1 at every mask tier, 50 candidates
    python experiment.py heldout            # Tables 9-11
    python experiment.py ablation           # Tables 15-16
    python tables.py

Each command goes subject by subject: train, generate, score, then remove the
subject's checkpoint and generated images (--keep retains them). The score rows
in results/ are what remains. Rerunning skips subjects already scored.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import config as C
from piar import manifest as M


def sh(script, *args):
    cmd = [sys.executable, f"{C.ROOT}/{script}", *[str(a) for a in args]]
    print("$", " ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        raise SystemExit(f"{script} failed")


def remove(*paths):
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)


def scored(run, kind, dataset, subject, tier=None):
    """Whether the subject has rows in the last file a scoring call writes."""
    path = M.results_path(run, kind, dataset, tier)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        return any(json.loads(line)["subject"] == subject for line in fh)


def masked_dirs(run, kind, tiers, entry):
    ck = M.checkpoint_name(entry, run)
    return [M.masked_dir(run, kind, t, ck) for t in tiers]


def gpu_args(gpu):
    return ["--gpu", gpu] if gpu is not None else []


def parse_runs(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main_run(run, datasets, gpu, keep, sweep=False):
    """One run: a victim per subject, the three masked arms on every fine-tuning
    photograph, the two free-form baselines, and the checkpoint-specificity
    control, which attacks a subject with the next subject's checkpoint. With
    sweep (run 1), also the other mask tiers and 50-candidate pools."""
    g = gpu_args(gpu)
    tiers = list(C.MASK_TIERS) if sweep else [C.MAIN_TIER]
    what = "masked,wrong,freeform" + (",budget" if sweep else "")
    for ds in datasets:
        subs = M.subjects(ds)

        def done(e):
            return scored(run, "budget" if sweep else "freeform", ds, e["subject"])

        def finish(e):
            """The control for e needs the next checkpoint, so e is scored once
            that one is trained; the first checkpoint stays for the last subject."""
            sh("attack.py", "--dataset", ds, "--run", run, "--subjects", e["subject"],
               "--target", "wrong", "--w", "1,3", *g)
            sh("score.py", "--dataset", ds, "--run", run, "--subjects", e["subject"],
               "--what", what, "--tier", "all" if sweep else C.MAIN_TIER, *g)
            if not keep:
                ck = M.checkpoint_name(e, run)
                wrong = M.checkpoint_name(M.wrong_checkpoint_entry(ds, e), run)
                remove(*masked_dirs(run, "masked", tiers, e),
                       M.masked_dir(run, "masked_wrong", C.MAIN_TIER, wrong),
                       os.path.dirname(M.freeform_dir(run, ck, "cfg")))
                if e["index"] != subs[0]["index"]:
                    remove(M.checkpoint_dir(e, run))

        for i, e in enumerate(subs):
            if done(e):
                continue
            sh("train.py", "--dataset", ds, "--run", run, "--subjects", e["subject"], *g)
            if sweep:
                sh("attack.py", "--dataset", ds, "--run", run, "--subjects", e["subject"],
                   "--tier", C.MAIN_TIER, "--n", C.N_MASKED_SWEEP, *g)
            sh("attack.py", "--dataset", ds, "--run", run, "--subjects", e["subject"],
               "--tier", "all" if sweep else C.MAIN_TIER, *g)
            sh("extract.py", "--dataset", ds, "--run", run, "--subjects", e["subject"], *g)
            if i and not done(subs[i - 1]):
                finish(subs[i - 1])
        if not done(subs[-1]):
            finish(subs[-1])
        if not keep:
            remove(M.checkpoint_dir(subs[0], run))


def heldout(gpu, keep):
    """Run 0: each subject's held-out photograph under all ten masks of every tier."""
    g = gpu_args(gpu)
    for ds in ("celebahq", "customconcept101", "dreambooth_heldout"):
        for e in M.subjects(ds):
            if scored(0, "heldout", ds, e["subject"], C.MASK_TIERS[-1]):
                continue
            sh("train.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"], *g)
            sh("attack.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"],
               "--target", "heldout", "--tier", "all", *g)
            sh("score.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"],
               "--what", "heldout", "--tier", "all", "--n", C.N_MASKED, *g)
            if not keep:
                remove(M.checkpoint_dir(e, 0), *masked_dirs(0, "heldout", C.MASK_TIERS, e))


def ablation(datasets, gpu, keep):
    """Run 0 on six subjects per dataset, drawn with a fixed seed: blind
    reconstruction at w = 0..5 and prompted reconstruction at w = 3."""
    g = gpu_args(gpu)
    for ds in datasets:
        subjects = sorted(M.subjects(ds), key=lambda e: e["index"])
        picked = sorted(random.Random(C.ABLATION_SEED).sample(subjects, C.ABLATION_N_SUBJECTS),
                        key=lambda e: e["index"])
        for e in picked:
            if scored(0, "ablation_prompted", ds, e["subject"]):
                continue
            sh("train.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"], *g)
            sh("attack.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"],
               "--w", ",".join(f"{w:g}" for w in C.ABLATION_W), "--kind", "ablation_blind", *g)
            sh("attack.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"],
               "--w", f"{C.W_PIAR:g}", "--prompt", "training", "--kind", "ablation_prompted", *g)
            sh("score.py", "--dataset", ds, "--run", 0, "--subjects", e["subject"], "--what", "ablation", *g)
            if not keep:
                remove(M.checkpoint_dir(e, 0),
                       *masked_dirs(0, "ablation_blind", [C.MAIN_TIER], e),
                       *masked_dirs(0, "ablation_prompted", [C.MAIN_TIER], e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("main", "sweep", "heldout", "ablation"):
        p = sub.add_parser(name)
        p.add_argument("--gpu", default=None)
        p.add_argument("--keep", action="store_true", help="keep checkpoints and generated images")
        if name != "heldout":
            p.add_argument("--datasets", default=",".join(C.DATASETS))
        if name == "main":
            p.add_argument("--runs", default="1", help="e.g. 1-5 or 2,3")
    args = ap.parse_args()
    if args.cmd == "main":
        for run in parse_runs(args.runs):
            main_run(run, args.datasets.split(","), args.gpu, args.keep)
    elif args.cmd == "sweep":
        main_run(1, args.datasets.split(","), args.gpu, args.keep, sweep=True)
    elif args.cmd == "heldout":
        heldout(args.gpu, args.keep)
    else:
        ablation(args.datasets.split(","), args.gpu, args.keep)


if __name__ == "__main__":
    main()
