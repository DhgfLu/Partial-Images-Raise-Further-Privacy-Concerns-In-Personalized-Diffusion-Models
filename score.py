#!/usr/bin/env python
"""Score every reconstruction and extracted image against its target.

    python score.py --dataset celebahq --run 1                       # masked, wrong, freeform
    python score.py --dataset celebahq --run 1 --what masked --tier all
    python score.py --dataset celebahq --run 1 --what budget          # free-form at every budget
    python score.py --dataset celebahq --run 0 --what heldout --tier all
    python score.py --dataset celebahq --run 0 --what ablation
    python score.py --dataset celebahq --run 1 --subjects 7613

Masked arms: for each masked input, the SSCD medoid of the first N candidates is
scored against the target photograph, for every N in config.BUDGETS that the
pool allows. Free-form: the pool is clustered into N0 extracted images, and each
fine-tuning photograph is matched to its most similar one.

One row per evaluated unit, to results/run<r>/<what>_<dataset>[_<tier>].jsonl.
The rows of the subjects scored replace their earlier rows; other subjects' rows stay.
"""
import argparse
import glob
import json
import os
import time

import config as C
from piar import manifest as M
from piar import metrics
from piar.selection import extract_cliques, medoid

MASKED_KINDS = {"masked": "train", "wrong": "wrong", "heldout": "heldout",
                "ablation_blind": "blind", "ablation_prompted": "prompted"}
FOLDER = {"masked": "masked", "wrong": "masked_wrong", "heldout": "heldout",
          "ablation_blind": "ablation_blind", "ablation_prompted": "ablation_prompted"}


def resolve_photo(entry, stem, heldout):
    """The target photograph behind a group name's photo stem."""
    if heldout:
        return M.heldout_photo(entry)
    for p in M.train_photos(entry):
        if os.path.splitext(os.path.basename(p))[0] == stem:
            return p
    raise KeyError(f"{stem} is not a fine-tuning image of {entry['subject']}")


def score_masked(dataset, run, what, tier, budgets, faces, out, subjects=None):
    entries = {e["subject"]: e for e in M.subjects(dataset)}
    root = f"{C.OUTPUTS}/run{run}/{FOLDER[what]}/{tier}"
    rows, missing = [], 0
    for ck_dir in sorted(glob.glob(f"{root}/*")):
        ck = os.path.basename(ck_dir)
        for w_dir in sorted(glob.glob(f"{ck_dir}/w*")):
            w = float(os.path.basename(w_dir)[1:])
            for group in sorted(glob.glob(f"{w_dir}/*")):
                subject, stem, mask = os.path.basename(group).split("__")
                if subjects and subject not in subjects:
                    continue
                gens = M.generations(group)
                if not gens:
                    missing += 1
                    continue
                e = entries[subject]
                target = resolve_photo(e, stem, what == "heldout")
                feats = metrics.sscd_features(gens)
                for n in budgets:
                    if n > len(gens):
                        continue
                    pick = gens[medoid(feats[:n])]
                    r = metrics.score_pair(pick, target, faces=faces)
                    rows.append({"dataset": dataset, "run": run, "condition": MASKED_KINDS[what],
                                 "subject": subject, "checkpoint": M.checkpoint_name(e, run),
                                 "wrong_checkpoint": ck if what == "wrong" else None,
                                 "arm": C.ARMS.get(w, f"w{w:g}"), "w": w, "tier": tier,
                                 "photo": os.path.basename(target), "mask": mask, "n": n,
                                 "selected": os.path.basename(pick), **r})
    write(rows, out)
    print(f"  {what} {tier}: {len(rows)} rows" + (f", {missing} empty groups" if missing else ""))


def score_freeform(dataset, run, budgets_per_image, faces, out, subjects=None):
    rows = []
    for e in M.subjects(dataset):
        if subjects and e["subject"] not in subjects:
            continue
        ck = M.checkpoint_name(e, run)
        real = M.train_photos(e)
        n0 = len(real)
        for method in ("cfg", "finextract"):
            pool = M.generations(M.freeform_dir(run, ck, method))
            if not pool:
                continue
            feats = metrics.sscd_features(pool)
            for x in budgets_per_image:
                total = x * n0
                if total > len(pool):
                    continue
                picks, n_cliques, th = extract_cliques(feats[:total], n0)
                kept = [pool[i] for i in picks]
                for r in metrics.score_pool(kept, real, faces=faces):
                    rows.append({"dataset": dataset, "run": run, "subject": e["subject"],
                                 "checkpoint": ck, "method": method,
                                 "budget_per_image": x, "budget_total": total,
                                 "n_pool": total, "n_kept": len(kept), "n_cliques": n_cliques,
                                 "threshold": th, "photo": os.path.basename(r["photo"]),
                                 "matched": os.path.basename(r["matched"]),
                                 **{k: r[k] for k in ("sscd", "lpips", "ssim", "clip", "arcface")}})
    write(rows, out)
    print(f"  freeform: {len(rows)} rows")


def write(rows, path):
    """Replace the rows of the subjects just scored, keep everyone else's."""
    scored = {r["subject"] for r in rows}
    if os.path.exists(path):
        with open(path) as fh:
            rows = [r for r in map(json.loads, fh) if r["subject"] not in scored] + rows
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS) + ["dreambooth_heldout"])
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--subjects", default=None, help="comma-separated subset; default all")
    ap.add_argument("--what", default="masked,wrong,freeform",
                    help="comma-separated: masked, wrong, freeform, budget, heldout, ablation")
    ap.add_argument("--tier", default=C.MAIN_TIER, help="mask tier, or 'all'")
    ap.add_argument("--n", default=",".join(str(b) for b in C.BUDGETS),
                    help="masked budgets to score, comma-separated")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    faces = args.dataset == "celebahq"
    subjects = set(args.subjects.split(",")) if args.subjects else None
    tiers = list(C.MASK_TIERS) if args.tier == "all" else [args.tier]
    budgets = sorted(int(x) for x in args.n.split(","))
    t0 = time.time()
    for what in args.what.split(","):
        if what in ("masked", "heldout"):
            for tier in tiers:
                score_masked(args.dataset, args.run, what, tier, budgets, faces,
                             M.results_path(args.run, what, args.dataset, tier), subjects)
        elif what == "wrong":                       # the control is run on the main tier only
            score_masked(args.dataset, args.run, what, C.MAIN_TIER, budgets, faces,
                         M.results_path(args.run, what, args.dataset, C.MAIN_TIER), subjects)
        elif what == "ablation":
            for kind in ("ablation_blind", "ablation_prompted"):
                score_masked(args.dataset, args.run, kind, C.MAIN_TIER, budgets, faces,
                             M.results_path(args.run, kind, args.dataset), subjects)
        elif what == "freeform":
            score_freeform(args.dataset, args.run, [C.FREEFORM_PER_IMAGE], faces,
                           M.results_path(args.run, "freeform", args.dataset), subjects)
        elif what == "budget":
            score_freeform(args.dataset, args.run, list(C.BUDGETS), faces,
                           M.results_path(args.run, "budget", args.dataset), subjects)
        else:
            raise SystemExit(f"unknown --what {what}")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {C.RESULTS}/run{args.run}/")


if __name__ == "__main__":
    main()
