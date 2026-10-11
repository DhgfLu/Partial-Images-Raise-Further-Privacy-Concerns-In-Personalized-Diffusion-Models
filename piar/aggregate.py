"""Aggregation (Appendix A.5).

Score rows are one per evaluated unit: a (photograph, mask) pair for the
masked arms, a fine-tuning photograph for the free-form baselines. They are
combined in this order:

    1. within a subject, average over its units (A-ESR thresholds each unit
       first; ArcFace averages over units where a face was detected);
    2. within a run, average over subjects              -> dataset-level score;
    3. across runs, mean and sample SD                  -> Tables 1, 2, 4.

Appendix tables that report variation across subjects use step 1 of a single
run and take the mean and sample SD across subjects.
"""
import json
from collections import defaultdict
from statistics import mean, stdev

METRICS = ("AS", "A-ESR@0.7", "A-ESR@0.8", "CLIP", "LPIPS", "SSIM", "ArcFace")


def load_rows(*paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def subject_scores(rows, keys):
    """Step 1. Groups rows by `keys` plus the subject and averages the units.
    Returns {group_key: {subject: {metric: value}}}."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[tuple(r[k] for k in keys)][r["subject"]].append(r)
    out = {}
    for g, by_subject in groups.items():
        out[g] = {}
        for s, units in by_subject.items():
            sscd = [u["sscd"] for u in units]
            faces = [u["arcface"] for u in units if u.get("arcface") is not None]
            out[g][s] = {
                "AS": mean(sscd),
                "A-ESR@0.7": mean(x > 0.7 for x in sscd),
                "A-ESR@0.8": mean(x > 0.8 for x in sscd),
                "CLIP": mean(u["clip"] for u in units),
                "LPIPS": mean(u["lpips"] for u in units),
                "SSIM": mean(u["ssim"] for u in units),
                "ArcFace": mean(faces) if faces else None,
                "n_units": len(units),
            }
    return out


def mean_sd(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None, 0
    return mean(v), (stdev(v) if len(v) > 1 else None), len(v)


def across_subjects(rows, keys):
    """Steps 1-2 for one run, keeping the spread: {group: {metric: (mean, sd, n)}}."""
    out = {}
    for g, by_subject in subject_scores(rows, keys).items():
        out[g] = {m: mean_sd(v[m] for v in by_subject.values()) for m in METRICS}
    return out


def dataset_scores(rows, keys):
    """Steps 1-2 for one run, means only: {group: {metric: value}}."""
    return {g: {m: v[m][0] for m in METRICS} for g, v in across_subjects(rows, keys).items()}


def across_runs(per_run):
    """Step 3. per_run is a list of dataset_scores() results, one per run.
    Returns {group: {metric: (mean, sd, n_runs)}}."""
    groups = set().union(*[set(d) for d in per_run])
    out = {}
    for g in groups:
        out[g] = {m: mean_sd(d[g][m] for d in per_run if g in d) for m in METRICS}
    return out


def fmt(stat, scale=100.0, digits=1):
    """'88.5±3.5' from (mean, sd, n); '–' when undefined."""
    if stat is None or stat[0] is None:
        return "–"
    m, sd = stat[0] * scale, stat[1]
    return f"{m:.{digits}f}" if sd is None else f"{m:.{digits}f}±{sd * scale:.{digits}f}"
