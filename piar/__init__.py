"""PIAR: Partial-Image-Assisted Reconstruction.

pipeline   load Stable Diffusion + BrushNet, prepare a masked input
guidance   reconstruct a masked photograph with contrastive guidance (Eq. 2)
freeform   the CFG and FineXtract free-form baselines
select     SSCD medoid; clique extraction for free-form generations
metrics    AS, A-ESR, CLIP, LPIPS, SSIM, ArcFace
manifest   which subject trains on which images, and where files live
aggregate  per-subject, per-run and across-run statistics
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:          # so `import config` works from anywhere
    sys.path.insert(0, _ROOT)
