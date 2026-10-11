"""PIAR configuration.

Two blocks. PATHS is what you edit: where the raw datasets and pretrained
weights live, and where outputs go. PROTOCOL is what the paper reports and
should not be changed if you want to reproduce it.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------- PATHS --
# Raw datasets, as downloaded (see README):
#   CelebAMask-HQ release: the folder that contains CelebA-HQ-img/
#   DreamBooth:            the dataset/ folder of github.com/google/dreambooth
#   CustomConcept101:      the folder that contains one sub-folder per concept
RAW = {
    "celebahq":         os.environ.get("PIAR_RAW_CELEBAHQ", "/path/to/CelebAMask-HQ"),
    "dreambooth":       os.environ.get("PIAR_RAW_DREAMBOOTH", "/path/to/dreambooth/dataset"),
    "customconcept101": os.environ.get("PIAR_RAW_CUSTOMCONCEPT101", "/path/to/customconcept101"),
}
# Pretrained weights
WEIGHTS = {
    "sd14":     os.environ.get("PIAR_SD14", "/path/to/stable-diffusion-v1-4"),
    "brushnet": os.environ.get("PIAR_BRUSHNET", "/path/to/random_mask_brushnet_ckpt"),
    "sscd":     os.environ.get("PIAR_SSCD", "/path/to/sscd_disc_large.torchscript.pt"),
}
# Study tree written by prepare_data.py (images + manifests) and everything
# the experiments produce (checkpoints, generations). Score rows go to RESULTS.
DATA    = os.environ.get("PIAR_DATA", f"{ROOT}/data")
OUTPUTS = os.environ.get("PIAR_OUTPUTS", f"{ROOT}/outputs")
RESULTS = f"{ROOT}/results"
MASKS   = f"{ROOT}/data/masks"                # data/masks/<tier>/*.png, shipped

# ------------------------------------------------------------------ PROTOCOL --
DATASETS = ("celebahq", "dreambooth", "customconcept101")
MASK_TIERS = ("partial", "small", "large", "xlarge")
MAIN_TIER = "small"

# guidance weights (Eq. 2): w=0 Base Inpainting, w=1 Personalized Inpainting
W_BASE, W_PERSONALIZED, W_PIAR = 0.0, 1.0, 3.0
ARMS = {W_BASE: "base", W_PERSONALIZED: "personalized", W_PIAR: "piar"}

# free-form baselines
CFG_SCALE    = 3.0      # within-model classifier-free guidance
FINEXTRACT_W = 3.0      # cross-model guidance, FineXtract's published values
FINEXTRACT_K = -0.02
FREEFORM_PER_IMAGE = 50 # FineXtract's budget: 50 * N0 generations per checkpoint
CLIQUE_THRESHOLD = 0.5  # fixed SSCD threshold for clique extraction

# sampling
DDIM_STEPS = 50
N_MASKED = 20           # candidates per masked input (main comparisons)
N_MASKED_SWEEP = 50     # candidates generated in run 1 for the budget sweep
BUDGETS = (1, 2, 5, 10, 20, 30, 40, 50)   # N for masked arms, x per photo for free-form

# training (Appendix A.2)
STEPS_PER_IMAGE = 200
LEARNING_RATE = "2e-6"

# evaluation
A_ESR_THRESHOLDS = (0.7, 0.8)

# ablation (Appendix C): six CelebA-HQ subjects drawn with this seed
ABLATION_SEED = 20260924
ABLATION_N_SUBJECTS = 6
ABLATION_W = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def train_seed(run: int) -> int:
    """Training seed of a run. Run 0 is the original run; runs 1-5 are the
    independent replicates behind Tables 1 and 2."""
    return 42 + 1000 * run


def seed_for(subject: str, mask: str, run: int = 0) -> int:
    """Sampling seed for one (subject, mask) in one run. SHA-256 of the key, so
    it is stable across processes; the guidance weight is not part of the key,
    so all three masked arms and both checkpoint-specificity conditions start
    from identical noise. Candidate k uses seed_for(...) + k."""
    key = f"{subject}|{mask}" if not run else f"{subject}|{mask}|r{run}"
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


def freeform_seed(checkpoint: str, subject: str, run: int = 0) -> int:
    """Sampling seed for free-form extraction; the method is not part of the
    key, so CFG and FineXtract share initial noise."""
    return seed_for(f"{checkpoint}|{subject}", "freeform", run)


def mask_for(subject_index: int, photo_index: int, n_photos: int, n_masks: int = 10) -> int:
    """Fixed rotation that pairs each fine-tuning photograph with one mask
    (Appendix A.3): all ten masks are used about equally, and the same index is
    used in every tier. subject_index is 1-based, photo_index is the position in
    the subject's sorted list of fine-tuning images."""
    return (subject_index * n_photos + photo_index) % n_masks
