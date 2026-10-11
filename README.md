# PIAR: Partial-Image-Assisted Reconstruction

Code for *Partial Images Raise Further Privacy Concerns in Personalized Diffusion Models*.

## Setup

1. `conda env create -f environment.yml && conda activate piar`
2. Download the datasets and weights below; each line ends with the folder or file that `config.py` points at. The official CelebAMask-HQ and BrushNet releases are on Google Drive, so the Hugging Face copies of the same files are linked instead.
   - [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ): unzip `CelebAMask-HQ.zip` from https://huggingface.co/datasets/liusq/CelebAMask-HQ → `CelebAMask-HQ/` (only `CelebA-HQ-img/` inside is used)
   - [DreamBooth](https://github.com/google/dreambooth): `git clone https://github.com/google/dreambooth` → `dreambooth/dataset/`
   - [CustomConcept101](https://github.com/adobe-research/custom-diffusion/tree/main/customconcept101): unzip https://huggingface.co/datasets/nupurkmr9/custom-diffusion/resolve/main/benchmark_dataset.zip → `benchmark_dataset/`
   - [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4), 5.2 GB: `huggingface-cli download CompVis/stable-diffusion-v1-4 --include "*.json" "*.txt" "*.safetensors" --exclude "*fp16*" "*non_ema*" --local-dir stable-diffusion-v1-4 --local-dir-use-symlinks False` → `stable-diffusion-v1-4/`
   - [BrushNet](https://github.com/TencentARC/BrushNet), the `random_mask_brushnet_ckpt` checkpoint: `huggingface-cli download Sanster/brushnet_random_mask --exclude "*fp16*" --local-dir random_mask_brushnet_ckpt --local-dir-use-symlinks False` → `random_mask_brushnet_ckpt/`
   - [SSCD](https://github.com/facebookresearch/sscd-copy-detection): https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt → that file
3. Set the six paths in `config.py`, then `python prepare_data.py`.

## Main results

```
python experiment.py main             # Tables 1-3 and 5, one run, 34 h
python experiment.py main --runs 1-5  # the paper's mean ± SD over five runs
```

## Other experiments

```
python experiment.py sweep      # mask coverage and generation budget: Tables 6-8, 12-14, Figures 5 and 7, 71 h
python experiment.py heldout    # held-out photographs: Tables 9-11, 50 h
python experiment.py ablation   # guidance weight and prompt: Tables 15-16, 3.5 h
```

## Tables

```
python tables.py
```

Writes every table of the paper and the figure CSVs to `tables/` from whatever runs are in `results/`. Every command above skips subjects already scored and can be resumed; the sweep is run 1 of the main results at every mask tier with 50 candidates, so `main` skips run 1 after it. 
Add `--gpu N` to run on a particular GPU. Times are as measured on one NVIDIA RTX PRO 6000 (96 GB) at the default batch size of 64 (held-out at 16); reduce `--batch-size` on smaller GPUs.

## Storage

Each command removes a subject's checkpoint and generated images once they are scored, so prepare about 10 GB; the score rows in `results/` are what remains. With `--keep` nothing is removed: about 250 GB per run of the main results, 330 GB for the sweep, 150 GB for held-out and 6 GB for the ablation.
