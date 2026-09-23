# REPA + MS-COCO setup

This repository uses the official REPA implementation for text-to-image
flow-matching experiments, pinned to a fixed upstream commit for reproducibility.

## Dataset used by REPA T2I

REPA's released text-to-image loader expects the COCO 2014 split sizes:

- train: 82,783 images
- validation: 40,504 images

The raw image-caption corpus is public. For training, REPA consumes:

1. a 256x256 center-cropped image,
2. Stable-Diffusion VAE moments for that image,
3. a CLIP ViT-L/14 embedding of one of its captions.

The preprocessing script in this repo preserves a manifest mapping every feature
index back to the original COCO image id and raw captions, which is important
for prompt-embedding analysis in the thesis.

## 1. Install REPA

```bash
bash scripts/setup_repa.sh

python -m venv .venv-repa
source .venv-repa/bin/activate
pip install --upgrade pip
pip install -r models/REPA/requirements.txt
```

REPA is pinned to:

```text
67f714503e3892f993844aab088ffc5791c92613
```

The setup script also applies one minimal upstream consistency fix: the pinned
`train_t2i.py` tries to unpack a fourth, unused dataset value, while the
released COCO dataset class returns three values.

## 2. Download the exact raw training corpus

```bash
bash scripts/download_coco2014.sh
```

This downloads the official COCO 2014 train images, validation images and
caption annotations into:

```text
data/coco/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

The `data/` directory is intentionally ignored by Git.

## 3. Precompute the features REPA expects

Activate the REPA environment and run:

```bash
source .venv-repa/bin/activate

python scripts/preprocess_repa_coco.py --split train
python scripts/preprocess_repa_coco.py --split val
```

The script uses the same conditioning family described by the upstream
protocol:

- text encoder: `openai/clip-vit-large-patch14`
- VAE: `stabilityai/sd-vae-ft-mse`

It writes:

```text
data/coco256_features/
├── train/
│   ├── 0.png
│   ├── 0.npy
│   ├── 0_0.npy
│   ├── 0_1.npy
│   └── ...
├── val/
├── empty_context.npy
├── train_manifest.jsonl
└── val_manifest.jsonl
```

For image index `i`:

- `i.png`: the 256x256 image consumed by REPA's representation-alignment branch
- `i.npy`: 8-channel VAE posterior moments
- `i_k.npy`: 77x768 CLIP hidden-state sequence for caption `k`

The manifest retains the original prompt strings, image id and filename.

### Smoke-test preprocessing first

Before processing all COCO images:

```bash
python scripts/preprocess_repa_coco.py --split train --limit 16 \
  --output-dir data/coco256_features_smoke
```

Then inspect the files before launching the full preprocessing run.

## 4. Train REPA MMDiT

```bash
source .venv-repa/bin/activate
bash scripts/train_repa_t2i.sh
```

Paths can be overridden without editing the launcher:

```bash
REPA_DATA_DIR=/scratch/coco256_features \
REPA_OUTPUT_DIR=/scratch/repa-exps \
bash scripts/train_repa_t2i.sh
```

## 5. Thesis representation analysis

The canonical raw training prompts remain available in:

```text
data/coco/annotations/captions_train2014.json
```

and the exact feature-index mapping is retained in:

```text
data/coco256_features/train_manifest.jsonl
```

This lets us analyze the full path:

```text
raw caption
  -> CLIP token embeddings / hidden states
  -> MMDiT text stream and joint hidden states
  -> conditional vector field
  -> latent trajectory
  -> generated image
```

That is preferable to keeping only precomputed `*_k.npy` files, because those
files no longer contain the original caption text.

## Reproducibility note

REPA's own README warns that the released code may contain errors introduced
during code cleanup. The integration here pins upstream and fixes the two
inconsistencies relevant to the COCO T2I path:

1. REPA's training-loop tuple-unpacking mismatch.
2. The upstream U-ViT preprocessing instructions do not save the PNG files
   required by REPA's later COCO loader.

For that reason this repo uses its own small, explicit preprocessing script
instead of depending on the U-ViT extraction script.
