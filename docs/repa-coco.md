# REPA + MS-COCO setup

This repository uses the official REPA implementation for the text-to-image
flow-matching experiments.

## Why COCO 2014

REPA's text-to-image loader expects the COCO 2014 split sizes:

- train: 82,783 images
- validation: 40,504 images

The raw image-caption pairs are fully public. The model does not consume the
JPEGs directly during training: REPA follows the U-ViT preprocessing protocol
and trains from Stable-Diffusion VAE moments plus CLIP caption embeddings.

## 1. Install REPA

```bash
bash scripts/setup_repa.sh

python -m venv .venv-repa
source .venv-repa/bin/activate
pip install --upgrade pip
pip install -r models/REPA/requirements.txt
```

REPA is pinned to commit:

```text
67f714503e3892f993844aab088ffc5791c92613
```

Pinning makes thesis experiments reproducible even if upstream changes.

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

## 3. Prepare features expected by REPA

REPA's README explicitly directs T2I users to the U-ViT preprocessing
protocol. Set up that preprocessing checkout and dataset link:

```bash
bash scripts/setup_uvit_preprocessing.sh
```

U-ViT requires its converted Stable-Diffusion VAE checkpoint at:

```text
models/U-ViT/assets/stable-diffusion/autoencoder_kl.pth
```

The upstream U-ViT README provides this converted autoencoder through its
"Preparation Before Training and Evaluation" section.

After placing the checkpoint:

```bash
cd models/U-ViT
python scripts/extract_mscoco_feature.py --split train
python scripts/extract_mscoco_feature.py --split val
python scripts/extract_empty_feature.py
cd ../..
```

The resulting directory should be:

```text
models/U-ViT/assets/datasets/coco256_features/
├── train/
│   ├── 0.npy
│   ├── 0_0.npy
│   ├── 0_1.npy
│   └── ...
├── val/
└── empty_context.npy
```

For each image index `i`, U-ViT stores:

- `i.npy`: Stable-Diffusion VAE moments for the image
- `i_k.npy`: CLIP embedding for caption `k`
- the preprocessing script also preserves the resized PNG used by REPA's
  representation-alignment branch

## 4. Train REPA MMDiT

```bash
source .venv-repa/bin/activate
bash scripts/train_repa_t2i.sh
```

You can override paths without editing the script:

```bash
REPA_DATA_DIR=/scratch/coco256_features \
REPA_OUTPUT_DIR=/scratch/repa-exps \
bash scripts/train_repa_t2i.sh
```

## 5. Thesis representation analysis

Keep the original raw captions as the canonical prompt corpus:

```text
data/coco/annotations/captions_train2014.json
```

This lets us later map each training example through:

```text
raw caption
  -> CLIP text embedding
  -> MMDiT text tokens / hidden states
  -> conditional vector field
  -> generated image latent
```

Do not discard the raw JSON annotations after preprocessing; the precomputed
`*_k.npy` files alone do not retain the original caption text.

## Known upstream caveat

At the pinned upstream revision, REPA's T2I code should be sanity-checked
before a long training run. The project README itself notes that released code
may contain preparation/cleaning errors. Start with a small dataloader +
single-step smoke test before launching the full 400k-step experiment.
