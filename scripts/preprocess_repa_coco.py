#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

CLIP_MODEL = "openai/clip-vit-large-patch14"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    crop = min(w, h)
    left = (w - crop) // 2
    top = (h - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def load_coco_annotations(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    captions = defaultdict(list)
    for ann in data["annotations"]:
        captions[int(ann["image_id"])].append(ann["caption"])

    images = sorted(data["images"], key=lambda item: int(item["id"]))
    return images, captions


@torch.inference_mode()
def encode_caption_batch(texts, tokenizer, text_encoder, device):
    toks = tokenizer(
        texts,
        truncation=True,
        max_length=77,
        padding="max_length",
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    return text_encoder(**toks).last_hidden_state.detach().cpu().numpy()


@torch.inference_mode()
def encode_image(image: Image.Image, vae, device):
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    posterior = vae.encode(x).latent_dist
    if hasattr(posterior, "parameters"):
        moments = posterior.parameters
    else:
        moments = torch.cat([posterior.mean, posterior.logvar], dim=1)
    return moments.squeeze(0).detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COCO 2014 in the exact feature layout expected by REPA T2I."
    )
    parser.add_argument("--coco-dir", type=Path, default=Path("data/coco"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/coco256_features"))
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small subset for a smoke test. Do not use for full training.",
    )
    args = parser.parse_args()

    split_name = f"{args.split}2014"
    ann_name = f"captions_{split_name}.json"
    image_dir = args.coco_dir / split_name
    ann_path = args.coco_dir / "annotations" / ann_name
    split_out = args.output_dir / args.split
    split_out.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing COCO images: {image_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations: {ann_path}")

    images, captions_by_id = load_coco_annotations(ann_path)
    if args.limit is not None:
        images = images[: args.limit]

    device = torch.device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device).eval()
    vae = AutoencoderKL.from_pretrained(VAE_MODEL).to(device).eval()

    manifest_path = args.output_dir / f"{args.split}_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, item in tqdm(enumerate(images), total=len(images), desc=f"COCO {args.split}"):
            image_id = int(item["id"])
            file_name = item["file_name"]
            captions = captions_by_id[image_id]

            image = center_crop_resize(Image.open(image_dir / file_name), args.resolution)
            image.save(split_out / f"{idx}.png")

            moments = encode_image(image, vae, device)
            np.save(split_out / f"{idx}.npy", moments)

            contexts = encode_caption_batch(captions, tokenizer, text_encoder, device)
            for k, context in enumerate(contexts):
                np.save(split_out / f"{idx}_{k}.npy", context)

            manifest.write(
                json.dumps(
                    {
                        "index": idx,
                        "image_id": image_id,
                        "file_name": file_name,
                        "captions": captions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if args.split == "train":
        empty = encode_caption_batch([""], tokenizer, text_encoder, device)[0]
        np.save(args.output_dir / "empty_context.npy", empty)

    print(f"Prepared {len(images)} {args.split} examples under {split_out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
