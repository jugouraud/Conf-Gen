import json
import random
from pathlib import Path
from typing import Iterable, Literal


DEFAULT_IMAGE_SIZE = 256
DEFAULT_CONTACT_SHEET_COLUMNS = 4
DEFAULT_CONTACT_SHEET_ITEMS = 16
CONTACT_SHEET_CAPTION_HEIGHT = 58
CONTACT_SHEET_PADDING = 10
CONTACT_SHEET_CAPTION_LENGTH = 110


def _load_coco_annotations(annotation_file: str | Path) -> tuple[dict[int, dict], list[dict]]:
    """Return COCO image metadata and captions, validating the expected schema."""
    annotation_path = Path(annotation_file)
    with annotation_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        raise ValueError("Expected a COCO captions JSON file with 'images' and 'annotations' fields.")

    images = {image["id"]: image for image in payload["images"]}
    captions = payload["annotations"]
    if not isinstance(captions, list):
        raise ValueError("COCO 'annotations' must be a list.")
    return images, captions


def iter_coco_prompt_image_pairs(
    images_dir: str | Path,
    annotation_file: str | Path,
    *,
    caption_policy: Literal["all", "first", "random"] = "all",
    seed: int | None = None,
) -> Iterable[dict]:
    """Yield normalized records for valid COCO image-caption pairs.

    ``caption_policy='all'`` yields every caption, while ``'first'`` and
    ``'random'`` select one caption per image.  ``random`` is reproducible when
    ``seed`` is supplied.
    """
    root = Path(images_dir)
    images, captions = _load_coco_annotations(annotation_file)
    grouped: dict[int, list[dict]] = {}
    for caption in captions:
        image_id = caption.get("image_id")
        if image_id in images and isinstance(caption.get("caption"), str):
            grouped.setdefault(image_id, []).append(caption)

    if caption_policy not in {"all", "first", "random"}:
        raise ValueError("caption_policy must be 'all', 'first', or 'random'.")
    chooser = random.Random(seed)

    for image_id in sorted(grouped):
        image = images[image_id]
        filename = image.get("file_name")
        if not isinstance(filename, str):
            continue
        image_path = root / filename
        if not image_path.is_file():
            continue

        selected = grouped[image_id]
        if caption_policy == "first":
            selected = selected[:1]
        elif caption_policy == "random":
            selected = [chooser.choice(selected)]

        for caption in selected:
            prompt = " ".join(caption["caption"].split())
            if prompt:
                yield {
                    "image_path": image_path,
                    "prompt": prompt,
                    "image_id": image_id,
                    "caption_id": caption.get("id"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                }


def create_coco_prompt_image_bundle(
    images_dir: str | Path,
    annotation_file: str | Path,
    output_dir: str | Path,
    *,
    name: str = "coco_prompt_image_pairs",
    caption_policy: Literal["all", "first", "random"] = "all",
    seed: int | None = None,
    image_mode: Literal["reference", "copy", "resize"] = "reference",
    image_size: int = DEFAULT_IMAGE_SIZE,
    limit: int | None = None,
) -> Path:
    """Create a JSONL prompt-image bundle and return its manifest path.

    Image records use ``image`` and ``prompt`` keys. ``reference`` writes paths
    relative to ``images_dir``; ``copy`` copies originals into ``output_dir/images``;
    and ``resize`` exports RGB, center-cropped square images at ``image_size``.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided.")
    if image_mode not in {"reference", "copy", "resize"}:
        raise ValueError("image_mode must be 'reference', 'copy', or 'resize'.")
    if image_size < 1:
        raise ValueError("image_size must be positive.")

    source_root = Path(images_dir).resolve()
    destination, manifest, asset_dir = _prepare_bundle_output(output_dir, name, image_mode)
    copied: dict[Path, Path] = {}
    with manifest.open("w", encoding="utf-8") as stream:
        for record_index, pair in enumerate(
            iter_coco_prompt_image_pairs(
                source_root, annotation_file, caption_policy=caption_policy, seed=seed
            )
        ):
            if limit is not None and record_index >= limit:
                break
            record = _build_bundle_record(
                pair, source_root, destination, asset_dir, image_mode, image_size, copied
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest


def _prepare_bundle_output(output_dir: str | Path, name: str, image_mode: str) -> tuple[Path, Path, Path]:
    """Create bundle directories and return destination, manifest, and asset paths."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    asset_dir = destination / "images"
    if image_mode != "reference":
        asset_dir.mkdir(exist_ok=True)
    return destination, destination / f"{name}.jsonl", asset_dir


def _build_bundle_record(
    pair: dict,
    source_root: Path,
    destination: Path,
    asset_dir: Path,
    image_mode: str,
    image_size: int,
    exported_images: dict[Path, Path],
) -> dict:
    """Return one manifest record, exporting its image when required."""
    source = pair["image_path"].resolve()
    try:
        relative_source = source.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"Image path must be inside images_dir: {source}") from error

    record = {key: value for key, value in pair.items() if key != "image_path"}
    if image_mode == "reference":
        record["image"] = relative_source.as_posix()
        return record

    target = exported_images.get(source)
    if target is None:
        target = asset_dir / relative_source
        if image_mode == "resize":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        _export_image(source, target, mode=image_mode, size=image_size)
        exported_images[source] = target
    record["image"] = target.relative_to(destination).as_posix()
    return record


def _export_image(source: Path, target: Path, *, mode: str, size: int) -> None:
    """Copy or resize a source image, importing Pillow only when required."""
    if mode == "copy":
        import shutil
        shutil.copy2(source, target)
        return

    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Pillow is required for image_mode='resize'. Install it with `pip install pillow`."
        ) from error

    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = max(size / image.width, size / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        resized.crop((left, top, left + size, top + size)).save(target.with_suffix(".png"))


def visualize_prompt_image_bundle(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    images_dir: str | Path | None = None,
    max_items: int = DEFAULT_CONTACT_SHEET_ITEMS,
    columns: int = DEFAULT_CONTACT_SHEET_COLUMNS,
    seed: int | None = 0,
) -> Path:
    """Save a labeled contact sheet from a JSONL bundle and return its path.

    Provide ``images_dir`` when visualizing a ``reference`` bundle.
    """
    if max_items < 1 or columns < 1:
        raise ValueError("max_items and columns must be positive.")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise ImportError("Pillow is required for visualization. Install it with `pip install pillow`.") from error

    manifest = Path(manifest_path)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("The bundle contains no records.")
    chosen = random.Random(seed).sample(records, min(max_items, len(records)))
    tile = DEFAULT_IMAGE_SIZE
    rows = (len(chosen) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            columns * (tile + CONTACT_SHEET_PADDING) + CONTACT_SHEET_PADDING,
            rows * (tile + CONTACT_SHEET_CAPTION_HEIGHT + CONTACT_SHEET_PADDING) + CONTACT_SHEET_PADDING,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, record in enumerate(chosen):
        image_file = Path(record["image"])
        if not image_file.is_absolute():
            image_file = (Path(images_dir) if images_dir is not None else manifest.parent) / image_file
        with Image.open(image_file) as image:
            image = image.convert("RGB")
            image.thumbnail((tile, tile))
            x = CONTACT_SHEET_PADDING + (index % columns) * (tile + CONTACT_SHEET_PADDING)
            y = CONTACT_SHEET_PADDING + (index // columns) * (
                tile + CONTACT_SHEET_CAPTION_HEIGHT + CONTACT_SHEET_PADDING
            )
            canvas.paste(image, (x + (tile - image.width) // 2, y + (tile - image.height) // 2))
        text = record["prompt"][:CONTACT_SHEET_CAPTION_LENGTH]
        draw.multiline_text((x, y + tile + 4), text, fill="black", font=font, spacing=2)

    result = Path(output_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(result)
    return result
