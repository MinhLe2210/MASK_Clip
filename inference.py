import argparse
import json
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model.MaskCLIP import MaskCLIP, main_keys, resolve_model_setting_name


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
LABEL_NAMES = ["real", "fake"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pixel=false image classification inference for MaskCLIP."
    )
    parser.add_argument("--image_path", type=str, help="Path to one image.")
    parser.add_argument("--image_dir", type=str, help="Path to a folder of images.")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to a trained MaskCLIP checkpoint.",
    )
    parser.add_argument(
        "--model_setting_name",
        default="ViTL",
        type=str,
        help=(
            "MaskCLIP model setting name. "
            f"Valid options: {', '.join(sorted(main_keys))}. "
            "Case-insensitive aliases are accepted."
        ),
    )
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--if_padding",
        action="store_true",
        help="Pad image to image_size instead of resizing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading checkpoint weights.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Save inference results to this JSON file.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --image_dir, also scan subfolders.",
    )
    return parser.parse_args()


def build_transform(image_size, if_padding=False):
    if if_padding:
        return albu.Compose(
            [
                albu.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=0,
                    value=0,
                    position="top_left",
                ),
                albu.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                albu.Crop(0, 0, image_size, image_size),
                ToTensorV2(),
            ]
        )

    return albu.Compose(
        [
            albu.Resize(image_size, image_size),
            albu.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            albu.Crop(0, 0, image_size, image_size),
            ToTensorV2(),
        ]
    )


def load_image_tensor(image_path, image_size, if_padding, device):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transform = build_transform(image_size=image_size, if_padding=if_padding)
    tensor = transform(image=image_np)["image"].unsqueeze(0)
    return tensor.to(device)


def get_image_paths(image_dir, recursive=False):
    pattern = "**/*" if recursive else "*"
    image_paths = [
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid state dict.")

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state_dict[key] = value
    return state_dict


def load_model(checkpoint_path, model_setting_name, device, strict=False):
    model_setting_name = resolve_model_setting_name(model_setting_name)
    model = MaskCLIP(model_setting_name=model_setting_name)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model, missing_keys, unexpected_keys


@torch.inference_mode()
def classify_image(model, image):
    clip_image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=True)
    model.clip.encode_image(clip_image)

    features = torch.stack([hook.output for hook in model.hooks], dim=2)
    selected_features = [features[:, :, i, :] for i in model.selected_layers]
    selected_features = torch.stack(selected_features, dim=2)
    cls_features = selected_features[0, :, :, :]

    text = ["an image"] * 2
    prompts, tokenized_prompts = model.prompt_learner(model.clip, text, image.device)
    text_features = model.encode_text(prompts, tokenized_prompts)
    text_features = torch.chunk(text_features, dim=0, chunks=2)
    text_features_mean = torch.stack(
        [text_features[0].mean(0), text_features[1].mean(0)], dim=0
    )
    text_features_mean = text_features_mean / text_features_mean.norm(
        dim=-1, keepdim=True
    )

    cls_features = model.cls_aggregator(cls_features)
    logits = cls_features @ text_features_mean.t()
    probabilities = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probabilities, dim=1).item())

    return {
        "label": LABEL_NAMES[pred_idx],
        "label_id": pred_idx,
        "probabilities": {
            LABEL_NAMES[i]: float(probabilities[0, i].detach().cpu())
            for i in range(len(LABEL_NAMES))
        },
        "logits": {
            LABEL_NAMES[i]: float(logits[0, i].detach().cpu())
            for i in range(len(LABEL_NAMES))
        },
    }


def infer_image_file(model, image_path, image_size, if_padding, device):
    image = load_image_tensor(
        image_path=image_path,
        image_size=image_size,
        if_padding=if_padding,
        device=device,
    )
    result = classify_image(model, image)
    label = result["label"]
    return {
        "file_name": image_path.name,
        "image_path": str(image_path),
        "result": label,
        "class": label,
        "prob": result["probabilities"][label],
        "probabilities": result["probabilities"],
    }


def save_results_json(results, output_json):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)

    if not args.image_path and not args.image_dir:
        raise ValueError("Please provide --image_path or --image_dir.")
    if args.image_path and args.image_dir:
        raise ValueError("Please provide only one of --image_path or --image_dir.")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_paths = []
    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_paths = [image_path]

    if args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Image folder not found: {image_dir}")
        image_paths = get_image_paths(image_dir=image_dir, recursive=args.recursive)
        if not image_paths:
            raise FileNotFoundError(f"No images found in folder: {image_dir}")

    device = torch.device(args.device)
    model, missing_keys, unexpected_keys = load_model(
        checkpoint_path=checkpoint_path,
        model_setting_name=args.model_setting_name,
        device=device,
        strict=args.strict,
    )

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        result = infer_image_file(
            model=model,
            image_path=image_path,
            image_size=args.image_size,
            if_padding=args.if_padding,
            device=device,
        )
        results.append(result)
        if args.image_dir and not args.json:
            print(
                f"[{index}/{len(image_paths)}] {result['file_name']}: "
                f"{result['class']} ({result['prob']:.6f})"
            )

    output_json = Path(args.output_json) if args.output_json else None
    if args.image_dir and output_json is None:
        output_json = Path("inference_results.json")

    if output_json is not None:
        save_results_json(results, output_json)

    if args.json:
        payload = results[0] if args.image_path and len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.image_path:
        result = results[0]
        print(f"image: {result['image_path']}")
        print(f"prediction: {result['class']} (prob={result['prob']:.6f})")
        print(
            "probabilities: "
            + ", ".join(
                f"{name}={score:.6f}" for name, score in result["probabilities"].items()
            )
        )

    if output_json is not None:
        print(f"saved json: {output_json}")

    if missing_keys:
        print(f"warning: missing checkpoint keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"warning: unexpected checkpoint keys: {len(unexpected_keys)}")


if __name__ == "__main__":
    main()
