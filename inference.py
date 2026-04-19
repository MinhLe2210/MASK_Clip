import argparse
import json
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model.MaskCLIP import MaskCLIP


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
LABEL_NAMES = ["real", "fake"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pixel=false image classification inference for MaskCLIP."
    )
    parser.add_argument("--image_path", required=True, type=str, help="Path to one image.")
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
        help="MaskCLIP model setting name. Default matches train.sh.",
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


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    checkpoint_path = Path(args.checkpoint_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    model, missing_keys, unexpected_keys = load_model(
        checkpoint_path=checkpoint_path,
        model_setting_name=args.model_setting_name,
        device=device,
        strict=args.strict,
    )
    image = load_image_tensor(
        image_path=image_path,
        image_size=args.image_size,
        if_padding=args.if_padding,
        device=device,
    )
    result = classify_image(model, image)
    result["image_path"] = str(image_path)
    result["checkpoint_path"] = str(checkpoint_path)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"image: {result['image_path']}")
    print(f"prediction: {result['label']} (label_id={result['label_id']})")
    print(
        "probabilities: "
        + ", ".join(
            f"{name}={score:.6f}" for name, score in result["probabilities"].items()
        )
    )

    if missing_keys:
        print(f"warning: missing checkpoint keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"warning: unexpected checkpoint keys: {len(unexpected_keys)}")


if __name__ == "__main__":
    main()
