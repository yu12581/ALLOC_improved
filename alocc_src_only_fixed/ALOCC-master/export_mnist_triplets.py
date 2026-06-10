import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from MNIST import MNIST
from model import ALOCC, ALOCC_LOSS, ALOCC_LOSS_CLS
from utils import DEVICE, set_random_seed


CELL_W = 28
CELL_H = 28
PADDING = 8
TEXT_H = 18
HEADER_H = 22
FONT_FILL = 255
BG_FILL = 0


def build_model(variant: str, bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = "dropout"):  # [S1-BOT]
    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type)
    if variant == "alocc":
        return ALOCC(in_h=28, out_h=28, **_bk)
    if variant == "alocc_tiny":
        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, **_bk)
    if variant == "alocc_loss":
        return ALOCC_LOSS(in_h=28, out_h=28, **_bk)
    if variant == "alocc_loss_cls":
        return ALOCC_LOSS_CLS(in_h=28, out_h=28, classify=True, **_bk)
    raise ValueError(f"Unknown variant: {variant}")


def tensor_to_uint8(img: torch.Tensor) -> Image.Image:
    arr = ((img.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).squeeze(0).numpy()
    return Image.fromarray(arr, mode="L")


def build_subset(specific: int, label_mode: str, sample_count: int, noise_std: float, abnormal_labels=None):
    dataset = MNIST.test_dataset
    labels = dataset.targets
    if label_mode == "normal":
        candidates = torch.where(labels == specific)[0]
    else:
        if abnormal_labels:
            allowed = torch.zeros_like(labels, dtype=torch.bool)
            for label in abnormal_labels:
                if label == specific:
                    continue
                allowed |= (labels == label)
            candidates = torch.where(allowed)[0]
        else:
            candidates = torch.where(labels != specific)[0]

    if len(candidates) < sample_count:
        raise ValueError(f"Not enough samples: requested={sample_count}, available={len(candidates)}")
    perm = torch.randperm(len(candidates))[:sample_count]
    indices = candidates[perm]

    imgs = dataset.data[indices].float()
    labels = dataset.targets[indices].to(dtype=torch.int64)
    imgs = (imgs / 127.5 - 1.0).unsqueeze(1).to(DEVICE, non_blocking=True)
    noisy_imgs = torch.clamp(imgs + torch.randn_like(imgs, device=DEVICE) * noise_std, -1.0, 1.0)
    return imgs, noisy_imgs, labels


def render_triplets(original, noisy, generated, labels, raw_scores, refined_scores, out_path: Path, title: str):
    sample_count = original.size(0)
    width = PADDING + 3 * (CELL_W + PADDING)
    height = HEADER_H + sample_count * (CELL_H + TEXT_H + PADDING) + PADDING
    canvas = Image.new("L", (width, height), color=BG_FILL)
    draw = ImageDraw.Draw(canvas)

    headers = ["original", "noisy", "generated"]
    for col, header in enumerate(headers):
        x = PADDING + col * (CELL_W + PADDING)
        draw.text((x, 4), header, fill=FONT_FILL)

    for row in range(sample_count):
        top = HEADER_H + row * (CELL_H + TEXT_H + PADDING)
        triplet = [
            tensor_to_uint8(original[row]),
            tensor_to_uint8(noisy[row]),
            tensor_to_uint8(generated[row]),
        ]
        for col, img in enumerate(triplet):
            x = PADDING + col * (CELL_W + PADDING)
            canvas.paste(img, (x, top))

        label_text = f"label={int(labels[row])} raw={raw_scores[row]:.3f} refined={refined_scores[row]:.3f}"
        draw.text((PADDING, top + CELL_H + 2), label_text, fill=FONT_FILL)

    canvas.save(out_path)


def export_group(model, specific: int, label_mode: str, sample_count: int, noise_std: float, output_dir: Path, abnormal_labels=None):
    real_imgs, noisy_imgs, labels = build_subset(
        specific=specific,
        label_mode=label_mode,
        sample_count=sample_count,
        noise_std=noise_std,
        abnormal_labels=abnormal_labels,
    )
    with torch.inference_mode():
        generated = model.G(noisy_imgs)
        raw_scores = torch.sigmoid(model.D(real_imgs)).squeeze(1).detach().cpu().tolist()
        refined_scores = torch.sigmoid(model.D(generated)).squeeze(1).detach().cpu().tolist()

    image_path = output_dir / f"{label_mode}_triplets.png"
    json_path = output_dir / f"{label_mode}_triplets.json"
    render_triplets(real_imgs, noisy_imgs, generated, labels, raw_scores, refined_scores, image_path, label_mode)

    payload = []
    for idx in range(sample_count):
        payload.append({
            "index": idx,
            "label": int(labels[idx]),
            "raw_score": float(raw_scores[idx]),
            "refined_score": float(refined_scores[idx]),
        })
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export MNIST original/noisy/generated triplets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["alocc", "alocc_tiny", "alocc_loss", "alocc_loss_cls"], default="alocc_loss")
    parser.add_argument("--specific", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.31)
    parser.add_argument("--abnormal-labels", type=int, nargs="*", default=None)
    parser.add_argument("--bottleneck-rank", type=int, default=0, help="[S1-BOT]")
    parser.add_argument("--bottleneck-dropout", type=float, default=0.0, help="[S1-BOT]")
    parser.add_argument("--bottleneck-noise-type", choices=["dropout", "gaussian"], default="dropout", help="[S1-BOT]")
    parser.add_argument("--seed", type=int, default=None, help="[A4-SEED] RNG seed; None=42 (historical default)")
    args = parser.parse_args()

    set_random_seed(args.seed)  # [A4-SEED]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant,  # [S1-BOT]
                        bottleneck_rank=int(getattr(args, "bottleneck_rank", 0) or 0),
                        bottleneck_dropout=float(getattr(args, "bottleneck_dropout", 0.0) or 0.0),
                        bottleneck_noise_type=str(getattr(args, "bottleneck_noise_type", "dropout")))
    model._load_checkpoint(args.checkpoint)

    export_group(model, args.specific, "normal", args.sample_count, args.noise_std, output_dir)
    export_group(
        model,
        args.specific,
        "abnormal",
        args.sample_count,
        args.noise_std,
        output_dir,
        abnormal_labels=args.abnormal_labels,
    )

    summary = {
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "variant": args.variant,
        "specific": args.specific,
        "sample_count": args.sample_count,
        "abnormal_labels": args.abnormal_labels,
        "device": DEVICE,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
