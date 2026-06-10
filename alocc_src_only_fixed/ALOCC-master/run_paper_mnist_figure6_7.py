import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

import mnist_experiment_runner
import export_mnist_triplets
from MNIST import MNIST
from utils import DEVICE, set_random_seed


@dataclass
class Args:
    variant: str
    output_dir: str
    specific: int
    epochs: int
    train_count: int
    test_inlier_count: int
    test_outlier_scale: float
    batch_size: int
    eval_batch_size: int
    noise_std: float
    r_alpha: float
    lr: float
    weight_decay: float
    label_smoothing: float
    stop_recon_threshold: float | None
    stop_min_epoch: int
    d_outclass_loss_scale: float
    out_per_class_count: int
    test_outlier_labels: list[int] | None
    selection_strategy: str
    selection_epoch_start: int | None
    selection_epoch_end: int | None
    selection_min_acc: float | None
    selection_min_auc: float | None
    selection_min_auc_hard: bool = False
    selection_log_fallback: bool = True
    bottleneck_rank: int = 0  # [S1-BOT]
    bottleneck_dropout: float = 0.0
    bottleneck_noise_type: str = "dropout"
    stable_window: int = 3
    stable_lambda: float = 0.5
    seed: int | None = None  # [A4-SEED]
    dataset: str = "mnist"  # [DS-FM]


def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int,
                    bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = "dropout",
                    dataset: str = "mnist"):  # [S1-BOT] [DS-FM]
    model = export_mnist_triplets.build_model(variant,
                                              bottleneck_rank=bottleneck_rank,
                                              bottleneck_dropout=bottleneck_dropout,
                                              bottleneck_noise_type=bottleneck_noise_type)
    model._load_checkpoint(checkpoint_path)

    # [DS-FM] dispatch test dataset by name
    if dataset == 'fashion':
        from FashionMNIST import FashionMNIST as _DSCls
    else:
        _DSCls = MNIST
    test_dataset = _DSCls(
        train=False,
        specific=specific,
        count=sample_count,
        out_class_scale=1.0,
        noise_std=noise_std,
    )
    if outlier_labels:
        test_dataset = mnist_experiment_runner.filter_test_dataset(
            test_dataset=test_dataset,
            inner_class=specific,
            allowed_outlier_labels=outlier_labels,
        )
    imgs, noisy_imgs, labels = test_dataset.tensors

    with torch.inference_mode():
        refined = model.G(noisy_imgs)
        raw_scores = torch.sigmoid(model.D(imgs)).squeeze(1).detach().cpu().tolist()
        refined_scores = torch.sigmoid(model.D(refined)).squeeze(1).detach().cpu().tolist()

    payload = []
    for i in range(len(labels)):
        payload.append(
            {
                "index": int(i),
                "label": int(labels[i]),
                "is_inlier": bool(int(labels[i]) == int(specific)),
                "raw_score": float(raw_scores[i]),
                "refined_score": float(refined_scores[i]),
            }
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned MNIST pipeline (Figure 6/7 style).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["alocc", "alocc_tiny", "alocc_loss", "alocc_loss_cls"], default="alocc_loss")
    parser.add_argument("--specific", type=int, default=1)
    parser.add_argument("--outlier-labels", type=int, nargs="*", default=[6, 7])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--test-inlier-count", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.31)
    parser.add_argument("--r-alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--stop-recon-threshold", type=float, default=None)
    parser.add_argument("--stop-min-epoch", type=int, default=1)
    parser.add_argument("--d-outclass-loss-scale", type=float, default=0.1)
    parser.add_argument("--out-per-class-count", type=int, default=128)
    parser.add_argument("--selection-strategy", choices=["stable_refined_auc"], default="stable_refined_auc")
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument("--stable-lambda", type=float, default=0.5)
    parser.add_argument("--selection-epoch-start", type=int, default=2)
    parser.add_argument("--selection-epoch-end", type=int, default=6)
    parser.add_argument("--selection-min-auc", type=float, default=0.60,
                        help="[RM-3c] threshold for paper selection; default lowered 0.95->0.60 (2026-04-19)")
    parser.add_argument("--selection-min-auc-hard", action="store_true", default=False,
                        help="[PR-A] Hard-fail instead of silent fallback.")
    parser.add_argument("--selection-log-fallback", dest="selection_log_fallback", action="store_true", default=True,
                        help="[PR-A] Warn on fallback (default: on).")
    parser.add_argument("--no-selection-log-fallback", dest="selection_log_fallback", action="store_false",
                        help="[PR-A] Disable fallback warning.")
    parser.add_argument("--triplet-count", type=int, default=12)
    parser.add_argument("--figure7-sample-count", type=int, default=40)
    parser.add_argument("--bottleneck-rank", type=int, default=0, help="[S1-BOT]")
    parser.add_argument("--bottleneck-dropout", type=float, default=0.0, help="[S1-BOT]")
    parser.add_argument("--bottleneck-noise-type", choices=["dropout", "gaussian"], default="dropout", help="[S1-BOT]")
    parser.add_argument("--seed", type=int, default=None, help="[A4-SEED] RNG seed; None=42 (historical default)")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist", help="[DS-FM] dataset source")
    args = parser.parse_args()

    set_random_seed(args.seed)  # [A4-SEED]
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_args = Args(
        variant=args.variant,
        output_dir=str(out_dir / "experiment"),
        specific=args.specific,
        epochs=args.epochs,
        train_count=args.train_count,
        test_inlier_count=args.test_inlier_count,
        test_outlier_scale=1.0,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        noise_std=args.noise_std,
        r_alpha=args.r_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        stop_recon_threshold=args.stop_recon_threshold,
        stop_min_epoch=args.stop_min_epoch,
        d_outclass_loss_scale=args.d_outclass_loss_scale,
        out_per_class_count=args.out_per_class_count,
        test_outlier_labels=args.outlier_labels,
        selection_strategy=args.selection_strategy,
        selection_epoch_start=args.selection_epoch_start,
        selection_epoch_end=args.selection_epoch_end,
        selection_min_acc=None,
        selection_min_auc=args.selection_min_auc,
        selection_min_auc_hard=args.selection_min_auc_hard,
        selection_log_fallback=args.selection_log_fallback,
        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]
        bottleneck_dropout=args.bottleneck_dropout,
        bottleneck_noise_type=args.bottleneck_noise_type,
        stable_window=int(getattr(args, "stable_window", 3)),
        stable_lambda=float(getattr(args, "stable_lambda", 0.5)),
        seed=args.seed,  # [A4-SEED]
        dataset=args.dataset,  # [DS-FM]
    )

    mnist_experiment_runner.run_experiment(exp_args)

    best_ckpt = out_dir / "experiment" / "best.pth"
    triplet_dir = out_dir / "triplets"
    triplet_dir.mkdir(parents=True, exist_ok=True)

    model = export_mnist_triplets.build_model(args.variant,  # [S1-BOT]
                                              bottleneck_rank=args.bottleneck_rank,
                                              bottleneck_dropout=args.bottleneck_dropout,
                                              bottleneck_noise_type=args.bottleneck_noise_type)
    model._load_checkpoint(str(best_ckpt))
    export_mnist_triplets.export_group(model, args.specific, "normal", args.triplet_count, args.noise_std, triplet_dir)
    export_mnist_triplets.export_group(
        model,
        args.specific,
        "abnormal",
        args.triplet_count,
        args.noise_std,
        triplet_dir,
        abnormal_labels=args.outlier_labels,
    )

    figure7 = _figure7_scores(
        checkpoint_path=str(best_ckpt),
        variant=args.variant,
        specific=args.specific,
        outlier_labels=args.outlier_labels,
        noise_std=args.noise_std,
        sample_count=args.figure7_sample_count,
        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]
        bottleneck_dropout=args.bottleneck_dropout,
        bottleneck_noise_type=args.bottleneck_noise_type,
        dataset=args.dataset,  # [DS-FM]
    )
    (out_dir / "figure7_scores.json").write_text(json.dumps(figure7, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "device": DEVICE,
        "experiment_dir": str(out_dir / "experiment"),
        "best_checkpoint": str(best_ckpt),
        "triplets_dir": str(triplet_dir),
        "figure7_scores": str(out_dir / "figure7_scores.json"),
        "specific": args.specific,
        "outlier_labels": args.outlier_labels,
        "lr": args.lr,
        "r_alpha": args.r_alpha,
        "noise_std": args.noise_std,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "stop_recon_threshold": args.stop_recon_threshold,
        "stop_min_epoch": args.stop_min_epoch,
        "selection_strategy": args.selection_strategy,
        "selection_epoch_start": args.selection_epoch_start,
        "selection_epoch_end": args.selection_epoch_end,
        "selection_min_auc": args.selection_min_auc,
        "selection_min_auc_hard": args.selection_min_auc_hard,
        "selection_log_fallback": args.selection_log_fallback,
        "stable_window": int(getattr(args, "stable_window", 3)),
        "stable_lambda": float(getattr(args, "stable_lambda", 0.5)),
        "seed": args.seed,  # [A4-SEED]
        "dataset": args.dataset,  # [DS-FM]
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
