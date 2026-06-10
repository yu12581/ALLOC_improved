import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager, nullcontext

from Metrics import calculate_metrics
from MNIST import MNIST
# [DS-FM] dataset class dispatch (default mnist = bitwise identical to pre-DS-FM)
def _dataset_cls(args):  # [DS-FM]
    name = (getattr(args, 'dataset', 'mnist') or 'mnist').lower()
    if name == 'mnist':
        return MNIST
    if name == 'fashion':
        from FashionMNIST import FashionMNIST
        return FashionMNIST
    raise ValueError(f"Unknown --dataset: {name}")
from model import ALOCC, ALOCC_LOSS, ALOCC_LOSS_CLS, ALOCC_LOSS_DUAL_D, ALOCC_LOSS_BASELINE_REF, ALOCC_LOSS_CLIP
from utils import DEVICE, set_random_seed


@contextmanager
def _eval_bn_use_batch_stats(module: nn.Module):
    bns: list[tuple[nn.BatchNorm2d, bool, float | None]] = []
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            bns.append((m, bool(m.training), m.momentum))
            m.train(True)
            m.momentum = 0.0
    try:
        yield
    finally:
        for m, training, momentum in bns:
            m.train(training)
            m.momentum = momentum


def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0,  # [S1-BOT]
                bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = "dropout",
                tf_verbatim_rmsprop: bool = False, lr_d: float = None, lr_g: float = None,
                use_cbam: bool = False):  # [CBAM]
    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type,
               tf_verbatim_rmsprop=tf_verbatim_rmsprop, use_cbam=bool(use_cbam))  # [CBAM]
    if variant == "alocc":
        return ALOCC(in_h=28, out_h=28, lr=lr, lr_d=lr_d, lr_g=lr_g, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    if variant == "alocc_tiny":
        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, lr=lr, lr_d=lr_d, lr_g=lr_g, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    if variant == "alocc_loss":
        return ALOCC_LOSS(in_h=28, out_h=28, lr=lr, lr_d=lr_d, lr_g=lr_g, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    if variant == "alocc_loss_cls":
        return ALOCC_LOSS_CLS(in_h=28, out_h=28, lr=lr, lr_d=lr_d, lr_g=lr_g, classify=True, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    if variant == "alocc_loss_dual_d":
        return ALOCC_LOSS_DUAL_D(in_h=28, out_h=28, lr=lr, lr_d=lr_d, lr_g=lr_g, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    if variant == "alocc_loss_baseline_ref":
        return ALOCC_LOSS_BASELINE_REF(in_h=28, out_h=28, lr=lr, lr_d=lr_d, lr_g=lr_g, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    if variant == "alocc_loss_clip":
        return ALOCC_LOSS_CLIP(in_h=28, out_h=28, lr=lr, lr_d=lr_d, lr_g=lr_g, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)
    raise ValueError(f"Unknown variant: {variant}")


def build_data(args):
    _DS = _dataset_cls(args)  # [DS-FM]
    train_dataset = _DS(
        train=True,
        specific=args.specific,
        count=args.train_count,
        noise_std=args.noise_std,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = _DS(  # [DS-FM]
        train=False,
        specific=args.specific,
        count=args.test_inlier_count,
        out_class_scale=args.test_outlier_scale,
        noise_std=args.noise_std,
    )
    if args.test_outlier_labels:
        test_dataset = filter_test_dataset(
            test_dataset=test_dataset,
            inner_class=args.specific,
            allowed_outlier_labels=args.test_outlier_labels,
        )
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    outclass_loader = None
    if args.variant not in ("alocc", "alocc_tiny"):
        outclass_dataset = _DS(  # [DS-FM]
            train=True,
            count=0,
            specific=args.specific,
            per_out_class_count=args.out_per_class_count,
            noise_std=args.noise_std,
        )
        outclass_loader = DataLoader(outclass_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader, outclass_loader


def filter_test_dataset(test_dataset, inner_class: int, allowed_outlier_labels):
    allowed = set(int(label) for label in allowed_outlier_labels)
    allowed.discard(inner_class)
    if not allowed:
        raise ValueError("test_outlier_labels must contain at least one label different from --specific")

    imgs, noisy_imgs, labels = test_dataset.data.tensors
    mask = (labels == inner_class)
    for label in sorted(allowed):
        mask |= (labels == label)
    return TensorDataset(imgs[mask], noisy_imgs[mask], labels[mask])


def compute_score_stats(model, dataloader, inner_class: int):
    in_scores, out_scores = [], []
    with torch.inference_mode():
        for real_imgs, noisy_imgs, labels in dataloader:
            gen_imgs = model.G(noisy_imgs)
            scores = torch.sigmoid(model.D(gen_imgs)).squeeze(1)
            in_mask = labels == inner_class
            out_mask = ~in_mask
            if in_mask.any():
                in_scores.append(scores[in_mask].detach().cpu())
            if out_mask.any():
                out_scores.append(scores[out_mask].detach().cpu())

    def mean_or_nan(values):
        if not values:
            return float("nan")
        return float(torch.cat(values).mean().item())

    return {
        "score_in_mean": mean_or_nan(in_scores),
        "score_out_mean": mean_or_nan(out_scores),
    }


def compute_paper_score_stats(model, dataloader, inner_class: int, eval_bn_batch_stats: bool = False):
    raw_scores, refined_scores, labels_list = [], [], []

    model.G.eval()
    model.D.eval()
    ctx = _eval_bn_use_batch_stats(model) if bool(eval_bn_batch_stats) else nullcontext()
    with torch.inference_mode(), ctx:
        for real_imgs, noisy_imgs, labels in dataloader:
            gen_imgs = model.G(noisy_imgs)
            raw_batch_scores = torch.sigmoid(model.D(real_imgs)).squeeze(1)
            refined_batch_scores = torch.sigmoid(model.D(gen_imgs)).squeeze(1)
            raw_scores.append(raw_batch_scores.detach().cpu())
            refined_scores.append(refined_batch_scores.detach().cpu())
            labels_list.append(labels.detach().cpu())

    raw_scores = torch.cat(raw_scores)
    refined_scores = torch.cat(refined_scores)
    labels = torch.cat(labels_list)
    y_true = (labels == inner_class).numpy()
    raw_np = raw_scores.numpy()
    refined_np = refined_scores.numpy()
    inlier_mask = y_true.astype(bool)
    outlier_mask = ~inlier_mask

    def safe_auc(y_score):
        return float(roc_auc_score(y_true, y_score))

    def safe_eer(y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        return float(brentq(lambda x: 1.0 - x - np.interp(x, fpr, tpr), 0.0, 1.0))

    def best_acc(y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        y_pred = (y_score >= thresholds[:, None])
        tp = (y_true & y_pred).sum(1)
        fp = (~y_true & y_pred).sum(1)
        fn = (y_true & ~y_pred).sum(1)
        f1_scores = 2 * tp / (2 * tp + fp + fn + 1e-8)
        best_idx = np.argmax(f1_scores)
        return float(accuracy_score(y_true, y_score >= thresholds[best_idx]))

    raw_in_mean = float(raw_np[inlier_mask].mean())
    raw_out_mean = float(raw_np[outlier_mask].mean())
    refined_in_mean = float(refined_np[inlier_mask].mean())
    refined_out_mean = float(refined_np[outlier_mask].mean())
    raw_gap = raw_in_mean - raw_out_mean
    refined_gap = refined_in_mean - refined_out_mean

    return {
        "raw_score_in_mean": raw_in_mean,
        "raw_score_out_mean": raw_out_mean,
        "raw_score_gap": raw_gap,
        "raw_auc": safe_auc(raw_np),
        "raw_eer": safe_eer(raw_np),
        "raw_acc": best_acc(raw_np),
        "refined_score_in_mean": refined_in_mean,
        "refined_score_out_mean": refined_out_mean,
        "refined_score_gap": refined_gap,
        "refined_auc": safe_auc(refined_np),
        "refined_eer": safe_eer(refined_np),
        "refined_acc": best_acc(refined_np),
        "score_gap_gain": refined_gap - raw_gap,
        "auc_gain": safe_auc(refined_np) - safe_auc(raw_np),
    }


PAPER_SCORE_ABSOLUTE_ANCHORS = {
    # key: (zero_line, one_line). Direction encoded by order.
    # Anchors approved 2026-04-19 (user): see PROJECT_LOG.md §9.5.
    "acc":            (0.80, 0.95),
    "auc_gain":       (0.00, 0.15),
    "ssim_gap":       (0.00, 0.30),
    "ssim_oc":        (0.50, 0.10),
    "score_gap":      (0.00, 0.20),
    "score_gap_gain": (0.00, 0.10),
    "refined_auc":    (0.50, 0.95),
}


def _normalize_metric_absolute(records, key: str, zero_line: float, one_line: float):
    """Linearly map each record[key] to [0,1] using absolute anchors (RM-3b)."""
    scale = one_line - zero_line
    normalized = {}
    for record in records:
        value = float(record[key])
        if scale == 0:
            normalized[record["epoch"]] = 0.5
            continue
        frac = (value - zero_line) / scale
        frac = max(0.0, min(1.0, frac))
        normalized[record["epoch"]] = frac
    return normalized


def _attach_distortion_score(records, alpha: float = 1.0, beta: float = 1.0):
    """RM-3a distortion score: max(ssim_gap,0)^alpha * max(auc,0)^beta."""
    for record in records:
        gap = max(0.0, float(record["ssim_gap"]))
        auc = max(0.0, float(record["auc"]))
        score = (gap ** float(alpha)) * (auc ** float(beta))
        record["distortion_score"] = float(score)


def _normalize_metric(records, key: str, invert: bool = False):
    values = [float(record[key]) for record in records]
    low, high = min(values), max(values)
    if np.isclose(high, low):
        return {record["epoch"]: 0.5 for record in records}

    normalized = {}
    scale = high - low
    for record in records:
        value = (float(record[key]) - low) / scale
        normalized[record["epoch"]] = 1.0 - value if invert else value
    return normalized


def _attach_paper_score(records, normalization: str = "relative"):
    if not records:
        return

    def _norm(key, invert=False):
        if normalization == "absolute" and key in PAPER_SCORE_ABSOLUTE_ANCHORS:
            z, o = PAPER_SCORE_ABSOLUTE_ANCHORS[key]
            return _normalize_metric_absolute(records, key, z, o)
        return _normalize_metric(records, key, invert=invert)

    normalized = {
        "acc": _norm("acc"),
        "auc": _norm("auc"),
        "eer": _norm("eer", invert=True),
        "raw_auc": _norm("raw_auc"),
        "ssim_ic": _norm("ssim_ic"),
        "ssim_oc": _norm("ssim_oc", invert=True),
        "vif_oc": _norm("vif_oc", invert=True),
        "ssim_gap": _norm("ssim_gap"),
        "vif_gap": _norm("vif_gap"),
        "gmsd_gap": _norm("gmsd_gap"),
        "score_gap": _norm("score_gap"),
        "raw_score_gap": _norm("raw_score_gap"),
        "score_gap_gain": _norm("score_gap_gain"),
        "auc_gain": _norm("auc_gain"),
    }
    weights = {
        "score_gap_gain": 0.18,
        "auc_gain": 0.18,
        "ssim_gap": 0.18,
        "ssim_oc": 0.14,
        "score_gap": 0.10,
        "gmsd_gap": 0.10,
        "vif_gap": 0.08,
        "ssim_ic": 0.08,
        "acc": 0.07,
        "auc": 0.05,
        "eer": 0.02,
        "raw_auc": 0.00,
        "raw_score_gap": 0.00,
        "vif_oc": 0.00,
    }

    for record in records:
        epoch = record["epoch"]
        paper_score = 0.0
        for key, weight in weights.items():
            component = normalized[key][epoch]
            paper_score += weight * component
        record["paper_score"] = float(paper_score)


def _select_records(records, selection_strategy: str, selection_epoch_start, selection_epoch_end, selection_min_acc, selection_min_auc, selection_min_auc_hard: bool = False, selection_log_fallback: bool = True, stable_window: int = 3, stable_lambda: float = 0.5):
    candidates = records
    if selection_epoch_start is not None:
        candidates = [record for record in candidates if record["epoch"] >= selection_epoch_start]
    if selection_epoch_end is not None:
        candidates = [record for record in candidates if record["epoch"] <= selection_epoch_end]
    if not candidates:
        raise ValueError("No checkpoint records remain after applying the epoch selection window")

    eligible = candidates
    if selection_min_acc is not None:
        eligible = [record for record in eligible if record["acc"] >= selection_min_acc]
    if selection_min_auc is not None:
        eligible = [record for record in eligible if record["auc"] >= selection_min_auc]
    fallback_triggered = False
    fallback_reason = None
    if not eligible:
        fallback_triggered = True
        reason_parts = []
        if selection_min_acc is not None:
            reason_parts.append(f"acc>={selection_min_acc}")
        if selection_min_auc is not None:
            reason_parts.append(f"auc>={selection_min_auc}")
        constraint = "+".join(reason_parts) if reason_parts else "(no thresholds set)"
        best_auc = max((r["auc"] for r in candidates), default=float("nan"))
        fallback_reason = (
            f"No candidate epoch in {[r['epoch'] for r in candidates]} satisfied {constraint}; "
            f"best auc in window = {best_auc:.4f}"
        )
        if selection_min_auc_hard:
            raise RuntimeError(f"[PR-A] selection_min_auc_hard=True and fallback would trigger. {fallback_reason}")
        if selection_log_fallback:
            print(f"[PR-A][selection] WARNING fallback to full candidate window: {fallback_reason}", file=sys.stderr)
        eligible = candidates

    if selection_strategy == "stable_refined_auc":
        cand_by_epoch = {int(r["epoch"]): r for r in candidates}
        k = max(1, int(stable_window))
        lam = float(stable_lambda)
        scored = []
        min_cand_epoch = min(cand_by_epoch)
        for r in eligible:
            t = int(r["epoch"])
            start = t - k + 1
            if start < min_cand_epoch:
                continue
            window = []
            ok = True
            for e in range(start, t + 1):
                rr = cand_by_epoch.get(e)
                if rr is None:
                    ok = False
                    break
                window.append(float(rr.get("auc", 0.0)))
            if not ok or len(window) != k:
                continue
            mu = float(np.mean(window))
            sigma = float(np.std(window))
            score = mu - lam * sigma
            scored.append((
                score,
                mu,
                -sigma,
                float(r.get("auc", 0.0)),
                float(r.get("score_gap", 0.0)),
                -float(r.get("eer", 1.0)),
                r,
            ))
        if scored:
            scored.sort(key=lambda x: x[:6], reverse=True)
            best_record = scored[0][-1]
        else:
            if selection_log_fallback:
                print(f"[AUC-STABLE][selection] WARNING stable_refined_auc has no full window (k={k}) within eligible epochs; fallback to max(auc) on eligible.", file=sys.stderr)
            best_record = max(
                eligible,
                key=lambda record: record.get("auc", 0.0),
            )
    elif selection_strategy == "best_auc":
        best_record = max(eligible, key=lambda r: float(r.get("auc", 0.0)))
    elif selection_strategy in ("top5_auc_min_ssimoc", "top3_auc_min_ssimoc"):
        topk = 3 if selection_strategy == "top3_auc_min_ssimoc" else 5
        eligible_sorted = sorted(
            eligible,
            key=lambda r: (
                float(r.get("auc", 0.0)),
                -float(r.get("ssim_oc", float("inf"))),
                -int(r.get("epoch", 0)),
            ),
            reverse=True,
        )
        top = eligible_sorted[:topk]
        if not top:
            best_record = max(eligible, key=lambda record: record.get("auc", 0.0))
        else:
            best_record = min(
                top,
                key=lambda r: (
                    float(r.get("ssim_oc", float("inf"))),
                    -float(r.get("auc", 0.0)),
                    -int(r.get("epoch", 0)),
                ),
            )
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

    selection_info = {
        "strategy": selection_strategy,
        "epoch_start": selection_epoch_start,
        "epoch_end": selection_epoch_end,
        "min_acc": selection_min_acc,
        "min_auc": selection_min_auc,
        "min_auc_hard": bool(selection_min_auc_hard),
        "log_fallback": bool(selection_log_fallback),
        "fallback_triggered": bool(fallback_triggered),
        "fallback_reason": fallback_reason,
        "candidate_epochs": [record["epoch"] for record in candidates],
        "eligible_epochs": [record["epoch"] for record in eligible],
        "stable_window": int(stable_window),
        "stable_lambda": float(stable_lambda),
    }
    if selection_strategy == "top5_auc_min_ssimoc":
        selection_info["topk"] = 5
        selection_info["tie_break"] = "min_ssim_oc_then_max_auc"
    return best_record, selection_info


def evaluate_checkpoints(
    model,
    checkpoint_dir: Path,
    dataloader,
    epochs: int,
    inner_class: int,
    selection_strategy: str,
    selection_epoch_start,
    selection_epoch_end,
    selection_min_acc,
    selection_min_auc,
    selection_min_auc_hard: bool = False,
    selection_log_fallback: bool = True,
    stable_window: int = 3,
    stable_lambda: float = 0.5,
    eval_bn_batch_stats: bool = False,
):
    records = []

    for epoch in range(1, epochs + 1):
        ckpt_path = checkpoint_dir / f"{epoch}.pth"
        model._load_checkpoint(str(ckpt_path))
        metrics = calculate_metrics(model, dataloader, inner_class=inner_class, verbose=False, eval_bn_batch_stats=bool(eval_bn_batch_stats))
        score_stats = compute_paper_score_stats(model, dataloader, inner_class=inner_class, eval_bn_batch_stats=bool(eval_bn_batch_stats))
        record = {
            "epoch": epoch,
            "f1": float(metrics[0]),
            "acc": float(metrics[1]),
            "eer": float(metrics[2]),
            "auc": float(metrics[3]),
            "ssim_ic": float(metrics[4]),
            "ssim_oc": float(metrics[5]),
            "vif_ic": float(metrics[6]),
            "vif_oc": float(metrics[7]),
            "raw_score_gap": score_stats["raw_score_gap"],
            "raw_auc": score_stats["raw_auc"],
            "score_gap": score_stats["refined_score_gap"],
        }
        record["ssim_gap"] = record["ssim_ic"] - record["ssim_oc"]
        record["vif_gap"] = record["vif_ic"] - record["vif_oc"]
        record["gmsd_gap"] = float(metrics[9]) - float(metrics[8])
        record["score_gap_gain"] = score_stats["score_gap_gain"]
        record["auc_gain"] = score_stats["auc_gain"]
        records.append(record)

    best_metrics, selection_info = _select_records(
        records=records,
        selection_strategy=selection_strategy,
        selection_epoch_start=selection_epoch_start,
        selection_epoch_end=selection_epoch_end,
        selection_min_acc=selection_min_acc,
        selection_min_auc=selection_min_auc,
        selection_min_auc_hard=selection_min_auc_hard,
        selection_log_fallback=selection_log_fallback,
        stable_window=stable_window,
        stable_lambda=stable_lambda,
    )
    best_epoch = best_metrics["epoch"]
    shutil.copyfile(checkpoint_dir / f"{best_epoch}.pth", checkpoint_dir / "best.pth")
    return best_epoch, best_metrics, records, selection_info


def run_experiment(args):
    set_random_seed(getattr(args, "seed", None))  # [A4-SEED]
    checkpoint_dir = Path(args.output_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, outclass_loader = build_data(args)
    model = build_model(
        variant=args.variant,
        lr=args.lr,
        lr_d=getattr(args, "lr_d", None),  # [TTUR]
        lr_g=getattr(args, "lr_g", None),  # [TTUR]
        weight_decay=getattr(args, "weight_decay", 0.0) or 0.0,
        label_smoothing=getattr(args, "label_smoothing", 0.0) or 0.0,
        bottleneck_rank=int(getattr(args, "bottleneck_rank", 0) or 0),  # [S1-BOT]
        bottleneck_dropout=float(getattr(args, "bottleneck_dropout", 0.0) or 0.0),
        bottleneck_noise_type=str(getattr(args, "bottleneck_noise_type", "dropout")),
        tf_verbatim_rmsprop=bool(getattr(args, "tf_verbatim_rmsprop", False)),  # [BASELINE-A][D2-B]
        use_cbam=bool(getattr(args, "use_cbam", False)),  # [CBAM]
    )

    if args.variant in ("alocc", "alocc_tiny"):
        trained_epochs = model._train(
            data_loader=train_loader,
            epoch=args.epochs,
            step=1,
            checkpoint_dir=str(checkpoint_dir),
            r_alpha=args.r_alpha,
            stop_recon_threshold=getattr(args, "stop_recon_threshold", None),
            stop_min_epoch=getattr(args, "stop_min_epoch", 1),
        )
    else:
        trained_epochs = model._train(
            data_loader=train_loader,
            outclass_loader=outclass_loader,
            epoch=args.epochs,
            step=1,
            checkpoint_dir=str(checkpoint_dir),
            r_alpha=args.r_alpha,
            d_outclass_loss_scale=args.d_outclass_loss_scale,
            outclass_every=int(getattr(args, "outclass_every", 20) or 20),
            g_steps=int(getattr(args, "g_steps", 2) or 2),
            d_steps=int(getattr(args, "d_steps", 1) or 1),  # [TTUR]
            stop_recon_threshold=getattr(args, "stop_recon_threshold", None),
            stop_min_epoch=getattr(args, "stop_min_epoch", 1),
            agc_ema_decay=float(getattr(args, "agc_ema_decay", 0.99)),   # [AGC]
            agc_k=float(getattr(args, "agc_k", 3.0)),                    # [AGC]
            agc_min_clip=float(getattr(args, "agc_min_clip", 1.0)),      # [AGC]
        )

    best_epoch, best_metrics, records, selection_info = evaluate_checkpoints(
        model=model,
        checkpoint_dir=checkpoint_dir,
        dataloader=test_loader,
        epochs=int(trained_epochs),
        inner_class=args.specific,
        selection_strategy=args.selection_strategy,
        selection_epoch_start=args.selection_epoch_start,
        selection_epoch_end=args.selection_epoch_end,
        selection_min_acc=args.selection_min_acc,
        selection_min_auc=args.selection_min_auc,
        selection_min_auc_hard=bool(getattr(args, "selection_min_auc_hard", False)),
        selection_log_fallback=bool(getattr(args, "selection_log_fallback", True)),
        stable_window=int(getattr(args, "stable_window", 3)),
        stable_lambda=float(getattr(args, "stable_lambda", 0.5)),
        eval_bn_batch_stats=bool(getattr(args, "eval_bn_batch_stats", False)),
    )

    summary = {
        "variant": args.variant,
        "device": DEVICE,
        "specific": args.specific,
        "epochs": args.epochs,
        "trained_epochs": int(trained_epochs),
        "train_count": args.train_count,
        "test_inlier_count": args.test_inlier_count,
        "test_outlier_scale": args.test_outlier_scale,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "noise_std": args.noise_std,
        "r_alpha": args.r_alpha,
        "lr": args.lr,
        "weight_decay": getattr(args, "weight_decay", 0.0) or 0.0,
        "label_smoothing": getattr(args, "label_smoothing", 0.0) or 0.0,
        "stop_recon_threshold": getattr(args, "stop_recon_threshold", None),
        "stop_min_epoch": getattr(args, "stop_min_epoch", 1),
        "d_outclass_loss_scale": args.d_outclass_loss_scale,
        "outclass_every": getattr(args, "outclass_every", 20),
        "g_steps": getattr(args, "g_steps", 2),
        "d_steps": getattr(args, "d_steps", 1),  # [TTUR]
        "lr_d": getattr(args, "lr_d", None),  # [TTUR]
        "lr_g": getattr(args, "lr_g", None),  # [TTUR]
        "agc_ema_decay": float(getattr(args, "agc_ema_decay", 0.99)),    # [AGC]
        "agc_k": float(getattr(args, "agc_k", 3.0)),                     # [AGC]
        "agc_min_clip": float(getattr(args, "agc_min_clip", 1.0)),       # [AGC]
        "out_per_class_count": args.out_per_class_count,
        "test_outlier_labels": args.test_outlier_labels,
        "switches": {
            "selection_min_auc_hard": bool(getattr(args, "selection_min_auc_hard", False)),
            "selection_log_fallback": bool(getattr(args, "selection_log_fallback", True)),
            "selection_fallback_triggered": bool(selection_info.get("fallback_triggered", False)),
            "stop_recon_threshold_active": getattr(args, "stop_recon_threshold", None) is not None,
            "d_outclass_loss_active": float(getattr(args, "d_outclass_loss_scale", 0.0) or 0.0) > 0.0,
            "selection_strategy": str(args.selection_strategy),
            "selection_min_auc": (float(args.selection_min_auc) if args.selection_min_auc is not None else None),
            "bottleneck_rank": int(getattr(args, "bottleneck_rank", 0) or 0),  # [S1-BOT]
            "bottleneck_dropout": float(getattr(args, "bottleneck_dropout", 0.0) or 0.0),
            "bottleneck_noise_type": str(getattr(args, "bottleneck_noise_type", "dropout")),
            "stable_window": int(getattr(args, "stable_window", 3)),
            "stable_lambda": float(getattr(args, "stable_lambda", 0.5)),
            "eval_bn_batch_stats": bool(getattr(args, "eval_bn_batch_stats", False)),
            "seed": (int(args.seed) if getattr(args, "seed", None) is not None else None),  # [A4-SEED]
            "tf_verbatim_rmsprop": bool(getattr(args, "tf_verbatim_rmsprop", False)),  # [BASELINE-A][D2-B]
            "use_cbam": bool(getattr(args, "use_cbam", False)),  # [CBAM]
        },
        "selection_info": selection_info,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "records": records,
    }

    with open(checkpoint_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "variant": args.variant,
        "best_epoch": best_epoch,
        "acc": round(best_metrics["acc"], 4),
        "auc": round(best_metrics["auc"], 4),
        "raw_auc": round(best_metrics["raw_auc"], 4),
        "auc_gain": round(best_metrics["auc_gain"], 4),
        "ssim_ic": round(best_metrics["ssim_ic"], 4),
        "ssim_oc": round(best_metrics["ssim_oc"], 4),
        "ssim_gap": round(best_metrics["ssim_gap"], 4),
        "raw_score_gap": round(best_metrics["raw_score_gap"], 4),
        "score_gap": round(best_metrics["score_gap"], 4),
        "score_gap_gain": round(best_metrics["score_gap_gain"], 4),
        "summary_path": str(checkpoint_dir / "summary.json"),
    }, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Small MNIST experiment runner for ALOCC variants.")
    parser.add_argument("--variant", choices=["alocc", "alocc_tiny", "alocc_loss", "alocc_loss_cls", "alocc_loss_dual_d", "alocc_loss_baseline_ref", "alocc_loss_clip"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--specific", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--test-inlier-count", type=int, default=200)
    parser.add_argument("--test-outlier-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.31)
    parser.add_argument("--r-alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--bottleneck-rank", type=int, default=0, help="[S1-BOT] low-rank 1x1 bottleneck; 0=disabled")
    parser.add_argument("--bottleneck-dropout", type=float, default=0.0, help="[S1-BOT] dropout/gaussian std applied after low-rank; 0=disabled")
    parser.add_argument("--bottleneck-noise-type", choices=["dropout", "gaussian"], default="dropout", help="[S1-BOT] noise family for bottleneck")
    parser.add_argument("--stop-recon-threshold", type=float, default=None)
    parser.add_argument("--stop-min-epoch", type=int, default=1)
    parser.add_argument("--d-outclass-loss-scale", type=float, default=0.1)
    parser.add_argument("--outclass-every", type=int, default=20)
    parser.add_argument("--g-steps", type=int, default=2)
    parser.add_argument("--d-steps", type=int, default=1, help="[TTUR] Discriminator training steps per batch")
    parser.add_argument("--lr-d", type=float, default=None, help="[TTUR] Discriminator learning rate (if None, use --lr)")
    parser.add_argument("--lr-g", type=float, default=None, help="[TTUR] Generator learning rate (if None, use --lr)")
    parser.add_argument("--out-per-class-count", type=int, default=32)
    parser.add_argument("--test-outlier-labels", type=int, nargs="*", default=None)
    parser.add_argument("--selection-strategy", choices=["stable_refined_auc", "best_auc", "top5_auc_min_ssimoc", "top3_auc_min_ssimoc"], default="stable_refined_auc")
    parser.add_argument("--selection-epoch-start", type=int, default=None)
    parser.add_argument("--selection-epoch-end", type=int, default=None)
    parser.add_argument("--selection-min-acc", type=float, default=None)
    parser.add_argument("--selection-min-auc", type=float, default=None)
    parser.add_argument("--selection-min-auc-hard", action="store_true", default=False,
                        help="[PR-A] Raise instead of silently falling back when no epoch meets --selection-min-auc.")
    parser.add_argument("--selection-log-fallback", dest="selection_log_fallback", action="store_true", default=True,
                        help="[PR-A] Log a stderr warning when fallback triggers (default: on).")
    parser.add_argument("--no-selection-log-fallback", dest="selection_log_fallback", action="store_false",
                        help="[PR-A] Disable the fallback warning.")
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument("--stable-lambda", type=float, default=0.5)
    parser.add_argument("--eval-bn-batch-stats", dest="eval_bn_batch_stats", action="store_true", default=False)
    parser.add_argument("--tf-verbatim-rmsprop", dest="tf_verbatim_rmsprop", action="store_true", default=False,
                        help="[BASELINE-A][D2-B] opt-in TF1.15 RMSprop defaults (alpha=0.9, eps=1e-10); default off restores PyTorch defaults.")
    parser.add_argument("--seed", type=int, default=None,
                        help="[A4-SEED] RNG seed override; None=42 (historical default, bitwise identical)")
    parser.add_argument("--agc-ema-decay", type=float, default=0.99,
                        help="[AGC] EMA 衰减系数，建议 0.999 使阈值更平滑（默认 0.99）")
    parser.add_argument("--agc-k", type=float, default=3.0,
                        help="[AGC] 裁剪阈值 = mu + k*sigma，降低可使裁剪更激进（默认 3.0，建议 1.0）")
    parser.add_argument("--agc-min-clip", type=float, default=1.0,
                        help="[AGC] 裁剪阈值下限，防止初期过激裁剪（默认 1.0，建议与网络梯度量级匹配）")
    parser.add_argument("--use-cbam", dest="use_cbam", action="store_true", default=False,
                        help="[CBAM] 在 Generator 编码器每个 block 后插入 CBAM 注意力模块")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())