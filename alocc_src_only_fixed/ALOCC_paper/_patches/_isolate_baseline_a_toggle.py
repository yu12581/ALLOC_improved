"""[BASELINE-A][ADR-007] Refactor: convert hard-coded D2-B verbatim RMSprop into an opt-in toggle.

Restores ALOCC.__init__ defaults to project-historical PyTorch RMSprop (alpha=0.99, eps=1e-8)
so the S1D subclass (which inherits __init__) gets its historical optimizer environment back.
The TF1.15 verbatim values (alpha=0.9, eps=1e-10) are preserved but only activated when the
new constructor flag `tf_verbatim_rmsprop=True` is passed (Baseline A path).

D1-B (BCE-on-noisy refinement in base ALOCC._train) is intentionally NOT touched here:
the S1D class (ALOCC_LOSS) overrides _train and never invokes the base implementation,
so D1-B does not leak. variant=alocc is the Baseline A path and keeps D1-B as-is.

This script is idempotent: re-running it on an already-patched file is a no-op.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

MODEL = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\model.py")
RUNNER = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\mnist_experiment_runner.py")

# ----------------------------------------------------------------------
# model.py edits
# ----------------------------------------------------------------------

OLD_INIT_TAIL = (
    '        bottleneck_rank: int = 0,  # [S1-BOT]\n'
    '        bottleneck_dropout: float = 0.0,  # [S1-BOT]\n'
    '        bottleneck_noise_type: str = "dropout",  # [S1-BOT]\n'
    '    ):\n'
    '        super(ALOCC, self).__init__()\n'
    '        self.device = DEVICE\n'
    '        self.label_smoothing = float(label_smoothing)\n'
)

NEW_INIT_TAIL = (
    '        bottleneck_rank: int = 0,  # [S1-BOT]\n'
    '        bottleneck_dropout: float = 0.0,  # [S1-BOT]\n'
    '        bottleneck_noise_type: str = "dropout",  # [S1-BOT]\n'
    '        tf_verbatim_rmsprop: bool = False,  # [BASELINE-A][D2-B][ADR-007] opt-in TF1.15 RMSprop defaults\n'
    '    ):\n'
    '        super(ALOCC, self).__init__()\n'
    '        self.device = DEVICE\n'
    '        self.label_smoothing = float(label_smoothing)\n'
    '        self.tf_verbatim_rmsprop = bool(tf_verbatim_rmsprop)  # [BASELINE-A][D2-B]\n'
)

OLD_OPT = (
    '        # [BASELINE-A][D2-B][TF1.15 models.py L161] tf.train.RMSPropOptimizer defaults: decay=0.9, epsilon=1e-10\n'
    '        self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=lr, alpha=0.9, eps=1e-10, weight_decay=weight_decay, foreach=True)\n'
    '        # [BASELINE-A][D2-B][TF1.15 models.py L162] same as D\n'
    '        self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=lr, alpha=0.9, eps=1e-10, weight_decay=weight_decay, foreach=True)\n'
)

NEW_OPT = (
    '        # [BASELINE-A][D2-B][ADR-007] Default = PyTorch RMSprop defaults (project-historical alpha=0.99, eps=1e-8).\n'
    '        # When tf_verbatim_rmsprop=True -> TF1.15 verbatim (decay=0.9, epsilon=1e-10) per ALOCC-original/models.py.\n'
    '        _rms_alpha = 0.9 if self.tf_verbatim_rmsprop else 0.99\n'
    '        _rms_eps   = 1e-10 if self.tf_verbatim_rmsprop else 1e-8\n'
    '        self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=lr, alpha=_rms_alpha, eps=_rms_eps, weight_decay=weight_decay, foreach=True)\n'
    '        self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=lr, alpha=_rms_alpha, eps=_rms_eps, weight_decay=weight_decay, foreach=True)\n'
)

# ----------------------------------------------------------------------
# mnist_experiment_runner.py edits
# ----------------------------------------------------------------------

OLD_BUILD_SIG = (
    'def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0,  # [S1-BOT]\n'
    '                bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = "dropout"):\n'
    '    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type)\n'
    '    if variant == "alocc":\n'
    '        return ALOCC(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
    '    if variant == "alocc_tiny":\n'
    '        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
    '    if variant == "alocc_loss":\n'
    '        return ALOCC_LOSS(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
    '    if variant == "alocc_loss_cls":\n'
    '        return ALOCC_LOSS_CLS(in_h=28, out_h=28, lr=lr, classify=True, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
)

NEW_BUILD_SIG = (
    'def build_model(variant: str, lr: float, weight_decay: float = 0.0, label_smoothing: float = 0.0,  # [S1-BOT]\n'
    '                bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = "dropout",\n'
    '                tf_verbatim_rmsprop: bool = False):  # [BASELINE-A][D2-B][ADR-007]\n'
    '    _bk = dict(bottleneck_rank=bottleneck_rank, bottleneck_dropout=bottleneck_dropout, bottleneck_noise_type=bottleneck_noise_type,\n'
    '               tf_verbatim_rmsprop=tf_verbatim_rmsprop)\n'
    '    if variant == "alocc":\n'
    '        return ALOCC(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
    '    if variant == "alocc_tiny":\n'
    '        return ALOCC(in_h=28, out_h=28, gf_dim=8, df_dim=8, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
    '    if variant == "alocc_loss":\n'
    '        return ALOCC_LOSS(in_h=28, out_h=28, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
    '    if variant == "alocc_loss_cls":\n'
    '        return ALOCC_LOSS_CLS(in_h=28, out_h=28, lr=lr, classify=True, weight_decay=weight_decay, label_smoothing=label_smoothing, **_bk)\n'
)

OLD_BUILD_CALL = (
    '    model = build_model(\n'
    '        variant=args.variant,\n'
    '        lr=args.lr,\n'
    '        weight_decay=getattr(args, "weight_decay", 0.0) or 0.0,\n'
    '        label_smoothing=getattr(args, "label_smoothing", 0.0) or 0.0,\n'
    '        bottleneck_rank=int(getattr(args, "bottleneck_rank", 0) or 0),  # [S1-BOT]\n'
    '        bottleneck_dropout=float(getattr(args, "bottleneck_dropout", 0.0) or 0.0),\n'
    '        bottleneck_noise_type=str(getattr(args, "bottleneck_noise_type", "dropout")),\n'
    '    )\n'
)

NEW_BUILD_CALL = (
    '    model = build_model(\n'
    '        variant=args.variant,\n'
    '        lr=args.lr,\n'
    '        weight_decay=getattr(args, "weight_decay", 0.0) or 0.0,\n'
    '        label_smoothing=getattr(args, "label_smoothing", 0.0) or 0.0,\n'
    '        bottleneck_rank=int(getattr(args, "bottleneck_rank", 0) or 0),  # [S1-BOT]\n'
    '        bottleneck_dropout=float(getattr(args, "bottleneck_dropout", 0.0) or 0.0),\n'
    '        bottleneck_noise_type=str(getattr(args, "bottleneck_noise_type", "dropout")),\n'
    '        tf_verbatim_rmsprop=bool(getattr(args, "tf_verbatim_rmsprop", False)),  # [BASELINE-A][D2-B]\n'
    '    )\n'
)

OLD_SEED_ARG = (
    '    parser.add_argument("--seed", type=int, default=None,\n'
)

NEW_SEED_ARG = (
    '    parser.add_argument("--tf-verbatim-rmsprop", dest="tf_verbatim_rmsprop", action="store_true", default=False,\n'
    '                        help="[BASELINE-A][D2-B] opt-in TF1.15 RMSprop defaults (alpha=0.9, eps=1e-10); default off restores PyTorch defaults.")\n'
    '    parser.add_argument("--seed", type=int, default=None,\n'
)

OLD_SWITCH = (
    '            "seed": (int(args.seed) if getattr(args, "seed", None) is not None else None),  # [A4-SEED]\n'
    '        },\n'
)

NEW_SWITCH = (
    '            "seed": (int(args.seed) if getattr(args, "seed", None) is not None else None),  # [A4-SEED]\n'
    '            "tf_verbatim_rmsprop": bool(getattr(args, "tf_verbatim_rmsprop", False)),  # [BASELINE-A][D2-B]\n'
    '        },\n'
)


def patch(path: Path, replacements: list[tuple[str, str]]) -> None:
    src = path.read_text(encoding="utf-8")
    bak = path.with_suffix(path.suffix + ".pre_isolate.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[OK] backup -> {bak}")
    out = src
    for old, new in replacements:
        if new in out and old not in out:
            print(f"[SKIP] already patched ({path.name})")
            continue
        if old not in out:
            raise SystemExit(f"[ERR] anchor not found in {path}: {old[:80]!r}")
        out = out.replace(old, new, 1)
    if out != src:
        path.write_text(out, encoding="utf-8")
        print(f"[OK] patched {path}")
    else:
        print(f"[NOOP] {path} already up-to-date")


if __name__ == "__main__":
    patch(MODEL, [(OLD_INIT_TAIL, NEW_INIT_TAIL), (OLD_OPT, NEW_OPT)])
    patch(RUNNER, [
        (OLD_BUILD_SIG, NEW_BUILD_SIG),
        (OLD_BUILD_CALL, NEW_BUILD_CALL),
        (OLD_SEED_ARG, NEW_SEED_ARG),
        (OLD_SWITCH, NEW_SWITCH),
    ])
    print("[DONE]")
