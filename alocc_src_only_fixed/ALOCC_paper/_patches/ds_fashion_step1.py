"""DS-FM Step 1: add Fashion-MNIST support behind --dataset flag.

- Generates FashionMNIST.py from MNIST.py via targeted token swaps (no MNIST.py edits).
- Patches mnist_experiment_runner.py to dispatch dataset class by args.dataset.
- Patches run_paper_mnist_figure6_7.py with --dataset flag + Args.dataset + figure7 dispatch.
- Default (--dataset mnist) preserves bitwise-identical behavior (ADR-007).
- Idempotent. Sentinel [DS-FM]. Backup suffix .ds_fm.bak.
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
SENTINEL = "[DS-FM]"


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".ds_fm.bak")
    if not bak.exists():
        shutil.copy2(p, bak)


def gen_fashion_mnist() -> None:
    src = ROOT / "MNIST.py"
    dst = ROOT / "FashionMNIST.py"
    txt = src.read_text(encoding="utf-8")
    # class rename (only the explicit symbols, not lowercase string paths)
    txt = txt.replace("class MNIST(Dataset):", "class FashionMNIST(Dataset):  # [DS-FM]")
    txt = txt.replace("data_root = 'dataset/mnist'", "data_root = 'dataset/fashion_mnist'")
    txt = txt.replace("datasets.MNIST(", "datasets.FashionMNIST(")
    txt = re.sub(r"\bMNIST\.train_dataset\b", "FashionMNIST.train_dataset", txt)
    txt = re.sub(r"\bMNIST\.test_dataset\b", "FashionMNIST.test_dataset", txt)
    # rename constructor-style calls in helper fns (train/test/..._class)
    txt = re.sub(r"\bMNIST\(", "FashionMNIST(", txt)
    txt = txt.replace("Loaded {len(self.data)} MNIST images", "Loaded {len(self.data)} Fashion-MNIST images")
    # checkpoint path strings kept as-is (they live under ./checkpoint/mnist_* which is fine for helpers)
    if f"# {SENTINEL}" not in txt:
        txt = "# [DS-FM] auto-generated from MNIST.py; do not edit by hand (see _patches/ds_fashion_step1.py)\n" + txt
    dst.write_text(txt, encoding="utf-8")
    print(f"[DS-FM] wrote {dst}")


def patch_runner() -> None:
    p = ROOT / "mnist_experiment_runner.py"
    txt = p.read_text(encoding="utf-8")
    if SENTINEL in txt:
        print(f"[DS-FM] runner already patched, skip")
        return
    _backup(p)

    # Insert dispatch helper right after `from MNIST import MNIST`
    anchor_import = "from MNIST import MNIST\n"
    inject_import = (
        "from MNIST import MNIST\n"
        "# [DS-FM] dataset class dispatch (default mnist = bitwise identical to pre-DS-FM)\n"
        "def _dataset_cls(args):  # [DS-FM]\n"
        "    name = (getattr(args, 'dataset', 'mnist') or 'mnist').lower()\n"
        "    if name == 'mnist':\n"
        "        return MNIST\n"
        "    if name == 'fashion':\n"
        "        from FashionMNIST import FashionMNIST\n"
        "        return FashionMNIST\n"
        "    raise ValueError(f\"Unknown --dataset: {name}\")\n"
    )
    assert anchor_import in txt, "anchor import not found"
    txt = txt.replace(anchor_import, inject_import, 1)

    # Replace the 3 MNIST(...) call sites inside build_data
    old_train = "    train_dataset = MNIST(\n        train=True,\n        specific=args.specific,\n        count=args.train_count,\n        noise_std=args.noise_std,\n    )\n"
    new_train = "    _DS = _dataset_cls(args)  # [DS-FM]\n    train_dataset = _DS(\n        train=True,\n        specific=args.specific,\n        count=args.train_count,\n        noise_std=args.noise_std,\n    )\n"
    assert old_train in txt, "build_data train_dataset block not found"
    txt = txt.replace(old_train, new_train, 1)

    old_test = "    test_dataset = MNIST(\n        train=False,\n        specific=args.specific,\n        count=args.test_inlier_count,\n        out_class_scale=args.test_outlier_scale,\n        noise_std=args.noise_std,\n    )\n"
    new_test = "    test_dataset = _DS(  # [DS-FM]\n        train=False,\n        specific=args.specific,\n        count=args.test_inlier_count,\n        out_class_scale=args.test_outlier_scale,\n        noise_std=args.noise_std,\n    )\n"
    assert old_test in txt, "build_data test_dataset block not found"
    txt = txt.replace(old_test, new_test, 1)

    old_out = "        outclass_dataset = MNIST(\n            train=True,\n            count=0,\n            specific=args.specific,\n            per_out_class_count=args.out_per_class_count,\n            noise_std=args.noise_std,\n        )\n"
    new_out = "        outclass_dataset = _DS(  # [DS-FM]\n            train=True,\n            count=0,\n            specific=args.specific,\n            per_out_class_count=args.out_per_class_count,\n            noise_std=args.noise_std,\n        )\n"
    assert old_out in txt, "build_data outclass_dataset block not found"
    txt = txt.replace(old_out, new_out, 1)

    p.write_text(txt, encoding="utf-8")
    print(f"[DS-FM] patched {p}")


def patch_entrypoint() -> None:
    p = ROOT / "run_paper_mnist_figure6_7.py"
    txt = p.read_text(encoding="utf-8")
    if SENTINEL in txt:
        print(f"[DS-FM] entrypoint already patched, skip")
        return
    _backup(p)

    # Args dataclass field
    anchor_seed_field = "    seed: int | None = None  # [A4-SEED]\n"
    inject_seed_field = "    seed: int | None = None  # [A4-SEED]\n    dataset: str = \"mnist\"  # [DS-FM]\n"
    assert anchor_seed_field in txt, "Args seed field not found"
    txt = txt.replace(anchor_seed_field, inject_seed_field, 1)

    # _figure7_scores: add dataset kwarg + dispatch
    old_sig = "def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int,\n                    bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\"):  # [S1-BOT]\n"
    new_sig = "def _figure7_scores(checkpoint_path: str, variant: str, specific: int, outlier_labels: list[int], noise_std: float, sample_count: int,\n                    bottleneck_rank: int = 0, bottleneck_dropout: float = 0.0, bottleneck_noise_type: str = \"dropout\",\n                    dataset: str = \"mnist\"):  # [S1-BOT] [DS-FM]\n"
    assert old_sig in txt, "_figure7_scores signature not found"
    txt = txt.replace(old_sig, new_sig, 1)

    old_ds_call = "    test_dataset = MNIST(\n        train=False,\n        specific=specific,\n        count=sample_count,\n        out_class_scale=1.0,\n        noise_std=noise_std,\n    )\n"
    new_ds_call = ("    # [DS-FM] dispatch test dataset by name\n"
                   "    if dataset == 'fashion':\n"
                   "        from FashionMNIST import FashionMNIST as _DSCls\n"
                   "    else:\n"
                   "        _DSCls = MNIST\n"
                   "    test_dataset = _DSCls(\n        train=False,\n        specific=specific,\n        count=sample_count,\n        out_class_scale=1.0,\n        noise_std=noise_std,\n    )\n")
    assert old_ds_call in txt, "_figure7_scores MNIST() call not found"
    txt = txt.replace(old_ds_call, new_ds_call, 1)

    # CLI flag (append right after --seed)
    anchor_seed_cli = "    parser.add_argument(\"--seed\", type=int, default=None, help=\"[A4-SEED] RNG seed; None=42 (historical default)\")\n"
    inject_seed_cli = anchor_seed_cli + "    parser.add_argument(\"--dataset\", choices=[\"mnist\", \"fashion\"], default=\"mnist\", help=\"[DS-FM] dataset source\")\n"
    assert anchor_seed_cli in txt, "--seed CLI flag not found"
    txt = txt.replace(anchor_seed_cli, inject_seed_cli, 1)

    # Args() constructor: add dataset=args.dataset (place after seed=args.seed,)
    anchor_seed_arg = "        seed=args.seed,  # [A4-SEED]\n"
    inject_seed_arg = "        seed=args.seed,  # [A4-SEED]\n        dataset=args.dataset,  # [DS-FM]\n"
    assert anchor_seed_arg in txt, "Args(seed=) init line not found"
    txt = txt.replace(anchor_seed_arg, inject_seed_arg, 1)

    # _figure7_scores() call: thread dataset kwarg
    old_fig_call = "    figure7 = _figure7_scores(\n        checkpoint_path=str(best_ckpt),\n        variant=args.variant,\n        specific=args.specific,\n        outlier_labels=args.outlier_labels,\n        noise_std=args.noise_std,\n        sample_count=args.figure7_sample_count,\n        bottleneck_rank=args.bottleneck_rank,  # [S1-BOT]\n        bottleneck_dropout=args.bottleneck_dropout,\n        bottleneck_noise_type=args.bottleneck_noise_type,\n    )\n"
    new_fig_call = old_fig_call.replace(
        "        bottleneck_noise_type=args.bottleneck_noise_type,\n    )\n",
        "        bottleneck_noise_type=args.bottleneck_noise_type,\n        dataset=args.dataset,  # [DS-FM]\n    )\n",
    )
    assert old_fig_call in txt, "_figure7_scores outer call not found"
    txt = txt.replace(old_fig_call, new_fig_call, 1)

    # summary payload: record dataset
    anchor_summary_seed = "        \"seed\": args.seed,  # [A4-SEED]\n"
    inject_summary_seed = anchor_summary_seed + "        \"dataset\": args.dataset,  # [DS-FM]\n"
    assert anchor_summary_seed in txt, "summary seed field not found"
    txt = txt.replace(anchor_summary_seed, inject_summary_seed, 1)

    p.write_text(txt, encoding="utf-8")
    print(f"[DS-FM] patched {p}")


if __name__ == "__main__":
    gen_fashion_mnist()
    patch_runner()
    patch_entrypoint()
    print("[DS-FM] step1 done.")
