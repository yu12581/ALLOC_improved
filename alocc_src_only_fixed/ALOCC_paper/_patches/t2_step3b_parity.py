"""T2 Step 3b: bitwise parity check (ADR-007).

Procedure:
  1. Import current (patched) model.py -> build ALOCC_LOSS with default flags
     (spectral_norm_d=False, bottleneck_rank=0, bottleneck_dropout=0.0).
  2. Temporarily swap model.py <- model.py.t2_sn.bak (pre-T2 state), rebuild
     fresh Python subprocess, serialize its state_dict + forward outputs, and
     restore the patched file.
  3. Compare state_dict keys / tensor values and forward(x) outputs.
     Expect max-abs diff == 0 on every tensor.

Uses torch.manual_seed(42) and CUDA deterministic flags.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
MODEL = ROOT / "model.py"
BAK = Path(str(MODEL) + ".t2_sn.bak")
PYTHON = Path(r"D:\Trae_coding\ALLOC\ALOCC-master\.venv\Scripts\python.exe")

DUMP_SNIPPET = r"""
import sys, os, pickle, io
sys.path.insert(0, r"D:\Trae_coding\ALLOC\ALOCC-master")
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from model import ALOCC_LOSS
kw = {}
# patched version supports these kwargs; bak version does not -> pass only when accepted
import inspect
sig = inspect.signature(ALOCC_LOSS.__init__)
if "spectral_norm_d" in sig.parameters:
    kw["spectral_norm_d"] = False
if "bottleneck_rank" in sig.parameters:
    kw["bottleneck_rank"] = 0
if "bottleneck_dropout" in sig.parameters:
    kw["bottleneck_dropout"] = 0.0
m = ALOCC_LOSS(in_h=28, out_h=28, **kw)
sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
torch.manual_seed(123)
x = torch.randn(2, 1, 28, 28, device="cuda")
m.eval()
with torch.no_grad():
    r = m.G(x).detach().cpu()
    d_real = m.D(x).detach().cpu()
    d_fake = m.D(m.G(x)).detach().cpu()
out_path = sys.argv[1]
torch.save({"sd": sd, "r": r, "d_real": d_real, "d_fake": d_fake}, out_path)
print("DUMPED", out_path)
"""


def dump(out_path: Path) -> None:
    code = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
    code.write(DUMP_SNIPPET)
    code.close()
    try:
        r = subprocess.run([str(PYTHON), code.name, str(out_path)], capture_output=True, text=True, check=False)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            raise RuntimeError(f"dump failed (rc={r.returncode})")
    finally:
        Path(code.name).unlink(missing_ok=True)


def main() -> None:
    if not BAK.exists():
        raise SystemExit("[FAIL] model.py.t2_sn.bak missing; cannot compare")

    tmpdir = Path(tempfile.mkdtemp(prefix="t2_parity_"))
    patched_dump = tmpdir / "patched.pt"
    bak_dump = tmpdir / "bak.pt"

    # A. patched model (T2 present, flags OFF)
    print("[dump] patched model (flags OFF)")
    dump(patched_dump)

    # B. swap to bak, dump, swap back
    side = Path(str(MODEL) + ".parity_swap")
    shutil.copyfile(MODEL, side)
    try:
        shutil.copyfile(BAK, MODEL)
        print("[swap] model.py <- .t2_sn.bak")
        print("[dump] pre-T2 model")
        dump(bak_dump)
    finally:
        shutil.copyfile(side, MODEL)
        side.unlink()
        print("[swap] restored patched model.py")

    # C. compare
    import torch
    a = torch.load(patched_dump, map_location="cpu", weights_only=False)
    b = torch.load(bak_dump, map_location="cpu", weights_only=False)

    ka, kb = set(a["sd"].keys()), set(b["sd"].keys())
    if ka != kb:
        print(f"[FAIL] state_dict keys differ")
        print(f"  only in patched: {sorted(ka - kb)}")
        print(f"  only in bak:     {sorted(kb - ka)}")
        raise SystemExit(1)

    max_diff = 0.0
    for k in sorted(ka):
        t1, t2 = a["sd"][k], b["sd"][k]
        if t1.shape != t2.shape:
            print(f"[FAIL] shape mismatch on {k}: {t1.shape} vs {t2.shape}")
            raise SystemExit(1)
        d = (t1.float() - t2.float()).abs().max().item()
        if d > max_diff:
            max_diff = d
    print(f"[sd] max abs diff over {len(ka)} tensors: {max_diff}")

    fwd_max = 0.0
    for key in ("r", "d_real", "d_fake"):
        d = (a[key].float() - b[key].float()).abs().max().item()
        print(f"[fwd] {key}: max abs diff = {d}")
        fwd_max = max(fwd_max, d)

    if max_diff == 0.0 and fwd_max == 0.0:
        print("[ok] bitwise parity confirmed (ADR-007)")
    else:
        print(f"[FAIL] parity broken: sd={max_diff}, fwd={fwd_max}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
