"""S1 Step 3b: ADR-007 bitwise-parity check for `flags OFF`.

Approach:
  1. With fixed seed (utils.set_random_seed -> 42), construct a Generator
     via the CURRENT (post-S1) model.py, flags OFF.
  2. Restore model.py from model.py.s1_bot.bak into a temp path, import it
     as a separate module, construct an equivalent pre-S1 Generator.
  3. Compare every parameter tensor element-wise (assert exact equality).
  4. If identical -> ADR-007 satisfied at the initialization level.
     Divergence would indicate RNG consumption changed by the patch.

Read-only on model.py (does not modify working tree).
"""
from __future__ import annotations
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\Trae_coding\ALLOC\ALOCC-master")
MODEL = ROOT / "model.py"
BAK   = ROOT / "model.py.s1_bot.bak"

if not BAK.exists():
    raise SystemExit(f"[fail] missing backup: {BAK}")

sys.path.insert(0, str(ROOT))

# --- pre-S1 module ---
import tempfile
tmp_dir = Path(tempfile.mkdtemp(prefix="pre_s1_"))
pre_model_path = tmp_dir / "model_pre_s1.py"
shutil.copyfile(BAK, pre_model_path)

spec_pre = importlib.util.spec_from_file_location("model_pre_s1", pre_model_path)
mod_pre = importlib.util.module_from_spec(spec_pre)
# the backup file imports `utils`, which is in ROOT — sys.path already has it
spec_pre.loader.exec_module(mod_pre)

# --- post-S1 module (current working tree) ---
import utils  # triggers set_random_seed()
import model as mod_post

# Re-seed and instantiate both Generators; post first, then re-seed, then pre
utils.set_random_seed()
g_post = mod_post.Generator(1, 16, 16, 28)

utils.set_random_seed()
g_pre = mod_pre.Generator(1, 16, 16, 28)

# Compare
sd_post = dict(g_post.state_dict())
sd_pre  = dict(g_pre.state_dict())

# pre cannot contain 'bottleneck.*' keys; post should have no bottleneck.* keys
# because self.bottleneck = nn.Identity() has no parameters.
extra_post = sorted(set(sd_post.keys()) - set(sd_pre.keys()))
extra_pre  = sorted(set(sd_pre.keys()) - set(sd_post.keys()))
if extra_post or extra_pre:
    print(f"[fail]  state_dict key mismatch")
    print(f"  only in post: {extra_post}")
    print(f"  only in pre:  {extra_pre}")
    sys.exit(2)

import torch
max_abs = 0.0
max_key = None
for k, v_pre in sd_pre.items():
    v_post = sd_post[k]
    if v_pre.shape != v_post.shape:
        print(f"[fail]  shape mismatch on {k}: pre={tuple(v_pre.shape)} post={tuple(v_post.shape)}")
        sys.exit(2)
    diff = (v_pre - v_post).abs().max().item()
    if diff > max_abs:
        max_abs, max_key = diff, k

print(f"[ok]   state_dict keys match ({len(sd_pre)} tensors)")
print(f"[ok]   max |pre - post| = {max_abs:.3e} at key={max_key!r}")
if max_abs == 0.0:
    print("[PASS] ADR-007 bitwise parity: flags OFF == pre-S1 backup")
else:
    print("[FAIL] non-zero divergence detected; investigate RNG consumption")
    sys.exit(3)

# Also verify that turning flags ON does produce new parameters under the expected keys.
utils.set_random_seed()
g_on = mod_post.Generator(1, 16, 16, 28, bottleneck_rank=16, bottleneck_dropout=0.3)
sd_on = dict(g_on.state_dict())
new_keys = sorted(set(sd_on.keys()) - set(sd_pre.keys()))
print(f"[info] flags ON adds {len(new_keys)} tensors: {new_keys}")
assert any("bottleneck.down" in k for k in new_keys), "bottleneck.down weight missing when ON"
assert any("bottleneck.up" in k for k in new_keys), "bottleneck.up weight missing when ON"
print("[ok]   flags ON registers bottleneck.down / bottleneck.up weights")
