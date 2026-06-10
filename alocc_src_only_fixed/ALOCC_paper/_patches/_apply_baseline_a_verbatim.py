"""[BASELINE-A] Apply TF1.15-verbatim corrections to model.py.

Decisions (from PROJECT_LOG audit 2026-04-27):
  D1-B: Refinement loss = BCE_with_logits(R(noisy), noisy)  [TF1.15 models.py L131]
  D2-B: RMSprop alpha=0.9, eps=1e-10                         [TF1.15 default RMSPropOptimizer]
  D3-B: Source-level reuse — annotate verbatim ports with line refs.
  D4-C: last_epoch selection — runner-side flag, no model.py change.

Subclasses (ALOCC_LOSS, ALOCC_LOSS_CLS) are S1D variants and are NOT touched
beyond the optimizer change inherited via __init__.
"""
from __future__ import annotations
import io
import os
import shutil
import sys
from datetime import datetime

TARGET = r"D:\Trae_coding\ALLOC\ALOCC-master\model.py"
BACKUP_SUFFIX = ".pre_baseline_a.bak"


def patch(src: str) -> str:
    out = src

    # --- D2-B: RMSprop alpha=0.9, eps=1e-10 (both D and G) -----------------
    old_d = ("self.optim_D = torch.optim.RMSprop(self.D.parameters(), "
             "lr=lr, weight_decay=weight_decay, foreach=True)")
    new_d = ("# [BASELINE-A][D2-B][TF1.15 models.py L161] tf.train.RMSPropOptimizer "
             "defaults: decay=0.9, epsilon=1e-10\n"
             "        self.optim_D = torch.optim.RMSprop(self.D.parameters(), "
             "lr=lr, alpha=0.9, eps=1e-10, weight_decay=weight_decay, foreach=True)")
    if old_d not in out:
        raise SystemExit("[FAIL] optim_D anchor not found")
    out = out.replace(old_d, new_d, 1)

    old_g = ("self.optim_G = torch.optim.RMSprop(self.G.parameters(), "
             "lr=lr, weight_decay=weight_decay, foreach=True)")
    new_g = ("# [BASELINE-A][D2-B][TF1.15 models.py L162] same as D\n"
             "        self.optim_G = torch.optim.RMSprop(self.G.parameters(), "
             "lr=lr, alpha=0.9, eps=1e-10, weight_decay=weight_decay, foreach=True)")
    if old_g not in out:
        raise SystemExit("[FAIL] optim_G anchor not found")
    out = out.replace(old_g, new_g, 1)

    # --- D1-B / D3-B: refinement loss = BCE-on-noisy (only inside ALOCC._train) -
    # The S1D subclasses (ALOCC_LOSS / ALOCC_LOSS_CLS) keep MSE-on-real via
    # self._refinement_loss helper. Only the base ALOCC._train switches.
    #
    # Anchor: the unique block that lives between
    #   "g_loss_gan = self.criterion_bce(fake_logits_new, real_label)"
    # and
    #   "g_loss = g_loss_gan + r_alpha * g_loss_r"
    # in the BASE ALOCC._train (NOT the LOSS subclasses, which have an
    # extra 'outclass_fake_imgs_new = ...' line right after).
    base_old = (
        "                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)\n"
        "                        # R should denoise back to the clean in-class target instead of copying noisy input.\n"
        "                        g_loss_r = self._refinement_loss(fake_imgs_new, real_imgs)\n"
        "                        g_loss = g_loss_gan + r_alpha * g_loss_r\n"
    )
    base_new = (
        "                        g_loss_gan = self.criterion_bce(fake_logits_new, real_label)\n"
        "                        # [BASELINE-A][D1-B][TF1.15 models.py L131] verbatim refinement:\n"
        "                        #   tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G, labels=self.z)\n"
        "                        # i.e. BCE-with-logits between G(noisy) and the noisy input itself.\n"
        "                        g_loss_r = F.binary_cross_entropy_with_logits(fake_imgs_new, noisy_imgs)\n"
        "                        g_loss = g_loss_gan + r_alpha * g_loss_r\n"
    )
    n_occurrences = out.count(base_old)
    if n_occurrences != 1:
        raise SystemExit(f"[FAIL] base ALOCC refinement anchor: expected 1 match, found {n_occurrences}")
    out = out.replace(base_old, base_new, 1)

    return out


def main():
    if not os.path.isfile(TARGET):
        raise SystemExit(f"[FAIL] target missing: {TARGET}")

    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    backup = TARGET + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(TARGET, backup)
        print(f"[OK] backup -> {backup}")
    else:
        print(f"[SKIP] backup already exists -> {backup}")

    new_src = patch(src)

    if new_src == src:
        print("[NOOP] no changes (already patched?)")
        return

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"[OK] patched {TARGET}")
    print(f"[OK] timestamp: {datetime.now().isoformat(timespec='seconds')}")
    # tag-grep summary
    for tag in ("[BASELINE-A][D1-B]", "[BASELINE-A][D2-B]"):
        n = new_src.count(tag)
        print(f"[CHECK] tag {tag} occurrences = {n}")


if __name__ == "__main__":
    main()
