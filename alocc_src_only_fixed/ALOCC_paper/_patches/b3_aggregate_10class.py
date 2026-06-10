"""[B-3] Aggregate 20 summaries (10 S1 + 10 OFF) under unified redline replay.

Produces:
- ALOCC_paper/s1_redline_10class.json (machine-readable)
- ALOCC_paper/s1_redline_10class.md (human-readable table)
"""
from __future__ import annotations
import json
import pathlib

ROOT = pathlib.Path(r"D:\Trae_coding\ALOCC_paper")

# Redline thresholds (match ADR-011 defaults).
TAU_OC = 0.15
TAU_RAW = 0.60


def redline_select(records: list[dict]) -> tuple[dict, bool, str]:
    """Return (selected_record, fallback, reason). Selection: earliest epoch with
    ssim_oc<=TAU_OC AND raw_auc>=TAU_RAW; fallback to max distortion_score."""
    eligible = [r for r in records
                if r.get("ssim_oc", 9.9) <= TAU_OC
                and r.get("raw_auc", 0.0) >= TAU_RAW]
    if eligible:
        pick = min(eligible, key=lambda r: r["epoch"])
        return pick, False, ""
    # fallback: pick max distortion_score if present else max raw_auc
    if "distortion_score" in records[0]:
        pick = max(records, key=lambda r: r.get("distortion_score", -1))
        return pick, True, "no epoch meets redline; fell back to max distortion_score"
    pick = max(records, key=lambda r: r.get("raw_auc", -1))
    return pick, True, "no epoch meets redline; fell back to max raw_auc"


def load_summary(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def config_dir(cfg: str, c: int) -> pathlib.Path:
    """Locate the experiment dir for (cfg, class)."""
    if cfg == "S1":
        return ROOT / f"s1_c{c}_r16_p03_redline" / "experiment"
    # OFF
    legacy = ROOT / f"s1_c{c}_off" / "experiment"
    if legacy.exists():
        return legacy
    return ROOT / f"s1_c{c}_off_redline" / "experiment"


def row(cfg: str, c: int) -> dict:
    d = config_dir(cfg, c)
    s = load_summary(d / "summary.json")
    if s is None:
        return {"class": c, "cfg": cfg, "status": "MISSING", "dir": str(d)}
    records = s.get("records", [])
    pick, fallback, reason = redline_select(records)

    # identity-shortcut trajectory metrics (from full records[])
    r_first = records[0]
    r_last = records[-1]
    max_gap_rec = max(records, key=lambda r: r["ssim_gap"])
    # correlation-ish: do ic / oc rise together?
    ic_rise = r_last["ssim_ic"] - r_first["ssim_ic"]
    oc_rise = r_last["ssim_oc"] - r_first["ssim_oc"]
    # coupling ratio: if oc_rise / ic_rise ~= 1.0, identity shortcut is complete.
    coupling = round(oc_rise / ic_rise, 3) if abs(ic_rise) > 1e-6 else None

    return {
        "class": c,
        "cfg": cfg,
        "dir": d.parent.name,
        "best_epoch_orig": s.get("best_epoch"),
        "strategy_orig": s.get("selection_info", {}).get("strategy"),
        "best_epoch_redline": pick["epoch"],
        "auc": round(pick.get("refined_auc", pick.get("auc", 0)), 4),
        "raw_auc": round(pick["raw_auc"], 4),
        "ssim_oc": round(pick["ssim_oc"], 4),
        "ssim_ic": round(pick["ssim_ic"], 4),
        "ssim_gap": round(pick["ssim_gap"], 4),
        "score_gap": round(pick["score_gap"], 5),
        "raw_score_gap": round(pick["raw_score_gap"], 5),
        "redline_pass": (not fallback),
        "redline_reason": reason,
        # trajectory
        "ic_ep1": round(r_first["ssim_ic"], 4),
        "ic_ep10": round(r_last["ssim_ic"], 4),
        "oc_ep1": round(r_first["ssim_oc"], 4),
        "oc_ep10": round(r_last["ssim_oc"], 4),
        "ic_rise": round(ic_rise, 4),
        "oc_rise": round(oc_rise, 4),
        "coupling_ratio": coupling,
        "max_gap_epoch": max_gap_rec["epoch"],
        "max_gap": round(max_gap_rec["ssim_gap"], 4),
        "max_gap_raw_auc": round(max_gap_rec["raw_auc"], 4),
        "max_gap_ssim_oc": round(max_gap_rec["ssim_oc"], 4),
    }


def main():
    rows = []
    for c in range(10):
        for cfg in ("OFF", "S1"):
            rows.append(row(cfg, c))

    # split for per-config pass rate
    s1_pass = sum(1 for r in rows if r["cfg"] == "S1" and r.get("redline_pass"))
    off_pass = sum(1 for r in rows if r["cfg"] == "OFF" and r.get("redline_pass"))

    # auc gain per class (S1.auc - OFF.auc, redline-selected)
    by_cc = {(r["class"], r["cfg"]): r for r in rows}
    deltas = []
    for c in range(10):
        s1r = by_cc.get((c, "S1"))
        offr = by_cc.get((c, "OFF"))
        if s1r and offr and "auc" in s1r and "auc" in offr:
            deltas.append({
                "class": c,
                "auc_off": offr["auc"],
                "auc_s1": s1r["auc"],
                "d_auc": round(s1r["auc"] - offr["auc"], 4),
                "oc_off": offr["ssim_oc"],
                "oc_s1": s1r["ssim_oc"],
                "d_oc": round(s1r["ssim_oc"] - offr["ssim_oc"], 4),
                "off_pass": offr.get("redline_pass", False),
                "s1_pass": s1r.get("redline_pass", False),
            })

    payload = {
        "config": {"tau_oc": TAU_OC, "tau_raw_auc": TAU_RAW,
                   "selection": "earliest epoch with ssim_oc<=tau_oc AND raw_auc>=tau_raw_auc"},
        "redline_pass_rate": {"OFF": f"{off_pass}/10", "S1": f"{s1_pass}/10"},
        "rows": rows,
        "deltas": deltas,
    }
    (ROOT / "s1_redline_10class.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # MD table
    md = ["# S1 + Redline: 10-class MNIST Robustness (B-3 aggregate)\n",
          f"Redline: ssim_oc <= {TAU_OC} AND raw_auc >= {TAU_RAW}; earliest epoch.",
          f"\n**Redline pass rate**: OFF = {off_pass}/10, S1 = {s1_pass}/10 "
          f"(net +{s1_pass - off_pass})\n",
          "| class | cfg | ep | AUC | raw_AUC | ssim_oc | ssim_gap | redline |",
          "|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for r in rows:
        pass_mark = "✅" if r.get("redline_pass") else "❌"
        md.append(
            f"| {r['class']} | {r['cfg']} | {r.get('best_epoch_redline','-')} | "
            f"{r.get('auc','-')} | {r.get('raw_auc','-')} | "
            f"{r.get('ssim_oc','-')} | {r.get('ssim_gap','-')} | {pass_mark} |"
        )
    md.append("\n## Per-class ΔAUC / Δssim_oc (S1 vs OFF, redline-selected)\n")
    md.append("| class | OFF AUC | S1 AUC | ΔAUC | OFF ssim_oc | S1 ssim_oc | Δssim_oc | OFF | S1 |")
    md.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for d in deltas:
        md.append(
            f"| {d['class']} | {d['auc_off']} | {d['auc_s1']} | "
            f"{d['d_auc']:+} | {d['oc_off']} | {d['oc_s1']} | {d['d_oc']:+} | "
            f"{'✅' if d['off_pass'] else '❌'} | {'✅' if d['s1_pass'] else '❌'} |"
        )

    # --- Identity-shortcut trajectory section -----------------------------
    md.append("\n## Identity-shortcut trajectory (ssim_ic / ssim_oc 是否反趋势？)\n")
    md.append(
        "理想 AE：`ssim_ic ↑` 且 `ssim_oc ↓ / 持平`；实际观测：二者几乎同向上扬。\n"
        "`coupling_ratio = Δssim_oc / Δssim_ic`（ep1→ep10）；1.0 = 完全同步、"
        "0.0 = 理想解耦、负值 = 反趋势。`max_gap` 在 `raw_auc` 已崩时达到则说明"
        "gap 由 D 失效伪造。\n"
    )
    md.append("| class | cfg | ic ep1→ep10 | oc ep1→ep10 | coupling | max_gap@ep | gap值 | 该ep raw_auc |")
    md.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for r in rows:
        if r.get("status") == "MISSING":
            continue
        md.append(
            f"| {r['class']} | {r['cfg']} | "
            f"{r['ic_ep1']}→{r['ic_ep10']} | "
            f"{r['oc_ep1']}→{r['oc_ep10']} | "
            f"{r['coupling_ratio']} | "
            f"ep{r['max_gap_epoch']} | {r['max_gap']} | "
            f"{r['max_gap_raw_auc']} |"
        )

    # summary stats for coupling
    def mean(L):
        L = [x for x in L if x is not None]
        return round(sum(L) / len(L), 3) if L else None
    off_couple = mean([r["coupling_ratio"] for r in rows if r["cfg"] == "OFF"])
    s1_couple = mean([r["coupling_ratio"] for r in rows if r["cfg"] == "S1"])
    md.append(f"\n**平均 coupling_ratio（10 类）**：OFF = {off_couple}, "
              f"S1 = {s1_couple}（S1 降低 identity-shortcut 同步度 "
              f"= {round((off_couple - s1_couple), 3) if off_couple and s1_couple else 'N/A'}）\n")

    # max-gap vs redline comparison
    md.append("## Max-gap epoch vs redline epoch 对照（S1 only）\n")
    md.append(
        "如果按 `max ssim_gap` 选模会怎样？ 和 redline 比较，尤其看 raw_auc：\n"
    )
    md.append("| class | redline ep | redline oc | redline raw_auc | max-gap ep | max-gap oc | max-gap raw_auc | 判定 |")
    md.append("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for r in rows:
        if r["cfg"] != "S1" or r.get("status") == "MISSING":
            continue
        rl_ep = r["best_epoch_redline"]
        mg_ep = r["max_gap_epoch"]
        same = (rl_ep == mg_ep)
        if same:
            verdict = "一致"
        elif r["max_gap_raw_auc"] < 0.60:
            verdict = "max-gap 假象（raw_auc < 0.60）"
        else:
            verdict = "max-gap 更晚但 raw_auc 仍活"
        md.append(
            f"| {r['class']} | ep{rl_ep} | {r['ssim_oc']} | {r['raw_auc']} | "
            f"ep{mg_ep} | {r['max_gap_ssim_oc']} | {r['max_gap_raw_auc']} | {verdict} |"
        )

    (ROOT / "s1_redline_10class.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"off_pass": off_pass, "s1_pass": s1_pass,
                      "out_json": str(ROOT / "s1_redline_10class.json"),
                      "out_md": str(ROOT / "s1_redline_10class.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
