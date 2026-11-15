"""Contact-state analysis: when the GT hand part touches / nearly touches / is away from the object, where does
the predicted part end up relative to the same (GT) object?

Unit = one hand part (palm, thumb, index, middle, ring, pinky) of one detected hand-frame. Part signed distance =
min over its palmar+dorsal regions of the signed distance (mm) from the hand surface to the GT object
(negative = inside the object), as computed by eval_arctic.py (<region>_minsd; ra_<region>_minsd = hand moved
onto the GT wrist; run eval_arctic.py with --sdf-pad 150 so distances are not clipped at 30 mm). Fingertip position error (tip_<f>_err, tip_<f>_err_ra) is from eval_arctic.py too.

Usage: python baselines/analyze_contact.py results/baselines/<tag>_frames.parquet [--out results/baselines/<tag>_contact]
Writes <out>.md (tables) and <out>.json (histograms for plotting).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ["palm", "thumb", "index", "middle", "ring", "pinky"]
GT_BINS = [(-np.inf, 2, "touching (≤2 mm)"), (2, 10, "near (2–10 mm)"), (10, 30, "free (10–30 mm)"), (30, 50, "clear (30–50 mm)"),
           (50, np.inf, "away (>50 mm)")]
PRED_BINS = [(-np.inf, -5, "deep pen <−5"), (-5, -2, "pen −5…−2"), (-2, 2, "touch ±2"), (2, 10, "hover 2–10"),
             (10, 50, "off 10–50"), (50, np.inf, "far >50")]
HIST_EDGES = np.r_[-np.inf, np.arange(-30, 101, 2.5), np.inf]


def bin_of(x, bins):
    out = np.full(len(x), -1)
    for i, (lo, hi, _) in enumerate(bins):
        out[(x > lo) & (x <= hi)] = i
    return out


def long_parts(d, prefix=""):
    """hand-frame rows -> one row per part with sd (mm) and fingertip errors."""
    rows = []
    for p in PARTS:
        sd = np.minimum(d[f"{prefix}{p}_palmar_minsd"], d[f"{prefix}{p}_dorsal_minsd"])
        r = d[["seq", "frame", "hand", "method", "lift", "obj"]].copy()
        r["part"], r["sd"] = p, sd.to_numpy()
        if p != "palm" and f"tip_{p}_err" in d:
            r["tip_err"], r["tip_err_ra"] = d[f"tip_{p}_err"].to_numpy(), d[f"tip_{p}_err_ra"].to_numpy()
        else:
            r["tip_err"], r["tip_err_ra"] = d["root_err"].to_numpy(), 0.0   # palm: wrist error
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out or a.frames.replace("_frames.parquet", "_contact"))
    d = pd.read_parquet(a.frames)
    key = ["seq", "frame", "hand"]
    gt = long_parts(d[d.method == "gt"])[key + ["part", "sd"]].rename(columns={"sd": "gt_sd"})
    pr = d[(d.method != "gt") & d.detected & d.pa_mpjpe.notna()]
    variants = []
    for (m, lift), g in pr.groupby(["method", "lift"], sort=False):
        variants.append((f"{m} {lift}", long_parts(g)))
        if "ra_palm_palmar_minsd" in g:
            ra = long_parts(g, "ra_")
            ra["tip_err"] = ra["tip_err_ra"].where(ra.part != "palm", np.nan)   # wrist-aligned tip error
            variants.append((f"{m} wrist-aligned", ra))
    order = {"hamer": 0, "wilor": 1, "hopformer": 2}
    variants.sort(key=lambda v: (order.get(v[0].split()[0], 9), "aligned" in v[0]))
    seen, uniq = set(), []
    for name, v in variants:        # wrist-aligned is the same for both lifts of a 2D method: keep one
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, v))

    md = [f"# Contact-state analysis: {Path(a.frames).name}", "",
          "Unit: one hand part (palm + 5 fingers) of a detected hand-frame. Signed distance to the GT object, mm "
          "(negative = inside). Rows: GT part state; columns: where the predicted part lands (row %). "
          "`wrist-aligned` = predicted hand translated onto the GT wrist (pose/shape only, no placement error).", ""]
    hist = {"edges": HIST_EDGES[1:-1].tolist(), "gt_bins": [b[2] for b in GT_BINS], "variants": {}}
    summary = []
    for name, v in uniq:
        j = v.merge(gt, on=key + ["part"])
        j = j[np.isfinite(j.sd) & np.isfinite(j.gt_sd)]
        gb, pb = bin_of(j.gt_sd.to_numpy(), GT_BINS), bin_of(j.sd.to_numpy(), PRED_BINS)
        md += [f"## {name}", "", "| GT state | n | " + " | ".join(b[2] for b in PRED_BINS) +
               " | signed-dist err p10 / p50 / p90 | |err| p50 | tip pos err p50 / p90 |",
               "|:--|--:|" + "--:|" * len(PRED_BINS) + "--:|--:|--:|"]
        hist["variants"][name] = {}
        for i, (_, _, lab) in enumerate(GT_BINS):
            s = gb == i
            n = int(s.sum())
            if not n:
                continue
            pct = [100 * (pb[s] == k).mean() for k in range(len(PRED_BINS))]
            e = (j.sd - j.gt_sd).to_numpy()[s]
            q = np.percentile(e, [10, 50, 90])
            tip = (j.tip_err.to_numpy()[s] if "tip_err" in j else np.array([np.nan]))
            tip = tip[np.isfinite(tip)]
            tq = np.percentile(tip, [50, 90]) if len(tip) else [np.nan, np.nan]
            md.append(f"| {lab} | {n} | " + " | ".join(f"{p:.1f}" for p in pct) +
                      f" | {q[0]:+.1f} / {q[1]:+.1f} / {q[2]:+.1f} | {np.median(np.abs(e)):.1f} | "
                      f"{tq[0]:.1f} / {tq[1]:.1f} |")
            hist["variants"][name][lab] = {"n": n, "counts": np.histogram(np.clip(j.sd.to_numpy()[s], -30, 100),
                                                                          HIST_EDGES[1:-1])[0].tolist(),
                                           "pct": pct, "err_q": q.tolist()}
            summary.append({"variant": name, "gt": lab, "n": n, **{PRED_BINS[k][2]: pct[k] for k in range(len(pct))},
                            "err_p50": q[1], "abs_err_p50": float(np.median(np.abs(e)))})
        # per-part, touching only
        t = j[j.gt_sd <= 2]
        if len(t):
            md += ["", "Touching parts only, by part (row %: pen<−2 / touch ±2 / hover 2–10 / off >10; "
                   "p50 signed dist; tip err p50):", "",
                   "| part | n | pen<−2 | touch ±2 | hover 2–10 | off >10 | sd p50 | tip err p50 |",
                   "|:--|--:|--:|--:|--:|--:|--:|--:|"]
            for p in PARTS:
                u = t[t.part == p]
                if not len(u):
                    continue
                sd = u.sd.to_numpy()
                tip = u.tip_err.to_numpy() if "tip_err" in u else np.array([np.nan])
                md.append(f"| {p} | {len(u)} | {100*(sd<-2).mean():.1f} | {100*((sd>=-2)&(sd<=2)).mean():.1f} | "
                          f"{100*((sd>2)&(sd<=10)).mean():.1f} | {100*(sd>10).mean():.1f} | {np.median(sd):+.1f} | "
                          f"{np.nanmedian(tip):.1f} |")
        md.append("")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    json.dump(hist, open(out.with_suffix(".json"), "w"))
    pd.DataFrame(summary).to_csv(out.with_suffix(".csv"), index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
