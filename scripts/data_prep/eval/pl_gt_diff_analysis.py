"""pl_gt_diff_analysis.py — PL(pseudo-label) vs GT difference, 4 experiments.

Follow-up to filter_domain_analysis.py.  Reuses _full_{tag}.json which already
contains, per frame:
  kp   : predicted 9 keypoints (8 corners + centroid) in full-res px (or None)
  gt8  : GT projected_cuboid first 8 corners (px)
  mean_match_px : order-free Hungarian mean reproj error (px)
  filters : {fullkp,diag,ratio,ransac_loo,combo} pass booleans

Error metric = order-free Hungarian assignment between predicted 8 corners and
GT 8 corners (convention/dims-agnostic, absorbs W/D swap & corner-order).

Experiment 1 — distribution of GT-reproj error of PASSED PL, per domain x filter
               (box + strip + median/IQR table).
Experiment 2 — separability: passed-error vs rejected-error per filter/domain,
               with median gap and AUC(error -> reject).
Experiment 3 — per-keypoint (corner 0..7 + centroid 8) GT error of passed PL,
               heatmap front(0-3)/back(4-7)/centroid(8), per filter x domain.
Experiment 4 — PL(pred) + GT cuboid overlay on real images, passed samples,
               good vs misleading-pass contrast.

Inference-free: pure post-processing of _full_{tag}.json.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

DOMAINS = ["indoor", "outside", "night"]
FILTERS = ["fullkp", "diag", "ratio", "ransac_loo", "combo"]
DOM_COLOR = {"indoor": "#4C72B0", "outside": "#DD8452", "night": "#55A868"}
FILT_COLOR = {"fullkp": "#8172B3", "diag": "#C44E52", "ratio": "#937860",
              "ransac_loo": "#DA8BC3", "combo": "#000000"}
GROSS_PX = 20.0   # catastrophic PL threshold (from prior screening)


# ── order-free per-corner assignment ─────────────────────────────────
def assign_corners(kp, gt8):
    """Hungarian assign predicted 8 corners to GT 8 corners.

    Returns (mean_px, n_match, matched_dists[8] with nan for unmatched-pred,
             pred_idx_for_each_assigned, gt_idx_for_each_assigned).
    matched_dists is indexed by PREDICTED corner slot (0..7); nan if that
    predicted corner is missing or unmatched.
    """
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    dists = np.full(8, np.nan)
    if valid.sum() < 6:
        return float("inf"), int(valid.sum()), dists
    idx_valid = np.where(valid)[0]
    P = pred8[valid]
    G = np.asarray(gt8, float)
    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        dists[idx_valid[r]] = cost[r, c]
    return float(cost[ri, ci].mean()), len(ri), dists


def centroid_err(kp, gt8):
    """Predicted centroid(8) vs GT cuboid centroid (mean of gt8)."""
    if kp[8] is None:
        return np.nan
    gc = np.asarray(gt8, float).mean(axis=0)
    return float(np.linalg.norm(np.asarray(kp[8], float) - gc))


def auc_error_reject(passed_err, rejected_err):
    """AUC: how well does 'higher error' predict 'rejected'.

    label=1 for rejected, score=error. AUC>0.5 means filter rejects high-error.
    """
    e = np.concatenate([passed_err, rejected_err])
    y = np.concatenate([np.zeros(len(passed_err)), np.ones(len(rejected_err))])
    if len(passed_err) == 0 or len(rejected_err) == 0:
        return float("nan")
    order = np.argsort(e)
    yo = y[order]
    # rank-sum AUC (prob a random rejected has higher error than random passed)
    n_pos = yo.sum()
    n_neg = len(yo) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(e)) + 1
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def load(tag):
    fp = os.path.join(ROOT, "data", "pallet", "eval_results",
                      "filter_domain_analysis", f"_full_{tag}.json")
    return json.load(open(fp))


# ════════════════════════════════════════════════════════════════════
def collect(data):
    """Build per (domain,filter) lists of passed errors + per-kp dists +
    rejected errors.  'detectable' = >=6 corners so mean_match is finite."""
    rec = {}
    for dom in DOMAINS:
        rows = data[dom]
        det = [r for r in rows if r["mean_match_px"] is not None]
        rec[dom] = {"detectable": det}
        for filt in FILTERS:
            passed, passed_kp, rejected = [], [], []
            for r in det:
                e = r["mean_match_px"]
                if r["filters"][filt]:
                    passed.append(e)
                    _, _, d = assign_corners(r["kp"], r["gt8"])
                    ce = centroid_err(r["kp"], r["gt8"])
                    passed_kp.append(np.concatenate([d, [ce]]))  # len 9
                else:
                    rejected.append(e)
            rec[dom][filt] = {
                "passed": np.array(passed, float),
                "rejected": np.array(rejected, float),
                "passed_kp": (np.array(passed_kp, float)
                              if passed_kp else np.zeros((0, 9))),
            }
    return rec


def stats_line(arr):
    if len(arr) == 0:
        return dict(n=0, med=np.nan, q1=np.nan, q3=np.nan, mean=np.nan,
                    gross=0, grossp=np.nan)
    return dict(n=len(arr), med=np.median(arr), q1=np.percentile(arr, 25),
                q3=np.percentile(arr, 75), mean=arr.mean(),
                gross=int((arr > GROSS_PX).sum()),
                grossp=float((arr > GROSS_PX).mean()))


# ── Experiment 1 ─────────────────────────────────────────────────────
def exp1(rec, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=True)
    table = {}
    for ax, dom in zip(axes, DOMAINS):
        data_box, labels, ns = [], [], []
        for filt in FILTERS:
            a = rec[dom][filt]["passed"]
            data_box.append(a if len(a) else np.array([np.nan]))
            labels.append(filt)
            ns.append(len(a))
            table.setdefault(dom, {})[filt] = stats_line(a)
        bp = ax.boxplot(data_box, labels=labels, showfliers=False,
                        widths=0.6, patch_artist=True, medianprops=dict(
                            color="black", lw=2))
        for patch, filt in zip(bp["boxes"], FILTERS):
            patch.set_facecolor(FILT_COLOR[filt]); patch.set_alpha(0.45)
        for xi, (filt, a) in enumerate(zip(FILTERS, data_box), start=1):
            arr = rec[dom][filt]["passed"]
            if len(arr):
                jit = np.random.RandomState(0).uniform(-0.13, 0.13, len(arr))
                ax.scatter(xi + jit, arr, s=14, color=FILT_COLOR[filt],
                           edgecolor="white", lw=0.3, alpha=0.7, zorder=3)
                ax.text(xi, -3.0, f"n={len(arr)}", ha="center", va="top",
                        fontsize=8, color="#333")
                ax.text(xi, np.median(arr), f" {np.median(arr):.1f}",
                        ha="left", va="center", fontsize=8, fontweight="bold")
        ax.axhline(GROSS_PX, ls="--", color="red", lw=1, alpha=0.6)
        ax.text(5.4, GROSS_PX, "gross 20px", color="red", fontsize=8,
                va="bottom", ha="right")
        ax.axhline(10, ls=":", color="green", lw=1, alpha=0.6)
        det = rec[dom]["detectable"]
        ax.set_title(f"{dom}  (detectable={len(det)})", fontsize=12,
                     fontweight="bold", color=DOM_COLOR[dom])
        ax.set_xlabel("filter"); ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(-6, 60)
    axes[0].set_ylabel("PASSED PL  GT order-free reproj error (px)")
    fig.suptitle("Exp1 — GT reproj error distribution of filter-PASSED "
                 "pseudo-labels  (model: dope_cropaug_ft_s2)",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, "lower = passed PL closer to GT. dashed red=gross(20px) "
             "catastrophic, dotted green=good(10px). boxes hide outliers; dots="
             "all passed frames. ratio passes <6kp frames excluded (no err).",
             ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    p = os.path.join(outdir, "exp1_passed_error_dist.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, table


# ── Experiment 2 ─────────────────────────────────────────────────────
def exp2(rec, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), sharey=True)
    sep = {}
    for ax, dom in zip(axes, DOMAINS):
        xpos, xticklab = [], []
        for fi, filt in enumerate(FILTERS):
            pa = rec[dom][filt]["passed"]
            re = rec[dom][filt]["rejected"]
            x0 = fi * 2.6
            for arr, off, lab, col in [(pa, 0, "pass", "#2E7D32"),
                                       (re, 1.0, "rej", "#C62828")]:
                if len(arr):
                    bp = ax.boxplot([arr], positions=[x0 + off], widths=0.8,
                                    showfliers=False, patch_artist=True,
                                    medianprops=dict(color="black", lw=2))
                    bp["boxes"][0].set_facecolor(col)
                    bp["boxes"][0].set_alpha(0.4)
                    jit = np.random.RandomState(1).uniform(-0.18, 0.18, len(arr))
                    ax.scatter(x0 + off + jit, np.clip(arr, 0, 80), s=8,
                               color=col, alpha=0.45, zorder=3)
                xpos.append(x0 + off); xticklab.append(lab)
            mp = np.median(pa) if len(pa) else np.nan
            mr = np.median(re) if len(re) else np.nan
            gap = mr - mp if np.isfinite(mp) and np.isfinite(mr) else np.nan
            auc = auc_error_reject(pa, re)
            sep.setdefault(dom, {})[filt] = dict(
                med_pass=float(mp) if np.isfinite(mp) else None,
                med_rej=float(mr) if np.isfinite(mr) else None,
                gap=float(gap) if np.isfinite(gap) else None,
                auc=float(auc) if np.isfinite(auc) else None,
                n_pass=int(len(pa)), n_rej=int(len(re)))
            txt = (f"{filt}\nΔmed={gap:.1f}" if np.isfinite(gap) else f"{filt}\nΔ=NA")
            txt += f"\nAUC={auc:.2f}" if np.isfinite(auc) else "\nAUC=NA"
            ax.text(x0 + 0.5, 78, txt, ha="center", va="top", fontsize=8.5,
                    fontweight="bold")
        ax.set_xticks(xpos); ax.set_xticklabels(xticklab, fontsize=7,
                                                rotation=0)
        ax.axhline(GROSS_PX, ls="--", color="red", lw=1, alpha=0.5)
        ax.set_title(f"{dom}", fontsize=12, fontweight="bold",
                     color=DOM_COLOR[dom])
        ax.grid(axis="y", alpha=0.25); ax.set_ylim(0, 82)
    axes[0].set_ylabel("GT order-free reproj error (px, clipped 80)")
    fig.suptitle("Exp2 — separability: PASSED (green) vs REJECTED (red) PL "
                 "error per filter.  Δmed=rej-pass, AUC=P(rej>pass)",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.005, "filter does work when Δmed>0 (passed cleaner) and "
             "AUC>0.5 (error predicts rejection). combo/ransac_loo tiny n.",
             ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = os.path.join(outdir, "exp2_pass_vs_reject_separability.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, sep


# ── Experiment 3 ─────────────────────────────────────────────────────
KP_LABELS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "ctr8"]


def exp3(rec, outdir):
    # heatmap: rows = (domain,filter) with passes, cols = 9 keypoints; median err
    rows, rowlabels = [], []
    for dom in DOMAINS:
        for filt in FILTERS:
            pk = rec[dom][filt]["passed_kp"]
            if len(pk) == 0:
                continue
            med = np.nanmedian(pk, axis=0)
            rows.append(med)
            rowlabels.append(f"{dom}/{filt} (n={len(pk)})")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(rows) + 2)))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=np.nanpercentile(M, 95))
    ax.set_xticks(range(9)); ax.set_xticklabels(KP_LABELS)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rowlabels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v > np.nanpercentile(M, 60) else "black")
    # front/back/centroid group separators
    ax.axvline(3.5, color="navy", lw=1.5); ax.axvline(7.5, color="navy", lw=1.5)
    ax.text(1.5, -1.15, "FRONT 0-3", ha="center", fontsize=9, color="navy",
            fontweight="bold")
    ax.text(5.5, -1.15, "BACK 4-7", ha="center", fontsize=9, color="navy",
            fontweight="bold")
    ax.text(8, -1.15, "CTR", ha="center", fontsize=9, color="navy",
            fontweight="bold")
    ax.set_ylim(len(rows) - 0.5, -1.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("median GT corner error (px)")
    ax.set_title("Exp3 — per-keypoint median GT error of PASSED PL "
                 "(order-free Hungarian per-corner; ctr8 vs GT cuboid center)",
                 fontsize=12, fontweight="bold", pad=24)
    fig.tight_layout()
    p = os.path.join(outdir, "exp3_per_keypoint_heatmap.png")
    fig.savefig(p, dpi=130); plt.close(fig)

    # grouped bar: front/back/centroid mean-of-median per domain (combo/loo skip)
    groups = {"front(0-3)": slice(0, 4), "back(4-7)": slice(4, 8),
              "ctr(8)": slice(8, 9)}
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        plot_filts = [f for f in FILTERS
                      if len(rec[dom][f]["passed_kp"]) > 0]
        x = np.arange(len(groups)); w = 0.8 / max(1, len(plot_filts))
        for fi, filt in enumerate(plot_filts):
            pk = rec[dom][filt]["passed_kp"]
            vals = [np.nanmedian(pk[:, sl]) for sl in groups.values()]
            ax.bar(x + fi * w, vals, w, label=f"{filt}({len(pk)})",
                   color=FILT_COLOR[filt], alpha=0.8, edgecolor="white")
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(list(groups))
        ax.set_title(dom, fontweight="bold", color=DOM_COLOR[dom])
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("median GT error (px)")
    fig2.suptitle("Exp3b — front/back/centroid GT error of PASSED PL by filter",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    p2 = os.path.join(outdir, "exp3b_front_back_centroid_bars.png")
    fig2.savefig(p2, dpi=130); plt.close(fig2)

    # build table
    tbl = {}
    for dom in DOMAINS:
        tbl[dom] = {}
        for filt in FILTERS:
            pk = rec[dom][filt]["passed_kp"]
            if len(pk) == 0:
                continue
            tbl[dom][filt] = {
                "n": len(pk),
                "front_med": float(np.nanmedian(pk[:, :4])),
                "back_med": float(np.nanmedian(pk[:, 4:8])),
                "ctr_med": float(np.nanmedian(pk[:, 8])),
                "per_kp_med": [None if not np.isfinite(v) else round(float(v), 1)
                               for v in np.nanmedian(pk, axis=0)]}
    return p, p2, tbl


# ── Experiment 4 ─────────────────────────────────────────────────────
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),       # front face
         (4, 5), (5, 6), (6, 7), (7, 4),       # back face
         (0, 4), (1, 5), (2, 6), (3, 7)]       # connectors


def draw_cuboid(img, pts8, color, thick=2, label_corners=False):
    pts = [None if p is None or (isinstance(p, float) and np.isnan(p))
           else (int(round(p[0])), int(round(p[1]))) for p in pts8]
    for a, b in EDGES:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], color, thick, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if p is not None:
            cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)
            if label_corners:
                cv2.putText(img, str(i), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def exp4(data, rec, outdir, per_domain=4):
    """Overlay PL(pred, cyan) + GT(magenta) for passed samples.
    Pick contrast: best-passed (low err) and worst-passed (high err) per domain.
    """
    PRED_C = (255, 255, 0)    # cyan (BGR)
    GT_C = (255, 0, 255)      # magenta
    odir = os.path.join(outdir, "exp4_overlays")
    os.makedirs(odir, exist_ok=True)
    saved = []
    for dom in DOMAINS:
        # candidate passed frames: prefer 'diag' (primary filter); fallback any
        cand = []
        for r in data[dom]:
            if r["mean_match_px"] is None:
                continue
            passing = [f for f in FILTERS if r["filters"][f]]
            if not passing:
                continue
            cand.append((r, passing))
        if not cand:
            continue
        cand.sort(key=lambda rp: rp[0]["mean_match_px"])
        # pick best 2 and worst 2 among passed
        picks = cand[:2] + cand[-2:]
        picks = picks[:per_domain]
        for r, passing in picks:
            img = cv2.imread(r["img"])
            if img is None:
                continue
            draw_cuboid(img, r["gt8"], GT_C, 2, label_corners=True)
            pred8 = [r["kp"][i] for i in range(8)]
            draw_cuboid(img, pred8, PRED_C, 2)
            if r["kp"][8] is not None:
                c = (int(r["kp"][8][0]), int(r["kp"][8][1]))
                cv2.drawMarker(img, c, PRED_C, cv2.MARKER_STAR, 12, 2)
            # badge panel
            h, w = img.shape[:2]
            pan = img.copy()
            cv2.rectangle(pan, (0, 0), (min(440, w), 96), (0, 0, 0), -1)
            img = cv2.addWeighted(pan, 0.55, img, 0.45, 0)
            err = r["mean_match_px"]
            verdict = "GOOD" if err < 10 else ("OK" if err < 20 else "MISLEAD")
            vcol = ((0, 255, 0) if err < 10 else
                    (0, 200, 255) if err < 20 else (0, 0, 255))
            cv2.putText(img, f"{dom}  err={err:.1f}px  [{verdict}]", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, vcol, 2, cv2.LINE_AA)
            cv2.putText(img, "pass: " + ",".join(passing), (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(img, "GT=magenta  PL/pred=cyan", (8, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(img, f"n_det={r['n_detected']}", (8, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                        cv2.LINE_AA)
            tag = "best" if err < 12 else "worst"
            op = os.path.join(odir, f"{dom}_{tag}_{r['frame']}_e{err:.0f}.png")
            cv2.imwrite(op, img)
            saved.append(op)
    # contact sheet
    if saved:
        thumbs = []
        for p in saved:
            im = cv2.imread(p)
            im = cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1])))
            thumbs.append(im)
        hmax = max(t.shape[0] for t in thumbs)
        thumbs = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                  cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in thumbs]
        rowsz = 3
        sheet_rows = []
        for i in range(0, len(thumbs), rowsz):
            chunk = thumbs[i:i + rowsz]
            while len(chunk) < rowsz:
                chunk.append(np.full_like(thumbs[0], 40))
            sheet_rows.append(np.hstack(chunk))
        sheet = np.vstack(sheet_rows)
        sp = os.path.join(outdir, "exp4_overlay_contact_sheet.png")
        cv2.imwrite(sp, sheet)
        return sp, saved
    return None, saved


# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "pl_gt_diff"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(0)

    data = load(args.tag)
    rec = collect(data)

    p1, t1 = exp1(rec, args.output_dir)
    p2, sep = exp2(rec, args.output_dir)
    p3, p3b, t3 = exp3(rec, args.output_dir)
    p4, saved = exp4(data, rec, args.output_dir)

    # console summary tables
    print("\n" + "=" * 92)
    print("EXP1+2  PASSED PL GT-reproj median(px) | IQR | gross% | "
          "Δmed(rej-pass) | AUC(err->rej)")
    print("=" * 92)
    hdr = (f"{'dom':<8}{'filter':<12}{'n':>4}{'med':>7}{'q1':>6}{'q3':>6}"
           f"{'gross%':>8}{'Δmed':>8}{'AUC':>6}")
    print(hdr); print("-" * len(hdr))
    for dom in DOMAINS:
        for filt in FILTERS:
            s = t1[dom][filt]
            sp = sep[dom][filt]
            if s["n"] == 0:
                print(f"{dom:<8}{filt:<12}{0:>4}{'  -- no pass --':>27}")
                continue
            gp = s["grossp"] * 100
            dm = sp["gap"]; au = sp["auc"]
            print(f"{dom:<8}{filt:<12}{s['n']:>4}{s['med']:>7.1f}"
                  f"{s['q1']:>6.1f}{s['q3']:>6.1f}{gp:>7.0f}%"
                  f"{(f'{dm:>8.1f}' if dm is not None else '     NA')}"
                  f"{(f'{au:>6.2f}' if au is not None else '    NA')}")
    print("\n" + "=" * 70)
    print("EXP3  PASSED PL per-group median GT err (px): front / back / ctr")
    print("=" * 70)
    print(f"{'dom':<8}{'filter':<12}{'n':>4}{'front':>8}{'back':>8}{'ctr':>8}")
    print("-" * 48)
    for dom in DOMAINS:
        for filt in FILTERS:
            if filt not in t3[dom]:
                continue
            e = t3[dom][filt]
            print(f"{dom:<8}{filt:<12}{e['n']:>4}{e['front_med']:>8.1f}"
                  f"{e['back_med']:>8.1f}{e['ctr_med']:>8.1f}")

    out = {"tag": args.tag, "gross_px": GROSS_PX,
           "exp1_stats": {d: {f: {k: (None if (isinstance(v, float) and
                          not np.isfinite(v)) else v) for k, v in s.items()}
                              for f, s in t1[d].items()} for d in DOMAINS},
           "exp2_separability": sep, "exp3_perkp": t3,
           "figures": {"exp1": p1, "exp2": p2, "exp3": p3, "exp3b": p3b,
                       "exp4_sheet": p4, "exp4_overlays": saved}}
    jp = os.path.join(args.output_dir, "pl_gt_diff_results.json")
    json.dump(out, open(jp, "w"), indent=2, default=str)
    print(f"\n[save] {p1}\n[save] {p2}\n[save] {p3}\n[save] {p3b}")
    print(f"[save] {p4}\n[save] {jp}")
    print(f"[overlays] {len(saved)} in {os.path.dirname(p4) if p4 else ''}")


if __name__ == "__main__":
    main()
