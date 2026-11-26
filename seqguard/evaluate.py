"""Metrics: detection quality, campaign-level detection, explanation faithfulness, ablations."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .explain import explain_window
from .model import SequenceAnomalyDetector
from .pipeline import Prepared, build_detector
from .windows import build_windows


def window_metrics(y: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> dict:
    pred = scores >= thr
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn = int(((~pred) & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    return {"windows": int(len(y)), "positives": int(y.sum()), "auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores)), "precision": float(p), "recall": float(r),
            "f1": float(f), "fpr": fp / max(fp + tn, 1), "tp": int((pred & (y == 1)).sum()), "fp": fp,
            "fn": int(((~pred) & (y == 1)).sum()), "tn": tn}


def campaign_metrics(windows, scores, thr: float = 0.5) -> dict:
    """A campaign counts as detected when any window that touches it is flagged."""
    by_campaign = defaultdict(list)
    for i, w in enumerate(windows):
        for c in {e.campaign for e in w.events if e.campaign}:
            by_campaign[c].append(i)
    rows, depth = {}, []
    for c, idx in by_campaign.items():
        idx.sort(key=lambda i: windows[i].start_ts)
        evs = sorted({id(e): e for i in idx for e in windows[i].events if e.campaign == c}.values(), key=lambda e: e.ts)
        pos = {id(e): k + 1 for k, e in enumerate(evs)}
        hit = next((i for i in idx if scores[i] >= thr), None)
        seen = max((pos[id(e)] for e in windows[hit].events if e.campaign == c), default=0) if hit is not None else None
        rows[c] = {"detected": hit is not None, "events_in_campaign": len(evs), "events_seen_at_first_alert": seen}
        if seen:
            depth.append(seen / len(evs))
    fam = defaultdict(lambda: [0, 0])
    for c, r in rows.items():
        fam[c.split("#")[0]][1] += 1
        fam[c.split("#")[0]][0] += int(r["detected"])
    return {"campaigns": len(rows), "detected": sum(r["detected"] for r in rows.values()),
            "median_fraction_of_chain_seen_at_alert": float(np.median(depth)) if depth else None,
            "by_family": {k: {"detected": v[0], "total": v[1]} for k, v in sorted(fam.items())}}


def explanation_metrics(det: SequenceAnomalyDetector, windows, scores, k: int = 3, thr: float = 0.5,
                        max_windows: int = 150, seed: int = 0, **explainer_kw) -> dict:
    """Measure faithfulness of top-k explanations via deletion test and intrusion hit rate."""
    rng = np.random.default_rng(seed)
    idx = [i for i, w in enumerate(windows) if w.label and scores[i] >= thr]
    if len(idx) > max_windows:
        idx = sorted(rng.choice(idx, max_windows, replace=False))
    out = defaultdict(list)
    for i in idx:
        w = windows[i]
        ex = explain_window(det, w, k=k, **explainer_kw)
        e = det.step_stats(w).excess
        top = [f.step for f in ex.top]
        masked = e.copy(); masked[top] = 0
        out["drop_top"].append(ex.score - float(det.calibrate(det.pool(masked))[0]))
        out["flip_top"].append(float(det.calibrate(det.pool(masked))[0]) < thr)
        rnd_drop, rnd_flip, rnd_hit = [], [], []
        is_attack = np.array([bool(w.step_event(s) and w.step_event(s).label) for s in range(len(e))])
        for _ in range(20):
            r = rng.choice(len(e), min(k, len(e)), replace=False)
            m2 = e.copy(); m2[r] = 0
            sc = float(det.calibrate(det.pool(m2))[0])
            rnd_drop.append(ex.score - sc); rnd_flip.append(sc < thr)
            rnd_hit.append(is_attack[r].sum())
        denom = max(min(k, int(is_attack.sum())), 1)
        out["drop_random"].append(np.mean(rnd_drop)); out["flip_random"].append(np.mean(rnd_flip))
        out["hit_top"].append(is_attack[top].sum() / denom)
        out["hit_random"].append(np.mean(rnd_hit) / denom)
        out["gap"].append(ex.efficiency_gap)
    m = {k_: float(np.mean(v)) for k_, v in out.items()}
    m["windows_explained"] = len(idx)
    m["max_efficiency_gap"] = float(np.max(out["gap"])) if out["gap"] else 0.0
    return m


def hdfs_ceiling(train_sequences, test_windows, scores, thr: float = 0.5) -> dict:
    """Check how many anomalous HDFS sessions are identical to normal training sessions."""
    seen = {tuple(e.token for e in s) for s in train_sequences}
    anom = [i for i, w in enumerate(test_windows) if w.label]
    twin = [i for i in anom if tuple(e.token for e in test_windows[i].events) in seen]
    distinct = [i for i in anom if i not in set(twin)]
    return {"anomalous_sessions": len(anom), "identical_to_a_normal_training_session": len(twin),
            "fraction_indistinguishable": len(twin) / max(len(anom), 1),
            "recall_on_distinguishable": float(np.mean([scores[i] >= thr for i in distinct])) if distinct else None,
            "max_attainable_recall": len(distinct) / max(len(anom), 1)}


ABLATIONS = {
    "unigram, mean pooling (no order)": {"order": 1, "pooling": "mean"},
    "unigram, LSE pooling": {"order": 1},
    "bigram, mean pooling": {"pooling": "mean"},
    "bigram, LSE pooling (default)": {},
}


def ablation(cfg: dict, data: Prepared) -> dict:
    rows = {}
    y = np.array([w.label for w in data.test_windows])
    for name, over in ABLATIONS.items():
        mc = {**cfg["model"], **over}
        det = build_detector(mc).fit(data.train_sequences, data.val_windows)
        m = window_metrics(y, det.score_windows(data.test_windows))
        rows[name] = {k: round(m[k], 4) for k in ("auroc", "auprc", "precision", "recall", "f1", "fpr")}
    return rows


def adaptation_experiment(det: SequenceAnomalyDetector, data: Prepared) -> dict:
    """Absorb benign sessions from first two test days, then re-score last two days."""
    ev = data.extra["test_events"]
    mid = min(e.ts for e in ev) + 2 * 86400
    first, second = [e for e in ev if e.ts < mid], [e for e in ev if e.ts >= mid]
    from .windows import group_by_entity, split_sessions
    gap = data.extra["idle_gap"]
    benign = [s for st in group_by_entity(first).values() for s in split_sessions(st, gap) if not any(e.label for e in s)]
    w = data.extra["window"]
    wins = build_windows(second, w["size"], w["stride"], gap)
    y = np.array([x.label for x in wins])
    before = window_metrics(y, det.score_windows(wins))
    tuned = SequenceAnomalyDetector.from_dict(copy.deepcopy(det.to_dict())).update(benign)
    after = window_metrics(y, tuned.score_windows(wins))
    keep = ("precision", "recall", "f1", "fpr", "fp")
    return {"benign_sessions_absorbed": len(benign), "windows_scored": len(wins),
            "before": {k: round(before[k], 4) for k in keep}, "after": {k: round(after[k], 4) for k in keep}}


def false_positive_breakdown(det, windows, scores, thr: float = 0.5) -> dict:
    """Which false positives come from software never seen during training?"""
    fps = [w for w, s in zip(windows, scores) if s >= thr and not w.label]
    novel = sum(1 for w in fps if det.step_stats(w).novel.any())
    return {"false_positives": len(fps), "containing_never_seen_event": novel,
            "other": len(fps) - novel}


def score_histogram(path, scores, labels, thr: float = 0.5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(scores[labels == 0], bins, alpha=0.75, label="normal windows", color="#1f4e79")
    ax.hist(scores[labels == 1], bins, alpha=0.75, label="windows containing an intrusion", color="#c0392b")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set(yscale="log", xlabel="anomaly score", ylabel="windows (log)", title="Score distribution on the test split")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
