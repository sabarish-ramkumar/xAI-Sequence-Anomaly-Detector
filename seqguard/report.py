"""Formats detection results for the terminal and saves score plots."""
from __future__ import annotations

import time

import numpy as np

from .explain import Explanation
from .schema import Window
from .translate import classify_reason, plain_english_summary


def _clock(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def format_report(w: Window, ex: Explanation, flagged: bool, threshold: float = 0.5, k: int = 3) -> str:
    lines = [f"Window  : {w.entity}  {_clock(w.start_ts)} -> {_clock(w.end_ts)}  ({len(w.events)} events)",
             f"Status  : {'FLAGGED' if flagged else 'NOT FLAGGED'}   score {ex.score:.2f}  threshold {threshold:.2f}"]
    if flagged:
        lines.append(f"Reason  : {classify_reason(ex.top)}")
        lines.append("Top contributors:")
        for i, f in enumerate(ex.top, 1):
            lines.append(f"  {i}. {f.label:<42s} {f.contribution:+.2f}   (step {f.step + 1})")
        lines.append("Analyst summary:")
        lines.append("  " + plain_english_summary(ex.top, ex.score))
    return "\n".join(lines)


def step_table(w: Window, ex: Explanation) -> str:
    """Per-step breakdown that makes the xAI output auditable in a demo."""
    top = {f.step for f in ex.top}
    rows = ["  step  event                                         surprisal  expected  contribution"]
    for f in ex.features:
        mark = "*" if f.step in top else " "
        rows.append(f" {mark}{f.step + 1:>3}   {f.label[:44]:<44s} {f.surprisal:>8.2f}  {f.expected:>8.2f}  {f.contribution:>+10.3f}")
    rows.append(f"  baseline score {ex.baseline:.3f} + sum(contributions) {ex.phi.sum():+.3f} = score {ex.score:.3f}")
    return "\n".join(rows)


def plot_session(path, name: str, windows: list[Window], scores: np.ndarray, peak: int, ex: Explanation,
                 threshold: float = 0.5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), gridspec_kw={"height_ratios": [1, 1.25]})
    a1.plot(range(1, len(scores) + 1), scores, marker="o", color="#1f4e79")
    a1.axhline(threshold, color="#c0392b", ls="--", lw=1)
    a1.scatter([peak + 1], [scores[peak]], s=90, color="#c0392b", zorder=3)
    a1.set(ylim=(0, 1.05), xlabel="window index (stride 1)", ylabel="anomaly score", title=f"{name}: score per window")
    labels = [f.label for f in ex.features]
    top = {f.step for f in ex.top}
    colors = ["#c0392b" if i in top else "#9aa5b1" for i in range(len(labels))]
    a2.barh(range(len(labels)), [f.contribution for f in ex.features], color=colors)
    a2.set_yticks(range(len(labels)), [f"{i + 1}. {l}" for i, l in enumerate(labels)], fontsize=8)
    a2.invert_yaxis()
    a2.axvline(0, color="k", lw=0.6)
    a2.set(xlabel="Shapley contribution to anomaly score", title=f"peak window #{peak + 1}: top-3 in red")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
