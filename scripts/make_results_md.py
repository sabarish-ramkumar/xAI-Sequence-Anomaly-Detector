"""Render results/*/metrics.json into docs/RESULTS.md so the report never drifts from the code."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sy = json.loads((root / "results/sysmon/metrics.json").read_text())
hd = json.loads((root / "results/hdfs/metrics.json").read_text())


def pct(x): return f"{100 * x:.1f}%"


def det_row(name, d):
    return f"| {name} | {d['windows']:,} | {d['positives']:,} | {d['auroc']:.3f} | {d['auprc']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {pct(d['fpr'])} |"


L = ["# Results", "", "All numbers are produced by `python -m seqguard evaluate` and rendered by `scripts/make_results_md.py`.",
     "Thresholds are fixed on held-out *normal* data only (99.5th percentile of validation scores); no attack labels are used for training or tuning.", "",
     "## Detection quality (decision threshold 0.5)", "",
     "| dataset | windows | positives | AUROC | AUPRC | precision | recall | F1 | FPR |", "|---|---|---|---|---|---|---|---|---|",
     det_row("Sysmon-style telemetry (test days)", sy["detection"]), det_row("HDFS (LogHub sample, chronological test split)", hd["detection"]), ""]
c = sy["campaigns"]
L += ["## Sysmon: intrusion campaigns", "", f"{c['detected']} of {c['campaigns']} injected campaigns raised at least one alert. "
      f"The first alert arrived after a median of {pct(c['median_fraction_of_chain_seen_at_alert'])} of the campaign's events had been observed.", "",
      "| campaign family | detected |", "|---|---|"]
L += [f"| {k} | {v['detected']}/{v['total']} |" for k, v in c["by_family"].items()]
fp = sy["false_positive_breakdown"]
L += ["", f"False positives: {fp['false_positives']} windows, of which {fp['containing_never_seen_event']} contain software that never appeared during training "
      f"(8 users adopt a new application in the test period) and {fp['other']} come from rare but legitimate transitions.", ""]
a = sy["adaptation"]
L += ["## Analyst feedback loop", "",
      f"Absorbing {a['benign_sessions_absorbed']} analyst-confirmed benign sessions from the first two test days and re-scoring the last two "
      f"(threshold unchanged): false positives {a['before']['fp']} -> {a['after']['fp']}, precision {a['before']['precision']:.3f} -> {a['after']['precision']:.3f}, "
      f"recall {a['before']['recall']:.3f} -> {a['after']['recall']:.3f}.", ""]
ce = hd["ceiling"]
L += ["## HDFS: what the sample allows", "",
      f"{ce['identical_to_a_normal_training_session']} of {ce['anomalous_sessions']} anomalous test sessions ({pct(ce['fraction_indistinguishable'])}) have exactly the same event sequence as a normal training session, "
      f"because the 100k-line sample keeps only the first lines of most blocks. No sequence model can separate those, so attainable recall is {pct(ce['max_attainable_recall'])}. "
      f"On the remaining anomalies the detector's recall is {pct(ce['recall_on_distinguishable'])}.", ""]
for name, d in (("Sysmon", sy), ("HDFS", hd)):
    e = d["explanation"]
    L += [f"## Explanation faithfulness: {name}", "",
          f"Computed on {e['windows_explained']} flagged windows. Neutralising the top-3 steps lowers the score by {e['drop_top']:.2f} on average vs {e['drop_random']:.2f} for three random steps; "
          f"the decision flips in {pct(e['flip_top'])} vs {pct(e['flip_random'])} of windows. Largest violation of the Shapley efficiency property: {e['max_efficiency_gap']:.1e}."]
    if "hit_top" in e:
        L.append(f" Share of top-3 steps that are real intrusion events: {pct(e['hit_top'])} vs {pct(e['hit_random'])} for random steps (normalised by min(3, intrusion events in window)).")
    L.append("")
for name, d in (("Sysmon", sy), ("HDFS", hd)):
    L += [f"## Ablation: {name}", "", "| variant | AUROC | AUPRC | precision | recall | F1 | FPR |", "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['auroc']:.3f} | {v['auprc']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {pct(v['fpr'])} |" for k, v in d["ablation"].items()]
    L.append("")
(root / "docs/RESULTS.md").write_text("\n".join(L))
print("\n".join(L))
