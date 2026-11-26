"""Command line entry point:  python -m seqguard <command>"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import evaluate as ev
from .explain import explain_window
from .loaders import load_sysmon_csv
from .model import SequenceAnomalyDetector
from .pipeline import load_config, train
from .report import format_report, plot_session, step_table
from .windows import build_windows

ROOT = Path(__file__).resolve().parent.parent


def episodes(windows, scores, thr):
    """Collapse runs of consecutive flagged windows of one session into a single alert (its peak window)."""
    out, run = [], []
    for i, (w, s) in enumerate(zip(windows, scores)):
        key = w.wid.rsplit(":", 1)[0]
        if s >= thr and (not run or windows[run[-1]].wid.rsplit(":", 1)[0] == key):
            run.append(i)
        else:
            if run:
                out.append(max(run, key=lambda j: scores[j]))
            run = [i] if s >= thr else []
    if run:
        out.append(max(run, key=lambda j: scores[j]))
    return out


def cmd_generate(a):
    subprocess.run([sys.executable, "scripts/generate_sysmon_dataset.py", "--out", a.out, "--seed", str(a.seed)], check=True)


def cmd_train(a):
    cfg = load_config(a.config)
    det, data = train(cfg, ROOT)
    det.save(cfg["output"]["model"])
    print(f"trained on {len(data.train_sequences)} normal sequences, vocabulary {len(det.model.vocab)} tokens")
    print(f"threshold (raw) {det.raw_threshold:.3f}, scale {det.scale:.3f} -> saved {cfg['output']['model']}")


def cmd_evaluate(a):
    cfg = load_config(a.config)
    det, data = train(cfg, ROOT)
    det.save(cfg["output"]["model"])
    out = Path(cfg["output"]["results"]); out.mkdir(parents=True, exist_ok=True)
    wins = data.test_windows
    y = np.array([w.label for w in wins])
    scores = det.score_windows(wins)
    res = {"dataset": cfg["dataset"], "train_sequences": len(data.train_sequences), "detection": ev.window_metrics(y, scores)}
    if cfg["dataset"] == "sysmon":
        res["campaigns"] = ev.campaign_metrics(wins, scores)
        res["false_positive_breakdown"] = ev.false_positive_breakdown(det, wins, scores)
        res["adaptation"] = ev.adaptation_experiment(det, data)
        ev.score_histogram(out / "score_distribution.png", scores, y)
    else:
        res["ceiling"] = ev.hdfs_ceiling(data.train_sequences, wins, scores)
    res["explanation"] = ev.explanation_metrics(det, wins, scores, k=cfg["explain"]["top_k"], max_windows=a.explain_n,
                                                exact_limit=cfg["explain"]["exact_limit"],
                                                n_permutations=cfg["explain"]["n_permutations"])
    if cfg["dataset"] == "hdfs":  # labels exist per session only, so event-level hit rates are meaningless
        res["explanation"].pop("hit_top", None); res["explanation"].pop("hit_random", None)
    if a.ablation:
        res["ablation"] = ev.ablation(cfg, data)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute() and not path.exists():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    return path


def cmd_detect(a):
    model_path = _resolve_path(a.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Run train first.")
        return
    det = SequenceAnomalyDetector.load(model_path)
    input_path = _resolve_path(a.input)
    try:
        events = load_sysmon_csv(input_path)
    except Exception as e:
        print(f"Error reading input: {e}")
        return
    wins = build_windows(events, a.window, 1)
    if not wins:
        print("0 windows scored (no valid event sequences found).")
        return
    scores = det.score_windows(wins)
    alerts = episodes(wins, scores, 0.5)
    print(f"{len(wins)} windows scored, {sum(scores >= 0.5)} above threshold, {len(alerts)} alert(s)\n")
    for i in alerts:
        ex = explain_window(det, wins[i], k=3)
        print(format_report(wins[i], ex, True), "\n")


def cmd_analyze(a):
    """Score and explain a hand-written sequence, e.g. 'winword.exe -> powershell.exe -> rundll32.exe -> external POST'."""
    from .adhoc import build_window, parse_events
    model_path = Path(a.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        det, _ = train(load_config(ROOT / "configs/sysmon.yaml"), ROOT)
        det.save(model_path)
    else:
        det = SequenceAnomalyDetector.load(model_path)
    evs = parse_events(a.events)
    if not evs:
        print("Error: no events provided.")
        return
    win = build_window(evs)
    score = float(det.score_windows([win])[0])
    ex = explain_window(det, win, k=min(3, len(evs)))
    print(f"\nAd-hoc analysis ({len(evs)} events)")
    print(format_report(win, ex, score >= 0.5))
    print("\nStep breakdown (* = top-3):")
    print(step_table(win, ex))


def run_session(det, path, outdir, show_steps=True):
    events = load_sysmon_csv(path)
    wins = build_windows(events, 8, 1)
    scores = det.score_windows(wins)
    peak = int(np.argmax(scores))
    ex = explain_window(det, wins[peak], k=3)
    flagged = scores[peak] >= 0.5
    title = Path(path).stem
    print(f"\n{title}: {len(events)} events, {len(wins)} windows, peak score {scores[peak]:.2f} (window {peak + 1})")
    print(format_report(wins[peak], ex, flagged))
    if show_steps:
        print("\nStep breakdown (* = top-3):")
        print(step_table(wins[peak], ex))
    plot_session(Path(outdir) / f"{title}.png", title, wins, scores, peak, ex)
    return scores


def cmd_demo(a):
    mp = ROOT / "models/sysmon_detector.json"
    if not mp.exists():
        cfg = load_config(ROOT / "configs/sysmon.yaml")
        det, _ = train(cfg, ROOT)
        det.save(mp)
    det = SequenceAnomalyDetector.load(mp)
    out = ROOT / "results/demo"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("benign_session", "attack_session"):
        run_session(det, ROOT / f"demo/{name}.csv", out)
    print(f"\nSaved plots to results/demo/")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="seqguard", description="Explainable sequence anomaly detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-data", help="regenerate the Sysmon-schema telemetry")
    g.add_argument("--out", default="data/sysmon/sysmon_events.csv"); g.add_argument("--seed", type=int, default=7)
    g.set_defaults(fn=cmd_generate)
    t = sub.add_parser("train", help="fit the detector on normal data and save it"); t.add_argument("-c", "--config", required=True)
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("evaluate", help="train, then report metrics on the test split")
    e.add_argument("-c", "--config", required=True); e.add_argument("--ablation", action="store_true")
    e.add_argument("--explain-n", type=int, default=150, help="flagged windows used for explanation faithfulness")
    e.set_defaults(fn=cmd_evaluate)
    d = sub.add_parser("detect", help="score a Sysmon-schema CSV with a saved model")
    d.add_argument("--model", default="models/sysmon_detector.json"); d.add_argument("--input", required=True)
    d.add_argument("--window", type=int, default=8); d.set_defaults(fn=cmd_detect)
    s = sub.add_parser("demo", help="run the benign and the attack demo sessions"); s.set_defaults(fn=cmd_demo)
    an = sub.add_parser("analyze", help="score and explain an ad-hoc sequence of events")
    an.add_argument("--events", required=True, help="events as a chain or comma list, e.g. 'winword.exe -> powershell.exe -> rundll32.exe -> external POST'")
    an.add_argument("--model", default="models/sysmon_detector.json")
    an.set_defaults(fn=cmd_analyze)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
