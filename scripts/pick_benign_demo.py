"""Select the median-scoring held-out benign office session for the demo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seqguard.model import SequenceAnomalyDetector
from seqguard.pipeline import load_config
from seqguard.loaders import load_sysmon_csv
from seqguard.windows import build_windows, group_by_entity, split_sessions

cfg = load_config("configs/sysmon.yaml")
det = SequenceAnomalyDetector.load(cfg["output"]["model"])
events = load_sysmon_csv(cfg["data"]["events"], "test")
cands = []
for ent, stream in group_by_entity(events).items():
    if not ent.startswith("off"):
        continue
    for sess in split_sessions(stream, cfg["windows"]["idle_gap"]):
        wins = build_windows(sess[:18], 8, 1)
        if len(sess) >= 18 and not any(e.label for e in sess) and wins:
            cands.append((ent, sess[:18], float(det.score_windows(wins).max())))
med = np.median([c[2] for c in cands])
ent, sess, peak = min(cands, key=lambda c: abs(c[2] - med))
user, host = ent.split("@")
df = pd.read_csv(cfg["data"]["events"])
t = pd.to_datetime(df["timestamp"], utc=True)
lo, hi = pd.Timestamp(sess[0].ts, unit="s", tz="UTC"), pd.Timestamp(sess[-1].ts, unit="s", tz="UTC")
sub = df[(df.user == user) & (df.host == host) & (t >= lo) & (t <= hi)]
cols = ["timestamp", "host", "user", "event_id", "parent_image", "image", "dest_category", "action"]
sub[cols].to_csv("demo/benign_session.csv", index=False)
print(f"{len(cands)} candidate sessions, median peak {med:.3f}; exported {ent} ({len(sub)} events, peak {peak:.3f})")
