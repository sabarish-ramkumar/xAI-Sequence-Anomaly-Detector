"""CSV loaders for HDFS and Sysmon-style telemetry."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .schema import Event

BLOCK_RE = re.compile(r"(blk_-?\d+)")


# --------------------------------------------------------------------------- HDFS
def load_hdfs(structured_csv: str | Path, label_csv: str | Path):
    """Load LogHub HDFS logs grouped by block ID with session-level labels."""
    df = pd.read_csv(structured_csv)
    labels = pd.read_csv(label_csv).set_index("BlockId")["Label"].eq("Anomaly").astype(int).to_dict()
    stamp = pd.to_datetime(df["Date"].astype(str).str.zfill(6) + df["Time"].astype(str).str.zfill(6),
                           format="%y%m%d%H%M%S")
    ts = ((stamp - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy()
    ts = ts + df["LineId"].to_numpy() * 1e-6  # keep original line order for ties
    templates = dict(zip(df["EventId"], df["EventTemplate"]))
    sessions: dict[str, list[Event]] = {}
    for t, eid, content, tmpl in zip(ts, df["EventId"], df["Content"], df["EventTemplate"]):
        for blk in dict.fromkeys(BLOCK_RE.findall(content)):
            ev = Event(ts=float(t), entity=blk, kind="log", token=eid,
                       fields={"actor": "hdfs", "target": eid, "template": tmpl},
                       label=labels.get(blk, 0))
            sessions.setdefault(blk, []).append(ev)
    sess_labels = {b: labels.get(b, 0) for b in sessions}
    return sessions, sess_labels, templates


# ------------------------------------------------------------------------- Sysmon
SYSMON_COLUMNS = ["timestamp", "host", "user", "event_id", "parent_image", "image",
                  "dest_category", "action"]


def _clean_img(path: object) -> str:
    """Normalize process paths: strip quotes, extract executable basename, and lowercase."""
    if path is None or pd.isna(path):
        return ""
    p = str(path).strip().strip('"').strip("'")
    if not p:
        return ""
    return p.replace("\\", "/").split("/")[-1].lower()


def _token(row) -> tuple[str, str, dict]:
    try:
        eid = int(row.event_id)
    except (ValueError, TypeError):
        eid = 0

    parent = _clean_img(getattr(row, "parent_image", ""))
    child = _clean_img(getattr(row, "image", ""))
    dest = str(getattr(row, "dest_category", "") or "").lower().strip()
    action = str(getattr(row, "action", "") or "").strip()

    if not parent:
        parent = "unknown"
    if not child:
        child = "unknown"

    if eid in (1, 4688):  # process create (Sysmon 1 or Windows Security 4688)
        return "process", f"{parent}>{child}", {
            "actor": parent, "target": child, "action": "launch", "event_type": "process_create"}
    if eid == 3:  # network connection
        target = dest if dest else "external"
        act = action if action else "CONNECT"
        return "network", f"{child}>net:{target}:{act}", {
            "actor": child, "target": target, "action": act,
            "event_type": "network_connect", "dest_category": target}
    if eid == 11:  # file create
        target = action if action else "create"
        return "file", f"{child}>file:{target}", {
            "actor": child, "target": target, "action": target, "event_type": "file_create"}
    if eid in (4624, 0) or "logon" in str(action).lower():  # logon event
        act = action if action else "interactive"
        return "logon", f"user>logon:{act}", {
            "actor": "user", "target": "session", "action": act, "event_type": "logon"}

    # Drop unsupported event IDs (e.g. 7 image load, 10 process access) so real Sysmon exports don't flood alerts
    return None


def load_sysmon_csv(path: str | Path, split: str | None = None) -> list[Event]:
    """Load Sysmon-style CSV telemetry (supports EventID 1, 3, 11, 4624, 4688)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV from {path}: {e}")

    missing = set(SYSMON_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    df = df.fillna({"parent_image": "", "image": "", "dest_category": "", "action": "", "campaign": "", "stage": ""})

    try:
        ts = (pd.to_datetime(df["timestamp"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    except Exception:
        import warnings
        warnings.warn(f"Warning: Failed to parse timestamps in {path}; falling back to sequence ordering.")
        ts = pd.Series(range(len(df)))

    labels = df["label"] if "label" in df.columns else pd.Series(0, index=df.index)
    out = []
    for row, t, lab in zip(df.itertuples(index=False), ts, labels):
        res = _token(row)
        if res is None:
            continue
        kind, tok, fields = res
        user = getattr(row, "user", "") or "unknown"
        host = getattr(row, "host", "") or "local"
        out.append(Event(ts=float(t), entity=f"{user}@{host}", kind=kind, token=tok, fields=fields,
                         label=int(lab) if pd.notna(lab) else 0,
                         campaign=str(getattr(row, "campaign", "") or ""),
                         stage=str(getattr(row, "stage", "") or "")))
    return out
