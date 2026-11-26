"""Dataset preparation, training and persistence for the two supported datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .loaders import load_hdfs, load_sysmon_csv
from .model import SequenceAnomalyDetector
from .schema import Window
from .windows import build_windows, group_by_entity, session_windows, split_sessions


@dataclass
class Prepared:
    train_sequences: list
    val_windows: list[Window]
    test_windows: list[Window]
    extra: dict = field(default_factory=dict)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _sessions(events, idle_gap):
    return [s for stream in group_by_entity(events).values() for s in split_sessions(stream, idle_gap)]


def prepare_sysmon(cfg: dict, root: Path = Path(".")) -> Prepared:
    path, w = root / cfg["data"]["events"], cfg["windows"]
    train = load_sysmon_csv(path, "train")
    val = load_sysmon_csv(path, "val")
    test = load_sysmon_csv(path, "test")
    return Prepared(
        train_sequences=_sessions(train, w["idle_gap"]),
        val_windows=build_windows(val, w["size"], w["stride"], w["idle_gap"]),
        test_windows=build_windows(test, w["size"], w["stride"], w["idle_gap"]),
        extra={"test_events": test, "idle_gap": w["idle_gap"], "window": w})


def prepare_hdfs(cfg: dict, root: Path = Path(".")) -> Prepared:
    sessions, labels, templates = load_hdfs(root / cfg["data"]["structured"], root / cfg["data"]["labels"])
    order = sorted(sessions, key=lambda b: sessions[b][0].ts)
    cut = int(cfg["split"]["train_frac"] * len(order))
    pool, test_ids = order[:cut], order[cut:]
    normal = [b for b in pool if labels[b] == 0]
    n_val = int(cfg["split"]["val_frac_of_normal"] * len(normal))
    train_ids, val_ids = normal[:-n_val], normal[-n_val:]
    return Prepared(
        train_sequences=[sessions[b] for b in train_ids],
        val_windows=session_windows({b: sessions[b] for b in val_ids}),
        test_windows=session_windows({b: sessions[b] for b in test_ids}),
        extra={"templates": templates, "train_ids": train_ids})


PREPARE = {"sysmon": prepare_sysmon, "hdfs": prepare_hdfs}


def build_detector(model_cfg: dict) -> SequenceAnomalyDetector:
    return SequenceAnomalyDetector(
        order=model_cfg["order"], tau=model_cfg["tau"], quantile=model_cfg["quantile"], eos=model_cfg["eos"],
        pooling=model_cfg["pooling"], use_counts=model_cfg.get("use_counts", False),
        novelty_penalty=model_cfg.get("novelty_penalty", 1.5))


def train(cfg: dict, root: Path = Path(".")):
    data = PREPARE[cfg["dataset"]](cfg, root)
    det = build_detector(cfg["model"]).fit(data.train_sequences, data.val_windows)
    return det, data
