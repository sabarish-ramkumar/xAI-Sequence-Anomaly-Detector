import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from seqguard import evaluate as ev
from seqguard.loaders import load_sysmon_csv
from seqguard.pipeline import load_config, train
from seqguard.windows import build_windows


@pytest.fixture(scope="module")
def trained():
    cfg = load_config(ROOT / "configs" / "sysmon.yaml")
    return train(cfg, ROOT)


def test_every_injected_campaign_is_caught(trained):
    det, data = trained
    scores = det.score_windows(data.test_windows)
    res = ev.campaign_metrics(data.test_windows, scores)
    assert res["detected"] / res["campaigns"] >= 0.95


def test_detection_quality_on_held_out_days(trained):
    det, data = trained
    y = np.array([w.label for w in data.test_windows])
    m = ev.window_metrics(y, det.score_windows(data.test_windows))
    assert m["auroc"] > 0.97 and m["recall"] > 0.9


def test_demo_sessions_behave_as_documented(trained):
    det, _ = trained
    peak = {}
    for name in ("benign_session", "attack_session"):
        wins = build_windows(load_sysmon_csv(ROOT / "demo" / f"{name}.csv"), 8, 1)
        peak[name] = det.score_windows(wins).max()
    assert peak["benign_session"] < 0.5 < peak["attack_session"]


def test_explanations_are_more_faithful_than_random(trained):
    det, data = trained
    scores = det.score_windows(data.test_windows)
    m = ev.explanation_metrics(det, data.test_windows, scores, max_windows=30)
    assert m["drop_top"] > 2 * m["drop_random"] and m["hit_top"] > m["hit_random"] + 0.2
    assert m["max_efficiency_gap"] < 1e-9


def test_loader_normalisation_and_event_mapping(tmp_path):
    from seqguard.loaders import load_sysmon_csv
    p = tmp_path / "test_norm.csv"
    p.write_text(
        "timestamp,host,user,event_id,parent_image,image,dest_category,action\n"
        "2026-03-02 08:00:00,H1,U1,1,C:\\Windows\\System32\\cmd.exe,EXPLORER.EXE,,\n"
        "2026-03-02 08:00:05,H1,U1,4688,services.exe,SVCHOST.EXE,,\n"
        "2026-03-02 08:00:10,H1,U1,999,svchost.exe,svchost.exe,,\n"  # unknown ID ignored
        "2026-03-02 08:00:15,H1,U1,3,,powershell.exe,external,POST\n"
    )
    events = load_sysmon_csv(p)
    assert len(events) == 3  # event 999 is skipped
    assert events[0].token == "cmd.exe>explorer.exe"
    assert events[1].token == "services.exe>svchost.exe"
    assert events[2].token == "powershell.exe>net:external:POST"


def test_short_sequences_are_scored(trained):
    det, _ = trained
    from seqguard.schema import Event
    from seqguard.windows import build_windows
    # 2-event session
    evs = [
        Event(ts=1.0, entity="u@h", kind="process", token="winword.exe>powershell.exe", fields={"actor": "winword.exe", "target": "powershell.exe"}),
        Event(ts=2.0, entity="u@h", kind="process", token="powershell.exe>rundll32.exe", fields={"actor": "powershell.exe", "target": "rundll32.exe"}),
    ]
    wins = build_windows(evs, size=8, stride=1)
    assert len(wins) == 1
    score = det.score_windows(wins)[0]
    assert score > 0.50  # 2-event attack chain is flagged
