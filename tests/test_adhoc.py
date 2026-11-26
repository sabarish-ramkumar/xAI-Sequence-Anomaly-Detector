"""The sequences written in the task sheet must behave as the sheet says."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from seqguard.adhoc import build_window, parse_events
from seqguard.explain import explain_window
from seqguard.pipeline import load_config, train


@pytest.fixture(scope="module")
def det():
    return train(load_config(ROOT / "configs" / "sysmon.yaml"), ROOT)[0]


def score(det, text):
    return float(det.score_windows([build_window(parse_events(text))])[0])


def test_chain_notation_expands_to_edges():
    evs = parse_events("winword.exe -> powershell.exe -> rundll32.exe -> external POST")
    assert [e.token for e in evs] == ["winword.exe>powershell.exe", "powershell.exe>rundll32.exe",
                                      "rundll32.exe>net:external:POST"]


@pytest.mark.parametrize("text", [
    "user.exe -> explorer.exe -> chrome.exe -> website connection",                      # task sheet 1.3, sequence A
    "user logs in, explorer.exe, chrome.exe, chrome.exe -> website connection, chrome.exe -> file:download_pdf",  # 1.1
    "explorer.exe -> chrome.exe",                                                        # a single benign step
])
def test_benign_examples_are_not_flagged(det, text):
    assert score(det, text) < 0.5


@pytest.mark.parametrize("text", [
    "winword.exe -> powershell.exe -> rundll32.exe -> external connection",              # task sheet 1.3, sequence B
    "winword.exe, winword.exe -> powershell.exe, powershell.exe -> rundll32.exe, rundll32.exe -> external POST",  # 1.7
])
def test_attack_examples_are_flagged_with_the_expected_contributors(det, text):
    win = build_window(parse_events(text))
    assert det.score_windows([win])[0] > 0.9
    top = {f.label for f in explain_window(det, win, k=3).top}
    assert "winword.exe -> powershell.exe" in top or "powershell.exe -> rundll32.exe" in top
