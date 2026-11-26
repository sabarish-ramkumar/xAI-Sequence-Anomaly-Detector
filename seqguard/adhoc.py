"""Parse a hand-written event sequence (the notation used in the task sheet) into Events.

    winword.exe -> powershell.exe -> rundll32.exe -> external POST
    user logs in, explorer.exe, chrome.exe, chrome.exe -> website connection, chrome.exe -> file:download_pdf

Commas, semicolons or new lines separate items; ``a -> b -> c`` is a chain and expands to
the edges a>b and b>c. A lone name means "that program was started from the desktop shell".
"""
from __future__ import annotations

import re

from .schema import BOS, UNK_CTX, Event, Window

ARROW = re.compile(r"\s*(?:->|→|>)\s*")
LOGON_WORDS = {"user", "user.exe", "login", "logon", "logs in", "user logs in", "user logon"}
NET_HINTS = ("external", "website", "connection", "connect", "http", "internet", "net:", "post", "get", "c2", "server")
BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "iexplore.exe"}


def _exe(name: str) -> str:
    n = name.strip().lower().strip('"')
    return n if n.endswith(".exe") or n in ("user", "session") else n + ".exe"


def _event(i: int, kind: str, token: str, **fields) -> Event:
    return Event(ts=float(i * 10), entity="adhoc@analyst", kind=kind, token=token, fields=fields)


def _logon(i: int) -> Event:
    return _event(i, "logon", "user>logon:interactive", actor="user", target="session", action="interactive",
                  event_type="logon")


def _login(events: list[Event]) -> None:
    """'The user logged in' is shorthand for the logon event followed by the standard Windows logon hand-off."""
    if events and events[-1].kind == "logon":
        return
    events.append(_logon(len(events)))
    events.append(_process(len(events), "winlogon.exe", "userinit.exe"))


def _is_network(node: str) -> bool:
    return any(h in node.lower() for h in NET_HINTS)


def _network(i: int, actor: str, node: str) -> Event:
    n = node.lower()
    if n.startswith("net:"):
        _, dest, *rest = (n.split(":") + ["external", "connect"])[:3]
        action = rest[0].upper() if rest else "CONNECT"
    else:
        dest = "internal" if "internal" in n else "external"
        action = "POST" if "post" in n else "GET" if ("get" in n or actor in BROWSERS) else "CONNECT"
    return _event(i, "network", f"{actor}>net:{dest}:{action}", actor=actor, target=dest, action=action,
                  event_type="network_connect", dest_category=dest)


def _file(i: int, actor: str, node: str) -> Event:
    act = node.split(":", 1)[1] if node.lower().startswith("file:") else "download_pdf"
    return _event(i, "file", f"{actor}>file:{act}", actor=actor, target=act, action=act, event_type="file_create")


def _process(i: int, parent: str, child: str) -> Event:
    return _event(i, "process", f"{parent}>{child}", actor=parent, target=child, action="launch",
                  event_type="process_create")


def parse_events(text: str) -> list[Event]:
    events: list[Event] = []
    for item in (x.strip() for x in re.split(r"[,;\n]", text) if x.strip()):
        nodes = [n for n in ARROW.split(item) if n]
        if len(nodes) == 1:
            node = nodes[0]
            if node.lower() in LOGON_WORDS:
                _login(events)
            else:
                child = _exe(node)
                events.append(_process(len(events), "userinit.exe" if child == "explorer.exe" else "explorer.exe", child))
            continue
        for a, b in zip(nodes, nodes[1:]):
            i = len(events)
            if a.lower() in LOGON_WORDS:                      # "user.exe -> explorer.exe": the user logged in
                _login(events)
                b_exe = _exe(b)
                if not _is_network(b):
                    events.append(_process(len(events), "userinit.exe" if b_exe == "explorer.exe" else "explorer.exe", b_exe))
            elif b.lower() in LOGON_WORDS:
                _login(events)
            elif b.lower().startswith("file:") or "download" in b.lower():
                events.append(_file(i, _exe(a), b))
            elif _is_network(b) and not b.lower().endswith(".exe"):
                events.append(_network(i, _exe(a), b))
            else:
                events.append(_process(i, _exe(a), _exe(b)))
    return events


def build_window(events: list[Event]) -> Window:
    """A hand-typed sequence is a fragment unless it opens with a logon: without a known
    predecessor the first step is scored with the neutral context instead of the session start."""
    prev = BOS if events and events[0].kind == "logon" else UNK_CTX
    return Window(entity="adhoc@analyst", events=events, prev_token=prev, wid="adhoc:0")
