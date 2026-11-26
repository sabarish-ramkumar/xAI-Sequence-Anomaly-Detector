"""Turns a flat event log into time-ordered sessions and sliding windows."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schema import BOS, UNK_CTX, Event, Window


def group_by_entity(events: Iterable[Event]) -> dict[str, list[Event]]:
    """Group events per user/host and sort every stream by time (stable)."""
    streams: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        streams[ev.entity].append(ev)
    for evs in streams.values():
        evs.sort(key=lambda e: e.ts)
    return dict(streams)


def split_sessions(stream: list[Event], idle_gap: float = 1800.0) -> list[list[Event]]:
    """A new session starts at a logon event or after `idle_gap` seconds of silence."""
    sessions: list[list[Event]] = []
    current: list[Event] = []
    for ev in stream:
        gap = ev.ts - current[-1].ts if current else 0.0
        if current and (ev.kind == "logon" or gap > idle_gap):
            sessions.append(current)
            current = []
        current.append(ev)
    if current:
        sessions.append(current)
    return sessions


def sliding_windows(session: list[Event], size: int, stride: int, min_len: int = 1,
                    tag: str = "") -> list[Window]:
    """Fixed-size windows over one session. The tail is always covered."""
    n = len(session)
    if n < min_len:
        return []
    last = max(n - size, 0)
    starts = list(range(0, last + 1, stride)) if last > 0 else [0]
    if not starts or starts[-1] != last:
        starts.append(last)
    out = []
    for s in starts:
        prev = session[s - 1].token if s > 0 else (BOS if session[0].kind == "logon" else UNK_CTX)
        out.append(Window(entity=session[0].entity, events=session[s:s + size],
                          prev_token=prev, wid=f"{tag}:{s}"))
    return out


def build_windows(events: Iterable[Event], size: int = 8, stride: int = 1,
                  idle_gap: float = 1800.0) -> list[Window]:
    """Process/network telemetry -> sliding windows per entity session."""
    windows: list[Window] = []
    for entity, stream in group_by_entity(events).items():
        for k, sess in enumerate(split_sessions(stream, idle_gap)):
            windows += sliding_windows(sess, size, stride, tag=f"{entity}#{k}")
    return windows


def session_windows(sessions: dict[str, list[Event]]) -> list[Window]:
    """Log data that already comes as one sequence per id (e.g. HDFS block):
    each sequence is a single terminal window."""
    return [Window(entity=sid, events=sorted(evs, key=lambda e: e.ts), terminal=True, wid=sid)
            for sid, evs in sessions.items() if evs]
