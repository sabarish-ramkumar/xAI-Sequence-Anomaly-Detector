"""Containers shared by the loaders, the window builder, the model and the explainer."""
from __future__ import annotations

from dataclasses import dataclass, field

BOS = "<BOS>"
EOS = "<EOS>"
UNK_CTX = "<UNK>"  # context of a fragment whose real predecessor is not known


@dataclass(slots=True)
class Event:
    """One observed event with model token and readable metadata."""

    ts: float
    entity: str
    kind: str  # process | network | file | logon | log
    token: str
    fields: dict = field(default_factory=dict)
    label: int = 0
    campaign: str = ""
    stage: str = ""


@dataclass(slots=True)
class Window:
    """Ordered slice of an entity's event stream, retaining `prev_token` for context."""

    entity: str
    events: list
    prev_token: str = BOS
    terminal: bool = False
    wid: str = ""

    @property
    def label(self) -> int:
        return int(any(e.label for e in self.events))

    @property
    def tokens(self) -> list[str]:
        toks = [e.token for e in self.events]
        return toks + [EOS] if self.terminal else toks

    @property
    def contexts(self) -> list[str]:
        toks = [e.token for e in self.events]
        return ([self.prev_token] + toks)[: len(self.tokens)]

    @property
    def start_ts(self) -> float:
        return self.events[0].ts

    @property
    def end_ts(self) -> float:
        return self.events[-1].ts

    def step_event(self, i: int):
        """Event behind step i, or None for the synthetic end-of-session step."""
        return self.events[i] if i < len(self.events) else None

    def __len__(self) -> int:
        return len(self.tokens)
