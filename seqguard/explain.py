"""Shapley attribution for flagged windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Callable

import numpy as np

from .model import SequenceAnomalyDetector, StepStats
from .schema import Window


class ShapleyExplainer:
    """Exact Shapley values up to `exact_limit` features, antithetic sampling above."""

    def __init__(self, value_fn: Callable[[np.ndarray], np.ndarray], n: int,
                 exact_limit: int = 12, n_permutations: int = 256, seed: int = 0):
        self.value_fn, self.n = value_fn, n
        self.exact_limit, self.n_permutations = exact_limit, n_permutations
        self.rng = np.random.default_rng(seed)

    def baseline(self) -> float:
        return float(self.value_fn(np.zeros((1, self.n), dtype=bool))[0])

    def full(self) -> float:
        return float(self.value_fn(np.ones((1, self.n), dtype=bool))[0])

    def attribute(self) -> np.ndarray:
        return self._exact() if self.n <= self.exact_limit else self._sampled()

    def _exact(self) -> np.ndarray:
        n = self.n
        idx = np.arange(1 << n)
        masks = ((idx[:, None] >> np.arange(n)) & 1).astype(bool)
        v = self.value_fn(masks)
        size = masks.sum(axis=1)
        weight = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
        phi = np.zeros(n)
        for i in range(n):
            without = idx[~masks[:, i]]
            phi[i] = (weight[size[without]] * (v[without | (1 << i)] - v[without])).sum()
        return phi

    def _sampled(self) -> np.ndarray:
        n = self.n
        m = max(16, min(self.n_permutations, int(3e6 // (n * n))))
        m += m % 2
        perms = np.array([self.rng.permutation(n) for _ in range(m // 2)])
        perms = np.concatenate([perms, perms[:, ::-1]])  # antithetic pairs
        ranks = np.argsort(perms, axis=1)
        ks = np.arange(n + 1)
        masks = ranks[:, None, :] < ks[None, :, None]       # (m, n+1, n) prefix coalitions
        v = self.value_fn(masks.reshape(-1, n)).reshape(m, n + 1)
        marginal = v[:, 1:] - v[:, :-1]
        phi = np.zeros(n)
        np.add.at(phi, perms.ravel(), marginal.ravel())
        return phi / m


@dataclass(slots=True)
class FeatureContribution:
    step: int
    token: str
    kind: str
    fields: dict
    contribution: float
    share: float
    prev_token: str
    surprisal: float
    expected: float
    prob: float
    event_count: int
    transition_count: int
    novel: bool

    @property
    def label(self) -> str:
        """Short analyst-facing name of the feature (the transition it describes)."""
        a, t = self.fields.get("actor", ""), self.fields.get("target", "")
        if self.kind == "process":
            return f"{a} -> {t}"
        if self.kind == "network":
            return f"{a} -> external {self.fields.get('action')}" if t == "external" \
                else f"{a} -> {t} {self.fields.get('action')}"
        if self.kind == "file":
            return f"{a} -> file {self.fields.get('action')}"
        if self.kind == "logon":
            return f"logon ({self.fields.get('action')})"
        return self.token if self.token != "<EOS>" else "end of session"


@dataclass(slots=True)
class Explanation:
    score: float
    baseline: float
    phi: np.ndarray                              # contributions in score units
    features: list[FeatureContribution]          # every step, in window order
    top: list[FeatureContribution] = field(default_factory=list)
    raw: float = 0.0
    phi_raw: np.ndarray = None                   # Shapley values of the pooled surprisal (nats)

    @property
    def efficiency_gap(self) -> float:
        """Largest violation of the Shapley efficiency axiom (should be ~0; checked in tests)."""
        return max(abs(float(self.phi_raw.sum()) - self.raw),
                   abs(float(self.phi.sum()) - (self.score - self.baseline)))


def explain_window(det: SequenceAnomalyDetector, w: Window, k: int = 3, **kw) -> Explanation:
    stats: StepStats = det.step_stats(w)
    e = stats.excess

    def value_fn(masks: np.ndarray) -> np.ndarray:
        return det.pool(np.where(masks, e, 0.0))

    ex = ShapleyExplainer(value_fn, len(e), **kw)
    phi_raw = ex.attribute()
    raw_base, raw_full = ex.baseline(), ex.full()
    base, full = float(det.calibrate(raw_base)), float(det.calibrate(raw_full))
    gap = raw_full - raw_base
    share = phi_raw / gap if abs(gap) > 1e-12 else np.zeros_like(phi_raw)
    phi = share * (full - base)
    feats = []
    for i, tok in enumerate(stats.tokens):
        ev = w.step_event(i)
        feats.append(FeatureContribution(
            step=i, token=tok, kind=ev.kind if ev else "end", fields=dict(ev.fields) if ev else {},
            contribution=float(phi[i]), share=float(share[i]),
            prev_token=stats.contexts[i], surprisal=float(stats.surprisal[i]),
            expected=float(stats.expected[i]), prob=float(stats.prob[i]),
            event_count=int(stats.event_count[i]), transition_count=int(stats.transition_count[i]),
            novel=bool(stats.novel[i])))
    top = sorted(feats, key=lambda f: f.contribution, reverse=True)[:k]
    return Explanation(score=full, baseline=base, phi=phi, features=feats, top=top, raw=raw_full, phi_raw=phi_raw)
