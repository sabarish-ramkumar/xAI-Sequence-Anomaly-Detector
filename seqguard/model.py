"""Bigram transition model and anomaly scorer."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .schema import BOS, EOS, Event, Window


class TransitionModel:
    """Kneser-Ney bigram (or unigram) over event tokens. Handles OOV with reserved mass and per-field novelty penalty."""

    def __init__(self, order: int = 2, oov_mass: float | None = None,
                 discount: float | None = None, novelty_penalty: float = 1.5):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.order = order
        self.oov_mass = oov_mass
        self.discount = discount
        self.novelty_penalty = novelty_penalty
        self.uni: dict[str, int] = {}
        self.bi: dict[str, dict[str, int]] = {}
        self.actors: set = set()
        self.targets: dict[str, set] = {}

    # ------------------------------------------------------------------ training
    def fit(self, sequences: list[list[Event]], eos: bool = False) -> "TransitionModel":
        self.uni, self.bi, self.actors, self.targets = {}, {}, set(), {}
        return self.update(sequences, eos)

    def update(self, sequences: list[list[Event]], eos: bool = False) -> "TransitionModel":
        """Add more normal sequences to the baseline (e.g. analyst-confirmed benign sessions)."""
        uni, bi = Counter(self.uni), defaultdict(Counter, {v: Counter(c) for v, c in self.bi.items()})
        targets = defaultdict(set, {k: set(v) for k, v in self.targets.items()})
        for seq in sequences:
            prev = BOS
            for ev in seq:
                uni[ev.token] += 1
                bi[prev][ev.token] += 1
                prev = ev.token
                self.actors.add(ev.fields.get("actor"))
                targets[ev.kind].add(ev.fields.get("target"))
            if eos:
                uni[EOS] += 1
                bi[prev][EOS] += 1
        self.uni, self.bi = dict(uni), {v: dict(c) for v, c in bi.items()}
        self.targets = dict(targets)
        return self._finalize()

    def _finalize(self) -> "TransitionModel":
        self.n_tokens = sum(self.uni.values())
        self.vocab = sorted(self.uni)
        hapax = sum(1 for c in self.uni.values() if c == 1)
        self.lam = self.oov_mass if self.oov_mass else (hapax + 1) / (self.n_tokens + 1)
        n1 = sum(1 for c in self.bi.values() for x in c.values() if x == 1)
        n2 = sum(1 for c in self.bi.values() for x in c.values() if x == 2)
        d = self.discount if self.discount else (n1 / (n1 + 2 * n2) if n1 + n2 else 0.75)
        self.d = float(min(max(d, 0.2), 0.95))
        cont: Counter = Counter()
        for succ in self.bi.values():
            cont.update(succ.keys())
        total = sum(cont.values())
        self.p0 = {w: (1 - self.lam) * cont[w] / total for w in self.vocab}
        self.ctx_total = {v: sum(c.values()) for v, c in self.bi.items()}
        self.ctx_types = {v: len(c) for v, c in self.bi.items()}
        self._ent_cache = {}
        return self

    # --------------------------------------------------------------- probabilities
    def _gamma(self, prev: str) -> float:
        c = self.ctx_total.get(prev)
        return 1.0 if not c else self.d * self.ctx_types[prev] / c

    def prob(self, prev: str, tok: str) -> float:
        if self.order == 1:
            c = self.uni.get(tok)
            return self.lam if c is None else (1 - self.lam) * c / self.n_tokens
        gamma = self._gamma(prev)
        if tok not in self.p0:
            return gamma * self.lam
        c = self.ctx_total.get(prev)
        if not c:
            return self.p0[tok]
        cnt = self.bi[prev].get(tok, 0)
        return max(cnt - self.d, 0.0) / c + gamma * self.p0[tok]

    def unknown_fields(self, ev: Event | None) -> int:
        if ev is None:
            return 0
        n = int(ev.fields.get("actor") not in self.actors)
        n += int(ev.fields.get("target") not in self.targets.get(ev.kind, ()))
        return n

    def surprisal(self, prev: str, tok: str, ev: Event | None = None) -> float:
        s = -math.log(self.prob(prev, tok))
        if tok not in self.p0:
            s += self.novelty_penalty * self.unknown_fields(ev)
        return s

    def entropy(self, prev: str) -> float:
        """Entropy of the predictive distribution after `prev`."""
        key = prev if self.order == 2 else ""
        if key in self._ent_cache:
            return self._ent_cache[key]
        p = np.array([self.prob(prev, w) for w in self.vocab])
        oov = (self._gamma(prev) if self.order == 2 else 1.0) * self.lam
        h = float(-(p * np.log(p)).sum() - oov * math.log(oov))
        self._ent_cache[key] = h
        return h

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"order": self.order, "oov_mass": self.oov_mass, "discount": self.discount,
                "novelty_penalty": self.novelty_penalty, "uni": self.uni, "bi": self.bi,
                "actors": sorted(a for a in self.actors if a is not None),
                "targets": {k: sorted(t for t in v if t is not None) for k, v in self.targets.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionModel":
        m = cls(d["order"], d["oov_mass"], d["discount"], d["novelty_penalty"])
        m.uni, m.bi = d["uni"], d["bi"]
        m.actors = set(d["actors"])
        m.targets = {k: set(v) for k, v in d["targets"].items()}
        return m._finalize()


class CountModel:
    """Per-token count distribution. Flags sessions where an event is absent or repeated too many times."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.dist: dict[str, dict[int, int]] = {}
        self.n_seq = 0

    def fit(self, sequences: list[list[Event]], tokens: list[str]) -> "CountModel":
        self.n_seq = len(sequences)
        per = {t: Counter() for t in tokens}
        for seq in sequences:
            c = Counter(e.token for e in seq)
            for t in tokens:
                per[t][c.get(t, 0)] += 1
        self.dist = {t: dict(c) for t, c in per.items()}
        self._cache = {}
        return self

    def _table(self, tok: str):
        if tok in self._cache:
            return self._cache[tok]
        d = self.dist[tok]
        support = max(d) + 1
        denom = self.n_seq + self.alpha * (support + 1)
        probs = np.array([(d.get(n, 0) + self.alpha) / denom for n in range(support)])
        beyond = self.alpha / denom
        h = float(-(probs * np.log(probs)).sum() - beyond * math.log(beyond))
        mode = max(d, key=d.get)
        self._cache[tok] = (probs, beyond, h, mode)
        return self._cache[tok]

    def terms(self, events: list[Event]):
        """-> tokens, observed counts, surprisal, expected surprisal, probability, modal count"""
        c = Counter(e.token for e in events)
        toks = list(self.dist)
        s, h, p, obs, mode = [], [], [], [], []
        for t in toks:
            probs, beyond, ent, md = self._table(t)
            n = c.get(t, 0)
            pr = probs[n] if n < len(probs) else beyond
            s.append(-math.log(pr)); h.append(ent); p.append(pr); obs.append(n); mode.append(md)
        return toks, np.array(obs), np.array(s), np.array(h), np.array(p), np.array(mode)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "n_seq": self.n_seq,
                "dist": {t: {str(k): v for k, v in d.items()} for t, d in self.dist.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "CountModel":
        m = cls(d["alpha"])
        m.n_seq = d["n_seq"]
        m.dist = {t: {int(k): v for k, v in dd.items()} for t, dd in d["dist"].items()}
        m._cache = {}
        return m


@dataclass(slots=True)
class StepStats:
    """Everything the explainer needs to know about one scored window."""

    tokens: list
    contexts: list
    surprisal: np.ndarray
    expected: np.ndarray       # entropy of the predictive distribution at each step
    excess: np.ndarray         # surprisal - expected
    prob: np.ndarray
    event_count: np.ndarray    # how often the token occurred in training
    transition_count: np.ndarray
    novel: np.ndarray          # token never seen in training
    kinds: list = None         # "step" for sequence steps, "count" for per-type count terms
    meta: list = None          # extra readable facts per feature


@dataclass(slots=True)
class Decision:
    score: float
    raw: float
    flagged: bool
    threshold: float = 0.5


class SequenceAnomalyDetector:
    def __init__(self, order: int = 2, tau: float = 2.0, quantile: float = 0.995,
                 eos: bool = False, novelty_penalty: float = 1.5,
                 oov_mass: float | None = None, pooling: str = "lse",
                 use_counts: bool = False, count_alpha: float = 0.5):
        self.tau, self.quantile, self.eos, self.pooling = tau, quantile, eos, pooling
        self.use_counts = use_counts
        self.counts = CountModel(count_alpha) if use_counts else None
        self.model = TransitionModel(order=order, oov_mass=oov_mass, novelty_penalty=novelty_penalty)
        self.raw_threshold = 1.0
        self.scale = 1.0

    # ------------------------------------------------------------------ fitting
    def fit(self, train_sequences: list[list[Event]], val_windows: list[Window]) -> "SequenceAnomalyDetector":
        """Train on normal sequences; set threshold from held-out normal windows only."""
        self.model.fit(train_sequences, eos=self.eos)
        if self.use_counts:
            self.counts.fit(train_sequences, [t for t in self.model.vocab if t != EOS])
        raw = self.raw_scores(val_windows)
        # tiny margin: a validation window sitting exactly on the quantile must not be flagged
        self.raw_threshold = float(np.quantile(raw, self.quantile)) + 1e-6
        q75, q25 = np.quantile(raw, [0.75, 0.25])
        self.scale = float(max((q75 - q25) / 1.349, 0.25))
        return self

    # ------------------------------------------------------------------ scoring
    def step_stats(self, w: Window) -> StepStats:
        m = self.model
        toks, ctxs = w.tokens, w.contexts
        n = len(toks)
        s, h, p = np.zeros(n), np.zeros(n), np.zeros(n)
        ec, tc, nov = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
        for i, (c, t) in enumerate(zip(ctxs, toks)):
            s[i] = m.surprisal(c, t, w.step_event(i))
            h[i] = m.entropy(c)
            p[i] = m.prob(c, t)
            ec[i] = m.uni.get(t, 0)
            tc[i] = m.bi.get(c, {}).get(t, 0)
            nov[i] = t not in m.p0
        kinds, meta = ["step"] * n, [{}] * n
        if self.use_counts:
            ct, obs, cs, ch, cp, mode = self.counts.terms(w.events)
            toks = toks + [f"count:{t}" for t in ct]
            ctxs = ctxs + [""] * len(ct)
            s, h, p = np.concatenate([s, cs]), np.concatenate([h, ch]), np.concatenate([p, cp])
            ec = np.concatenate([ec, np.full(len(ct), self.counts.n_seq)])
            tc, nov = np.concatenate([tc, np.zeros(len(ct))]), np.concatenate([nov, np.zeros(len(ct), dtype=bool)])
            kinds += ["count"] * len(ct)
            meta += [{"token": t, "observed": int(o), "typical": int(md)} for t, o, md in zip(ct, obs, mode)]
        return StepStats(toks, ctxs, s, h, s - h, p, ec, tc, nov, kinds, meta)

    def pool(self, excess: np.ndarray) -> np.ndarray:
        """Aggregate per-step excess surprisal (rows = alternative windows) into one raw score."""
        e = np.atleast_2d(excess)
        if self.pooling == "mean":
            return e.mean(axis=1)
        z = e / self.tau
        m = z.max(axis=1, keepdims=True)
        return self.tau * (m[:, 0] + np.log(np.exp(z - m).mean(axis=1)))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(np.asarray(raw) - self.raw_threshold) / self.scale))

    def excess(self, w: Window) -> np.ndarray:
        return self.step_stats(w).excess

    def raw_scores(self, windows: list[Window]) -> np.ndarray:
        return np.array([self.pool(self.excess(w))[0] for w in windows])

    def score_windows(self, windows: list[Window]) -> np.ndarray:
        return self.calibrate(self.raw_scores(windows))

    def decide(self, w: Window, threshold: float = 0.5) -> Decision:
        raw = float(self.pool(self.excess(w))[0])
        score = float(self.calibrate(raw))
        return Decision(score=score, raw=raw, flagged=score >= threshold, threshold=threshold)

    def update(self, sequences: list[list[Event]]) -> "SequenceAnomalyDetector":
        """Fold analyst-confirmed benign sequences into the model. Threshold stays fixed."""
        self.model.update(sequences, eos=self.eos)
        return self

    # --------------------------------------------------------------- persistence
    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "SequenceAnomalyDetector":
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {"tau": self.tau, "quantile": self.quantile, "eos": self.eos, "pooling": self.pooling,
                "raw_threshold": self.raw_threshold, "scale": self.scale, "model": self.model.to_dict(),
                "use_counts": self.use_counts, "counts": self.counts.to_dict() if self.use_counts else None}

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceAnomalyDetector":
        det = cls(order=d["model"]["order"], tau=d["tau"], quantile=d["quantile"], eos=d["eos"],
                  novelty_penalty=d["model"]["novelty_penalty"], oov_mass=d["model"]["oov_mass"],
                  pooling=d["pooling"], use_counts=d.get("use_counts", False))
        det.model = TransitionModel.from_dict(d["model"])
        if det.use_counts:
            det.counts = CountModel.from_dict(d["counts"])
        det.raw_threshold, det.scale = d["raw_threshold"], d["scale"]
        return det
