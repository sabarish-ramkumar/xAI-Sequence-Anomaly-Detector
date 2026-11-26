import math

import numpy as np

from seqguard.model import SequenceAnomalyDetector, TransitionModel
from seqguard.schema import Event, Window


def seq(tokens, ent="u"):
    return [Event(ts=i, entity=ent, kind="process", token=t,
                  fields={"actor": t.split(">")[0], "target": t.split(">")[-1]}) for i, t in enumerate(tokens)]


NORMAL = [seq(["a>b", "b>c", "c>d", "d>e"]) for _ in range(40)] + [seq(["a>b", "b>c", "c>f", "f>e"]) for _ in range(10)]


def test_predictive_distribution_sums_to_one():
    for order in (1, 2):
        m = TransitionModel(order=order).fit(NORMAL)
        for ctx in ["a>b", "b>c", "never-seen", "<BOS>"]:
            total = sum(m.prob(ctx, w) for w in m.vocab) + m._gamma(ctx) * m.lam if order == 2 else \
                sum(m.prob(ctx, w) for w in m.vocab) + m.lam
            assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_unseen_event_is_more_surprising_than_rare_seen_event():
    m = TransitionModel().fit(NORMAL)
    assert m.surprisal("b>c", "z>y") > m.surprisal("b>c", "c>f") > m.surprisal("b>c", "c>d")


def test_novelty_penalty_scales_with_unknown_fields():
    m = TransitionModel().fit(NORMAL)
    known = Event(0, "u", "process", "a>e", {"actor": "a", "target": "e"})
    one_new = Event(0, "u", "process", "a>q", {"actor": "a", "target": "q"})
    two_new = Event(0, "u", "process", "p>q", {"actor": "p", "target": "q"})
    s = [m.surprisal("b>c", e.token, e) for e in (known, one_new, two_new)]
    assert s[0] < s[1] < s[2]


def test_expected_surprisal_equals_entropy():
    m = TransitionModel().fit(NORMAL)
    ctx = "c>d"
    expected = sum(m.prob(ctx, w) * -math.log(m.prob(ctx, w)) for w in m.vocab)
    oov = m._gamma(ctx) * m.lam
    assert math.isclose(m.entropy(ctx), expected - oov * math.log(oov), rel_tol=1e-9)


def _detector():
    val = [Window("u", seq(["a>b", "b>c", "c>d", "d>e"]), wid=str(i)) for i in range(30)]
    return SequenceAnomalyDetector(quantile=0.99).fit(NORMAL, val)


def test_attack_chain_scores_above_threshold_and_normal_below():
    det = _detector()
    normal = Window("u", seq(["a>b", "b>c", "c>d", "d>e"]))
    attack = Window("u", seq(["a>b", "b>x", "x>y", "y>z"]))
    assert det.decide(normal).score < 0.5 < det.decide(attack).score


def test_threshold_maps_to_one_half():
    det = _detector()
    assert math.isclose(float(det.calibrate(np.array([det.raw_threshold]))[0]), 0.5)


def test_save_load_roundtrip_preserves_scores(tmp_path):
    det = _detector()
    w = Window("u", seq(["a>b", "b>x", "x>y", "y>z"]))
    det.save(tmp_path / "m.json")
    again = SequenceAnomalyDetector.load(tmp_path / "m.json")
    assert math.isclose(det.decide(w).score, again.decide(w).score, rel_tol=1e-12)


def test_update_absorbs_confirmed_benign_activity():
    det = _detector()
    w = Window("u", seq(["a>b", "b>x", "x>y", "y>z"]))
    before = det.decide(w).score
    det.update([seq(["a>b", "b>x", "x>y", "y>z"]) for _ in range(60)])
    assert det.decide(w).score < before
