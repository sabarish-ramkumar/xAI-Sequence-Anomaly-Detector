import itertools
import math

import numpy as np
import pytest

from seqguard.explain import ShapleyExplainer, explain_window
from seqguard.model import SequenceAnomalyDetector
from seqguard.schema import Event, Window


def brute_force_shapley(f, n):
    phi = np.zeros(n)
    for perm in itertools.permutations(range(n)):
        mask = np.zeros(n, dtype=bool)
        prev = f(mask[None])[0]
        for i in perm:
            mask[i] = True
            cur = f(mask[None])[0]
            phi[i] += cur - prev
            prev = cur
    return phi / math.factorial(n)


def game(masks):
    w = np.array([3.0, 1.0, 0.0, 2.0, 1.0])
    return (masks * w).sum(axis=1) + 0.5 * masks[:, 0] * masks[:, 3] + np.log1p(masks[:, 1:3].sum(axis=1))


def test_exact_matches_brute_force_definition():
    phi = ShapleyExplainer(game, 5).attribute()
    assert np.allclose(phi, brute_force_shapley(game, 5))


def test_efficiency_and_null_player():
    phi = ShapleyExplainer(game, 5).attribute()
    ex = ShapleyExplainer(game, 5)
    assert math.isclose(phi.sum(), ex.full() - ex.baseline(), abs_tol=1e-12)
    null = ShapleyExplainer(lambda m: m[:, 0] * 2.0, 4).attribute()
    assert np.allclose(null[1:], 0)


def test_sampling_close_to_exact():
    f = lambda m: np.log1p((m * np.arange(1, 11)).sum(axis=1))
    exact = ShapleyExplainer(f, 10, exact_limit=12).attribute()
    approx = ShapleyExplainer(f, 10, exact_limit=2, n_permutations=2000, seed=1).attribute()
    assert np.abs(exact - approx).max() < 0.05                         # Monte-Carlo error, ~0.02 at 2000 permutations
    assert set(np.argsort(exact)[-3:]) == set(np.argsort(approx)[-3:])
    assert math.isclose(approx.sum(), exact.sum(), rel_tol=1e-9)     # telescoping keeps efficiency exact


def _seq(tokens):
    return [Event(i, "u", "process", t, {"actor": t.split(">")[0], "target": t.split(">")[-1], "action": "launch"})
            for i, t in enumerate(tokens)]


@pytest.fixture(scope="module")
def det():
    normal = [_seq(["a>b", "b>c", "c>d", "d>e", "e>f", "f>g"]) for _ in range(60)]
    val = [Window("u", _seq(["a>b", "b>c", "c>d", "d>e", "e>f", "f>g"]), wid=str(i)) for i in range(30)]
    return SequenceAnomalyDetector(quantile=0.99).fit(normal, val)


def test_explanation_finds_planted_anomaly_and_obeys_efficiency(det):
    w = Window("u", _seq(["a>b", "b>c", "c>zz", "d>e", "e>f", "f>g"]))
    ex = explain_window(det, w, k=3)
    assert ex.top[0].step == 2
    assert ex.efficiency_gap < 1e-9
    assert math.isclose(ex.phi.sum(), ex.score - ex.baseline, abs_tol=1e-9)


def test_long_windows_use_sampling_and_stay_efficient(det):
    toks = ["a>b", "b>c", "c>d", "d>e", "e>f", "f>g"] * 4
    toks[9] = "q>r"
    ex = explain_window(det, Window("u", _seq(toks)), k=3, exact_limit=8)
    assert ex.top[0].step == 9
    assert ex.efficiency_gap < 1e-9


def test_agrees_with_shap_kernel_explainer():
    shap = pytest.importorskip("shap")
    f = lambda X: game(np.asarray(X) > 0.5)
    ours = ShapleyExplainer(game, 5).attribute()
    kernel = shap.KernelExplainer(f, np.zeros((1, 5))).shap_values(np.ones((1, 5)), nsamples=2 ** 5 - 2, silent=True)
    assert np.allclose(ours, np.asarray(kernel).ravel(), atol=1e-6)
