"""seqguard: sequence anomaly detection with Shapley explanations and analyst-readable summaries."""
from .explain import Explanation, FeatureContribution, ShapleyExplainer, explain_window
from .model import Decision, SequenceAnomalyDetector, TransitionModel
from .schema import BOS, EOS, Event, Window
from .translate import plain_english_summary, translate_contributions
from .windows import build_windows, session_windows

__all__ = ["BOS", "EOS", "Event", "Window", "TransitionModel", "SequenceAnomalyDetector", "Decision",
           "ShapleyExplainer", "FeatureContribution", "Explanation", "explain_window",
           "plain_english_summary", "translate_contributions", "build_windows", "session_windows"]
__version__ = "1.0.0"
