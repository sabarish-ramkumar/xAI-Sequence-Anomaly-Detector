from seqguard.explain import FeatureContribution
from seqguard.translate import plain_english_summary, translate_contributions


def feat(step, kind, actor, target, action="", c=0.3, novel=False, tc=0, ec=5):
    return FeatureContribution(step=step, token=f"{actor}>{target}", kind=kind,
                               fields={"actor": actor, "target": target, "action": action}, contribution=c,
                               share=0.3, prev_token="x", surprisal=9.0, expected=1.0, prob=1e-4,
                               event_count=ec, transition_count=tc, novel=novel)


def test_summary_is_built_from_the_given_features():
    a = plain_english_summary([feat(2, "process", "winword.exe", "powershell.exe", "launch", 0.5, True),
                               feat(3, "process", "powershell.exe", "rundll32.exe", "launch", 0.3, True),
                               feat(4, "network", "rundll32.exe", "external", "POST", 0.2, True)])
    b = plain_english_summary([feat(1, "process", "w3wp.exe", "cmd.exe", "launch", 0.6, True)])
    assert "winword.exe" in a and "rundll32.exe" in a and "POST" in a
    assert "w3wp.exe" in b and "winword.exe" not in b
    assert a.index("winword.exe") < a.index("powershell.exe") < a.index("POST")   # narrated in event order


def test_non_positive_contributions_are_ignored():
    s = plain_english_summary([feat(1, "process", "a.exe", "b.exe", "launch", -0.2)])
    assert "No single step stands out" in s


def test_contributions_in_the_brief_format():
    s = translate_contributions([("parent_process", "rundll32.exe", 0.34), ("child_process", "cmd.exe", 0.29),
                                 ("network_action", "external POST", 0.22)])
    assert "rundll32.exe launched cmd.exe" in s and "external POST" in s
