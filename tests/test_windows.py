from seqguard.schema import BOS, UNK_CTX, Event
from seqguard.windows import build_windows, session_windows, sliding_windows, split_sessions


def ev(ts, tok="a>b", kind="process", ent="u@h"):
    return Event(ts=ts, entity=ent, kind=kind, token=tok, fields={"actor": "a", "target": "b"})


def test_sessions_split_on_logon_and_idle_gap():
    stream = [ev(0), ev(10), ev(20, "user>logon:interactive", "logon"), ev(25), ev(5000)]
    sessions = split_sessions(stream, idle_gap=1800)
    assert [len(s) for s in sessions] == [2, 2, 1]


def test_sliding_windows_cover_tail_and_carry_context():
    sess = [ev(i, f"t{i}>x") for i in range(11)]
    wins = sliding_windows(sess, size=4, stride=3)
    assert wins[0].prev_token == UNK_CTX          # no logon at the start: a fragment, not a session start
    assert wins[1].prev_token == "t2>x"           # event right before the window
    assert wins[-1].events[-1] is sess[-1]        # last event is always covered
    assert all(len(w.events) == 4 for w in wins)


def test_events_are_ordered_by_time_before_windowing():
    shuffled = [ev(30, "c>d"), ev(10, "a>b"), ev(20, "b>c")]
    (w,) = build_windows(shuffled, size=3, stride=1)
    assert w.tokens == ["a>b", "b>c", "c>d"]


def test_terminal_window_appends_end_symbol_and_context():
    (w,) = session_windows({"blk_1": [ev(2, "E2", "log"), ev(1, "E1", "log")]})
    assert w.tokens == ["E1", "E2", "<EOS>"]
    assert w.contexts == [BOS, "E1", "E2"]


def test_session_that_opens_with_logon_starts_at_bos():
    sess = [ev(0, "user>logon:interactive", "logon"), ev(5), ev(9)]
    assert sliding_windows(sess, size=8, stride=1)[0].prev_token == BOS
