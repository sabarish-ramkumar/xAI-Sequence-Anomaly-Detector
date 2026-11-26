"""Converts xAI feature contributions into short analyst notes."""
from __future__ import annotations

from typing import Iterable, Sequence

# role -> (members, noun phrase used in the sentence)
ROLES: dict[str, tuple[frozenset, str]] = {
    "office": (frozenset({"winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe", "onenote.exe", "acrord32.exe"}),
               "document application"),
    "browser": (frozenset({"chrome.exe", "msedge.exe", "firefox.exe", "iexplore.exe"}), "web browser"),
    "script": (frozenset({"powershell.exe", "pwsh.exe", "cmd.exe", "wscript.exe", "cscript.exe"}), "script interpreter"),
    "lolbin": (frozenset({"rundll32.exe", "regsvr32.exe", "mshta.exe", "certutil.exe", "bitsadmin.exe",
                          "msiexec.exe", "installutil.exe", "wmic.exe", "wmiprvse.exe"}),
               "system utility"),
    "recon": (frozenset({"whoami.exe", "net.exe", "net1.exe", "nltest.exe", "ipconfig.exe", "systeminfo.exe",
                         "tasklist.exe", "arp.exe", "nslookup.exe", "quser.exe"}), "reconnaissance command"),
    "persist": (frozenset({"schtasks.exe", "sc.exe", "reg.exe"}), "persistence-capable utility"),
    "webworker": (frozenset({"w3wp.exe", "httpd.exe", "nginx.exe", "tomcat.exe", "php-cgi.exe"}), "web server worker"),
}

# (actor role, target role) -> (ATT&CK id, short name); network rules use ("*", "net-<dest>-<action>")
ATTACK_HINTS = {
    ("office", "script"): ("T1204.002 / T1059", "document spawning a script interpreter (macro-style execution)"),
    ("script", "lolbin"): ("T1218", "system binary proxy execution"),
    ("browser", "lolbin"): ("T1218", "browser handing off to a proxy-execution binary"),
    ("lolbin", "script"): ("T1059", "proxy-execution binary starting a shell"),
    ("webworker", "script"): ("T1505.003", "web shell: web server worker spawning a shell"),
    ("script", "recon"): ("T1033 / T1087 / T1016", "host and account discovery"),
    ("script", "persist"): ("T1053.005", "scheduled task / service persistence"),
    ("lolbin", "recon"): ("T1033 / T1482", "discovery launched from a proxy-execution binary"),
}
NET_HINTS = {
    ("lolbin", "external", "POST"): ("T1071.001", "command-and-control over web protocols"),
    ("script", "external", "POST"): ("T1071.001", "script posting data to an external host"),
    ("lolbin", "external", "GET"): ("T1105", "tool download by a built-in utility"),
    ("script", "internal", "WINRM"): ("T1021.006", "remote management towards an internal host"),
}


def role_of(proc: str) -> str | None:
    for role, (members, _) in ROLES.items():
        if proc in members:
            return role
    return None


def _np(proc: str, with_role: bool = True) -> str:
    role = role_of(proc)
    if not with_role or role is None:
        return proc
    noun = ROLES[role][1]
    article = "an" if noun[0] in "aeiou" else "a"
    return f"{article} {noun} ({proc})"


def _get(f, name, default=None):
    return f[name] if isinstance(f, dict) else getattr(f, name, default)


def _clause(f, introduced: set) -> str:
    kind, fields = _get(f, "kind"), _get(f, "fields") or {}
    actor, target, action = fields.get("actor", ""), fields.get("target", ""), fields.get("action", "")
    first = actor not in introduced
    introduced.add(actor)
    if kind == "process":
        tgt_first = target not in introduced
        introduced.add(target)
        return f"{_np(actor, first)} launched {_np(target, tgt_first)}"
    if kind == "network":
        verb = {"POST": "sent an HTTP POST to", "GET": "issued an HTTP GET to", "CONNECT": "opened a connection to",
                "SMB": "opened an SMB session with", "WINRM": "opened a WinRM session with",
                "LDAP": "queried LDAP on", "DNS": "made a DNS lookup via"}.get(action, f"made a {action} connection to")
        where = "an external destination" if target == "external" else f"an {target} host"
        return f"{_np(actor, first)} {verb} {where}"
    if kind == "file":
        return f"{_np(actor, first)} created a file ({action.replace('_', ' ')})"
    if kind == "logon":
        return f"a {action} logon occurred"
    if kind == "count":
        return (f"event {fields.get('token', '').replace('count:', '')} occurred {fields.get('observed')} time(s) "
                f"where {fields.get('typical')} is normal")
    if kind == "log":
        tmpl = fields.get("template") or _get(f, "token")
        return f"the log event '{tmpl}' occurred"
    return "the session ended"


def _rarity(f) -> str:
    if _get(f, "novel"):
        return "never seen in normal activity"
    n_tr = _get(f, "transition_count", 0)
    n_ev = _get(f, "event_count", 0)
    if _get(f, "kind") == "count":
        return ""
    if n_tr == 0:
        return f"seen {int(n_ev)} time(s) in normal activity but never in this order"
    if _get(f, "prob", 1) < 0.01:
        return f"rare after its predecessor (p={_get(f, 'prob'):.4f})"
    return ""


def hints_for(features: Sequence) -> list[tuple[str, str]]:
    seen, out = set(), []
    for f in features:
        fields = _get(f, "fields") or {}
        kind, actor, target, action = _get(f, "kind"), fields.get("actor", ""), fields.get("target", ""), fields.get("action", "")
        key = None
        if kind == "process":
            key = ATTACK_HINTS.get((role_of(actor), role_of(target)))
        elif kind == "network":
            key = NET_HINTS.get((role_of(actor), target, action))
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def classify_reason(features: Sequence) -> str:
    kinds = {_get(f, "kind") for f in features}
    if any(_get(f, "novel") for f in features) and kinds <= {"process", "network", "file"}:
        return "Never-before-seen activity in the process sequence"
    if kinds == {"process"}:
        return "Unusual process sequence"
    if "network" in kinds and "process" in kinds:
        return "Unusual process-to-network sequence"
    if kinds == {"network"}:
        return "Unusual network activity"
    if kinds & {"log", "count", "end"}:
        return "Unusual event sequence"
    return "Unusual activity sequence"


def plain_english_summary(features: Iterable, score: float | None = None, with_hints: bool = True) -> str:
    """Build a short analyst note from the top contributing features."""
    feats = sorted(features, key=lambda f: _get(f, "contribution"), reverse=True)
    feats = [f for f in feats if _get(f, "contribution") > 0]
    if not feats:
        return "No single step stands out."
    chrono = sorted(feats, key=lambda f: _get(f, "step"))
    introduced: set = set()
    clauses = []
    seen_novel = False
    for i, f in enumerate(chrono):
        text = _clause(f, introduced)
        note = _rarity(f)
        if note == "never seen in normal activity":
            note = "unseen" if seen_novel else note
            seen_novel = True
        if note:
            text += f" ({note})"
        if i == 0:
            clauses.append(text)
        else:
            gap = _get(f, "step") - _get(chrono[i - 1], "step")
            clauses.append(("then " if gap == 1 else "later ") + text)
    score_str = f" (score {score:.2f})" if score is not None else ""
    body = "Flagged" + score_str + ": " + ", ".join(clauses) + "."
    if with_hints:
        hints = hints_for(chrono)
        if hints:
            body += " ATT&CK: " + ", ".join(f"{name} [{tid}]" for tid, name in hints) + "."
    return body[0].upper() + body[1:]


def translate_contributions(rows: Iterable[tuple[str, str, float]]) -> str:
    """Translate the bare `(feature, value, contribution)` form used in the task brief,
    e.g. ("parent_process", "rundll32.exe", 0.34), into one sentence."""
    rows = sorted(rows, key=lambda r: r[2], reverse=True)
    byname = {n: v for n, v, _ in rows}
    parts = []
    parent, child = byname.get("parent_process"), byname.get("child_process")
    if parent and child:
        parts.append(f"{parent} launched {child}")
    elif parent or child:
        parts.append(f"{parent or child} was involved in an unusual process event")
    net = byname.get("network_action")
    if net:
        parts.append(f"the activity was followed by {net}" if parts else f"there was {net}")
    for n, v, _ in rows:
        if n not in ("parent_process", "child_process", "network_action"):
            parts.append(f"{n.replace('_', ' ')} was {v}")
    return "Analyst summary: the sequence was flagged because " + " and ".join(parts) + "."
