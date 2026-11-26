"""Generate synthetic Sysmon-style telemetry with ground-truth intrusion campaigns.
Run: python scripts/generate_sysmon_dataset.py --out data/sysmon/sysmon_events.csv
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

P, N, F = "p", "n", "f"  # process create, network connect, file create
BASE_DAY = datetime(2026, 3, 2, tzinfo=timezone.utc)
BASELINE_DAYS, VAL_DAYS, TEST_DAYS = 8, 2, 4

EXTRA_APPS = ["slack.exe", "spotify.exe", "zoom.exe", "teams.exe", "onedrive.exe"]
NEW_APPS = ["notion.exe", "figma.exe", "obsidian.exe"]  # only appear in the test period


# ------------------------------------------------------------ benign activities
def background(r, user):
    return r.choice([
        [(P, "services.exe", "svchost.exe")],
        [(P, "svchost.exe", "taskhostw.exe")],
        [(P, "svchost.exe", "wuauclt.exe"), (N, "wuauclt.exe", "external", "GET")],
        [(N, "svchost.exe", "external", "GET")],
        [(N, "svchost.exe", "internal", "LDAP")],
        [(P, "svchost.exe", "searchindexer.exe")],
        [(P, "svchost.exe", "backgroundtaskhost.exe")],
        [(P, "taskeng.exe", "updater.exe"), (N, "updater.exe", "external", "GET")],
        [(N, "msmpeng.exe", "external", "POST")],
        [(N, "svchost.exe", "internal", "DNS")],
    ])


def browse(r, u):
    b = u["browser"]
    ev = [(P, "explorer.exe", b)] + [(N, b, "external", r.choice(["GET", "GET", "GET", "POST"]))
                                     for _ in range(r.randint(2, 6))]
    if r.random() < 0.25:
        ev.append((F, b, "download_pdf"))
        if r.random() < 0.4:
            ev.append((P, b, "acrord32.exe"))
    return ev


def document(r, u):
    ev = [(P, "explorer.exe", "winword.exe"), (F, "winword.exe", "write_doc")]
    if r.random() < 0.5:
        ev.append((N, "winword.exe", "internal", "SMB"))
    if r.random() < 0.05:
        ev.append((P, "winword.exe", u["browser"]))  # clicked a hyperlink
    return ev


def spreadsheet(r, u):
    ev = [(P, "explorer.exe", "excel.exe"), (F, "excel.exe", "write_xlsx")]
    if r.random() < 0.5:
        ev.append((N, "excel.exe", "internal", "SMB"))
    return ev


def email(r, u):
    ev = [(P, "explorer.exe", "outlook.exe"), (N, "outlook.exe", "external", "CONNECT")]
    for app in r.sample(["winword.exe", "excel.exe", "acrord32.exe"], k=r.choice([0, 0, 1, 1, 2])):
        ev.append((P, "outlook.exe", app))
    ev.append((N, "outlook.exe", "external", "CONNECT"))
    return ev


def meeting(r, u):
    app = u["meeting_app"]
    return [(P, "explorer.exe", app), (N, app, "external", "CONNECT"), (N, app, "external", "CONNECT")]


def extra_app(r, u):
    app = r.choice(u["apps"])
    return [(P, "explorer.exe", app), (N, app, "external", "GET")]


def new_tool(r, u):
    app = u["new_app"]
    return [(P, "explorer.exe", app), (N, app, "external", "GET"), (N, app, "external", "POST")]


def dev(r, u):
    ev = [(P, "explorer.exe", "code.exe"), (P, "code.exe", "cmd.exe")]
    for _ in range(r.randint(1, 3)):
        tool = r.choice(["git.exe", "python.exe", "node.exe", "npm.exe"])
        ev.append((P, "cmd.exe", tool))
        if tool in ("git.exe", "npm.exe", "python.exe") and r.random() < 0.7:
            ev.append((N, tool, "external", r.choice(["GET", "CONNECT", "POST"])))
    return ev


def admin(r, u):
    ev = [(P, "explorer.exe", "powershell.exe")]
    for _ in range(r.randint(1, 4)):
        tool = r.choice(["ipconfig.exe", "whoami.exe", "net.exe", "ping.exe", "schtasks.exe",
                         "nltest.exe", "tasklist.exe", "systeminfo.exe"])
        ev.append((P, "powershell.exe", tool))
    if r.random() < 0.5:
        ev.append((N, "powershell.exe", "internal", "WINRM"))
    return ev


def web_request(r, u):
    ev = [(N, "w3wp.exe", "internal", "SQL")]
    if r.random() < 0.4:
        ev.append((N, "w3wp.exe", "external", "GET"))
    if r.random() < 0.02:
        ev.append((P, "w3wp.exe", "csc.exe"))  # JIT compile of an ASP.NET page
    return ev


PERSONAS = {
    "office": [(browse, 3), (document, 3), (spreadsheet, 2), (email, 3), (meeting, 1.5), (background, 3), (extra_app, 1)],
    "dev": [(dev, 5), (browse, 3), (meeting, 1), (email, 1), (background, 3), (extra_app, 1), (admin, 0.3)],
    "admin": [(admin, 4), (browse, 2), (email, 2), (meeting, 1), (background, 3), (dev, 0.4), (extra_app, 1)],
    "server": [(web_request, 6), (background, 4)],
}


# ------------------------------------------------------------- attack campaigns
def c_maldoc(r):
    ev = [("delivery", (P, "outlook.exe", "winword.exe")), ("execution", (P, "winword.exe", "powershell.exe")),
          ("proxy_exec", (P, "powershell.exe", "rundll32.exe")), ("c2", (N, "rundll32.exe", "external", "POST"))]
    if r.random() < 0.7:
        ev += [("discovery", (P, "rundll32.exe", "cmd.exe")), ("discovery", (P, "cmd.exe", "whoami.exe"))]
        if r.random() < 0.5:
            ev.append(("discovery", (P, "cmd.exe", "nltest.exe")))
    ev += [("exfil", (N, "rundll32.exe", "external", "POST")) for _ in range(r.randint(1, 3))]
    return ev


def c_hta(r):
    ev = [("execution", (P, r.choice(["explorer.exe", "chrome.exe"]), "mshta.exe")),
          ("execution", (P, "mshta.exe", "powershell.exe")), ("download", (P, "powershell.exe", "certutil.exe")),
          ("download", (N, "certutil.exe", "external", "GET")), ("download", (F, "certutil.exe", "write_exe")),
          ("proxy_exec", (P, "powershell.exe", "regsvr32.exe")), ("c2", (N, "regsvr32.exe", "external", "GET"))]
    return ev


def c_webshell(r):
    ev = [("execution", (P, "w3wp.exe", "cmd.exe"))]
    recon = [(P, "cmd.exe", "whoami.exe"), (P, "cmd.exe", "ipconfig.exe"), (P, "cmd.exe", "net.exe")]
    r.shuffle(recon)
    ev += [("discovery", x) for x in recon[: r.randint(2, 3)]]
    ev += [("download", (P, "cmd.exe", "certutil.exe")), ("download", (N, "certutil.exe", "external", "GET")),
           ("execution", (P, "cmd.exe", "powershell.exe")), ("c2", (N, "powershell.exe", "external", "POST"))]
    return ev


def c_bits_persist(r):
    return [("download", (P, "cmd.exe", "bitsadmin.exe")), ("download", (N, "bitsadmin.exe", "external", "GET")),
            ("download", (F, "bitsadmin.exe", "write_exe")), ("persistence", (P, "cmd.exe", "schtasks.exe")),
            ("execution", (P, "taskeng.exe", "update_svc.exe")), ("c2", (N, "update_svc.exe", "external", "POST"))]


def c_wmi_lateral(r):
    return [("execution", (P, "wmiprvse.exe", "powershell.exe")), ("execution", (P, "powershell.exe", "cmd.exe")),
            ("discovery", (P, "cmd.exe", "net.exe")), ("persistence", (P, "powershell.exe", "schtasks.exe")),
            ("lateral", (N, "powershell.exe", "internal", "WINRM")), ("c2", (N, "powershell.exe", "external", "POST"))]


def c_lotl(r):
    """Living-off-the-land: only tools an admin also uses. The hard case."""
    ev = [("execution", (P, "explorer.exe", "powershell.exe")), ("discovery", (P, "powershell.exe", "whoami.exe")),
          ("discovery", (P, "powershell.exe", "net.exe")), ("discovery", (P, "powershell.exe", "nltest.exe")),
          ("lateral", (N, "powershell.exe", "internal", "WINRM")), ("persistence", (P, "powershell.exe", "schtasks.exe")),
          ("c2", (N, "powershell.exe", "external", "POST"))]
    return ev


CAMPAIGNS = {  # name -> (builder, instances, eligible personas)
    "maldoc_powershell_rundll32": (c_maldoc, 10, ("office", "dev", "admin")),
    "hta_certutil_regsvr32": (c_hta, 6, ("office", "dev", "admin")),
    "webshell_recon": (c_webshell, 5, ("server",)),
    "bitsadmin_persistence": (c_bits_persist, 6, ("office", "dev", "admin")),
    "wmi_lateral_movement": (c_wmi_lateral, 5, ("admin", "dev")),
    "lotl_admin_abuse": (c_lotl, 6, ("office", "dev")),
}


# ----------------------------------------------------------------------- build
def make_users(r):
    users = []
    for persona, n in (("office", 24), ("dev", 14), ("admin", 10), ("server", 4)):
        for i in range(n):
            name = f"{persona[:3]}{i:02d}"
            users.append({"user": name, "host": f"WS-{name.upper()}" if persona != "server" else f"SRV-{i:02d}",
                          "persona": persona, "browser": r.choice(["chrome.exe", "msedge.exe", "firefox.exe"]),
                          "meeting_app": r.choice(["teams.exe", "zoom.exe"]),
                          "apps": r.sample(EXTRA_APPS, 2), "new_app": None})
    for u in r.sample([x for x in users if x["persona"] != "server"], 8):
        u["new_app"] = r.choice(NEW_APPS)
    return users


def day_session(r, u, day):
    acts, weights = zip(*PERSONAS[u["persona"]])
    n_act = r.randint(30, 50) if u["persona"] == "server" else r.randint(8, 16)
    events = []
    if u["persona"] != "server":
        events += [("", ("l", "interactive")), ("", (P, "winlogon.exe", "userinit.exe")),
                   ("", (P, "userinit.exe", "explorer.exe"))]
    for _ in range(n_act):
        act = r.choices(acts, weights)[0]
        events += [("", e) for e in act(r, u)]
        if u["new_app"] and day >= BASELINE_DAYS + VAL_DAYS and r.random() < 0.08:
            events += [("", e) for e in new_tool(r, u)]
    return [(s, e, 0, "") for s, e in events]


def inject(r, session, name, chain):
    pos = r.randint(3, max(3, len(session) - 1))
    block = []
    for i, (stage, ev) in enumerate(chain):
        block.append((stage, ev, 1, name))
        if i < len(chain) - 1 and r.random() < 0.3:
            block.append(("", (P, "svchost.exe", r.choice(["taskhostw.exe", "searchindexer.exe"])), 0, ""))
    return session[:pos] + block + session[pos:]


def to_rows(r, u, day, session):
    t = BASE_DAY + timedelta(days=day, hours=r.uniform(7.5, 15), seconds=0)
    rows = []
    split = "train" if day < BASELINE_DAYS else "val" if day < BASELINE_DAYS + VAL_DAYS else "test"
    for stage, ev, label, camp in session:
        t += timedelta(seconds=min(r.expovariate(1 / 18) + 1, 240))
        kind = ev[0]
        row = {"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "host": u["host"], "user": u["user"],
               "event_id": {"p": 1, "n": 3, "f": 11, "l": 4624}[kind], "parent_image": "", "image": "",
               "dest_category": "", "action": "", "label": label, "campaign": camp, "stage": stage, "split": split}
        if kind == P:
            row["parent_image"], row["image"] = ev[1], ev[2]
        elif kind == N:
            row["image"], row["dest_category"], row["action"] = ev[1], ev[2], ev[3]
        elif kind == F:
            row["image"], row["action"] = ev[1], ev[2]
        else:
            row["action"] = ev[1]
        rows.append(row)
    return rows


def generate(seed: int = 7) -> pd.DataFrame:
    r = random.Random(seed)
    users = make_users(r)
    total_days = BASELINE_DAYS + VAL_DAYS + TEST_DAYS
    sessions = {(u["user"], d): day_session(r, u, d) for u in users for d in range(total_days)}
    # place every campaign instance on a distinct (user, test-day) slot
    taken = set()
    for name, (builder, count, personas) in CAMPAIGNS.items():
        pool = [u for u in users if u["persona"] in personas]
        for _ in range(count):
            while True:
                u, d = r.choice(pool), r.randrange(BASELINE_DAYS + VAL_DAYS, total_days)
                if (u["user"], d) not in taken:
                    taken.add((u["user"], d))
                    break
            tag = f"{name}#{len(taken):02d}"
            sessions[(u["user"], d)] = inject(r, sessions[(u["user"], d)], tag, builder(r))
    rows = []
    for u in users:
        for d in range(total_days):
            rows += to_rows(r, u, d, sessions[(u["user"], d)])
    return pd.DataFrame(rows).sort_values(["timestamp", "host"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sysmon/sysmon_events.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    df = generate(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} events, {df.label.sum()} attack events, "
          f"{df[df.campaign != ''].campaign.nunique()} campaign instances -> {args.out}")
    print(df.groupby("split").size().to_string())
