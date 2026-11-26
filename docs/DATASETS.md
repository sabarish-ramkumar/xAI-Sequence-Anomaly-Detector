# Datasets

## HDFS — `data/hdfs/`

Real, labelled Hadoop distributed filesystem logs from Xu et al. (SOSP 2009), taken from the `loglizer` repository (MIT licence).

- `HDFS_100k.log_structured.csv`: first 100k log lines parsed into 19 event templates by LogHub.
- `anomaly_label.csv`: block-level labels (Normal / Anomaly) for the 7,940 blocks that appear in the sample.
- A session = all log lines mentioning one block ID (same definition as the original benchmark).
- 313 of 7,940 sessions are anomalous.

**Caveat**: this sample covers only the head of the log, so most blocks are truncated to ~13 events. 37% of anomalous test sessions are identical to a normal training session in the retained lines. That sets an upper bound on achievable recall; the detector reaches 98% on the remaining (distinguishable) anomalies.

## Sysmon-style telemetry — `data/sysmon/sysmon_events.csv`

A generated dataset with per-event ground-truth labels.

Columns: `timestamp, host, user, event_id, parent_image, image, dest_category, action, label, campaign, stage, split`.

**Why generated**: public Windows host datasets (OTRF, EVTX-ATTACK-SAMPLES) are attack captures only — no labelled benign baseline exists. Generating the data gives full control over the train/test split and attack injection, which is needed for fair evaluation.

**What is in it**: 32,602 events, 52 users/hosts, 14 days. Four user personas (office worker, developer, admin, web server). Eight users adopt a new application only in the test days (realistic drift). 38 attack instances across 6 families: macro document → PowerShell → rundll32 → C2, HTA → certutil/regsvr32, web shell → cmd, bitsadmin persistence, WMI lateral movement, and a living-off-the-land chain using only admin tools.

**What it is not**: a measurement of real-world detection rates. Regenerate with `python scripts/generate_sysmon_dataset.py --seed N`. The loader accepts any CSV with the same columns.
