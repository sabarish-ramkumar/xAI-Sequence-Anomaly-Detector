# seqguard: explainable sequence anomaly detector

**Task 1: Explainable Sequence Anomaly Detector**  
School of Computing, SASTRA Deemed University  
Author: Sabarish R (Reg. No: 128003217)

Learns what normal event sequences look like, scores new windows between 0 and 1, and, when a window is flagged,
explains the decision with Shapley values and a short analyst-readable summary.

```
$ python -m seqguard demo

attack_session: 14 events, 7 windows, peak score 1.00 (window 7)
Status  : FLAGGED   score 1.00  threshold 0.50
Reason  : Never-before-seen activity in the process sequence
Top contributors:
  1. winword.exe -> powershell.exe     +0.54   (step 3)
  2. powershell.exe -> rundll32.exe    +0.10   (step 4)
  3. rundll32.exe -> external POST     +0.10   (step 5)
Analyst summary:
  Flagged (score 1.00): a document application (winword.exe) launched a script interpreter ...
```

Every number above is computed by the model at run time; nothing in the demo or the translator is typed in.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate                            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m seqguard demo                              # benign + multi-stage attack session, with explanations
python -m seqguard evaluate -c configs/sysmon.yaml   # metrics on the held-out days (add --ablation)
python -m seqguard evaluate -c configs/hdfs.yaml     # metrics on the HDFS benchmark
python -m seqguard detect --input demo/attack_session.csv
python -m seqguard analyze --events "winword.exe -> powershell.exe -> rundll32.exe -> external POST"
python -m pytest -q                                  # 34 tests
```

Python 3.10+. Training takes a few seconds; no GPU, no network access needed (datasets are in `data/`).

## How the task maps to the code

| Task deliverable | Where |
|---|---|
| Sequence builder: read events, order by time, group into windows | `seqguard/loaders.py`, `seqguard/windows.py` |
| Model features and anomaly score (0-1, threshold) | `seqguard/model.py` |
| xAI: top-3 contributing features of a flagged window | `seqguard/explain.py` |
| Plain-English translator | `seqguard/translate.py` (`plain_english_summary`, `translate_contributions`) |
| Benign vs multi-stage attack demo | `python -m seqguard demo`, inputs in `demo/` |
| Datasets | `data/hdfs/` (real, labelled), `data/sysmon/` (generated, labelled) |

## Approach in one page

1. **Events become tokens.** A process launch is `parent>child`, a connection is `process>net:external:POST`,
   a file write is `process>file:download_pdf`. HDFS log lines use their template id (E5, E22, ...).
2. **Windows.** Per user/host, events are sorted by time, split into sessions (new logon or 30 minutes of silence)
   and cut into sliding windows of 8 events. Each window keeps the event that preceded it, so its first step is
   scored with real context. HDFS blocks are scored as whole sequences with an end-of-session symbol.
3. **Model.** An interpolated Kneser-Ney bigram over tokens, trained on normal data only. For every step the model
   computes the surprisal of what happened minus the entropy of what it expected after the previous event. That
   "excess surprisal" is near zero for normal behaviour whatever the context, and large for a step the model
   considers unlikely. Tokens never seen in training get a reserved probability mass plus a penalty per unfamiliar field.
4. **Score.** Excess values of a window are pooled with a log-sum-exp (so a short attack chain is not averaged away
   by benign neighbours) and passed through a sigmoid whose 0.5 point is the 99.5th percentile of held-out *normal*
   windows. A score above 0.5 means "rarer than 99.5% of normal windows".
5. **Explanation.** Features are the steps of the window. Exact Shapley values (permutation sampling above 12 steps)
   where a missing step is replaced by the expected surprisal in its context. They add up to
   `score(window) - score(baseline)`; this is unit-tested, and cross-checked against the `shap` library.
6. **Translation.** The top-3 features and their statistics (how often seen in training, probability, order) are
   written up in sentences; a small process-role lexicon supplies wording and ATT&CK hints.

Details and the math: `docs/METHODOLOGY.md`.

## Results (full tables in `docs/RESULTS.md`)

| | AUROC | precision | recall | FPR |
|---|---|---|---|---|
| Sysmon-style telemetry, 4 unseen days | 0.984 | 0.64 | 1.00 | 4.3% |
| HDFS LogHub sample, chronological test split | 0.809 | 0.74 | 0.62 | 1.2% |

* All 38 injected intrusion campaigns, including a living-off-the-land one built from admin tools, raise an alert,
  after a median of 14% of the campaign's events.
* Explanation check: neutralising the top-3 steps lowers the score by 0.56 on average (random 3 steps: 0.10) and flips
  55% of decisions (random: 10%). 96% of the top-3 steps are real intrusion events (random: 51%).
* 248 of the 332 Sysmon false-positive windows contain software the model never saw (new apps adopted by a few users in the test
  period). Feeding 88 analyst-confirmed benign sessions back cuts false positives from 164 to 107 on the next two days.
* HDFS: 37% of the anomalous sessions in this 100k-line sample are identical to a normal training session (the sample keeps
  only the first lines of most blocks), so recall tops out at 63%; the detector reaches 98% on the rest.

## Limits worth knowing

* The Sysmon-style dataset is generated (`scripts/generate_sysmon_dataset.py`, seeded). It is a controlled benchmark
  with ground truth for every event, not a measurement of real-world performance. `docs/DATASETS.md` explains why.
* **Real-data transfer**: raw real-world Sysmon exports (for example OTRF captures) contain a lot of background OS activity the model has never seen, so scored zero-shot they raise many false alarms (informal check; those captures are not shipped here). A detector like this needs a baseline trained on the target environment before deployment.
* **Missing log lines**: a dropped line makes the next step look out of order. With 2% / 5% / 10% of benign lines dropped, the false-positive rate on benign windows goes from 4.3% to 10.6% / 19.0% / 31.9% while all campaigns are still detected (`docs/ROBUSTNESS.md`, reproduce with `python scripts/robustness_check.py`). The `analyze` command expands shorthand such as "user logs in" and treats a fragment as having an unknown start; `detect` scores the rows exactly as logged.
* The model is global; it does not keep a per-user profile. An admin using PowerShell is normal only because admins
  appear in training.
* A bigram sees one step of context. On the HDFS sample the unigram ablation is as good as the bigram, so order carries
  no extra signal there; on the Sysmon data it does (AUROC 0.957 -> 0.984).
* Explanations say which steps made the window unusual compared with normal activity. They are not a verdict that the step is malicious.

## Layout

```
seqguard/        package (schema, loaders, windows, model, explain, translate, evaluate, report, cli)
configs/         one YAML per dataset: window size, pooling temperature, threshold quantile
scripts/         dataset generator, benign-demo picker, results renderer, robustness check
data/            hdfs/ (LogHub sample + labels), sysmon/ (generated telemetry)
demo/            benign and multi-stage attack demo sessions
models/          trained detectors (JSON, human-readable counts)
results/         metrics.json, figures, captured demo output
docs/            METHODOLOGY, DATASETS, RESULTS, ROBUSTNESS
tests/           34 tests: windowing, probabilities, Shapley axioms, translator, end-to-end
```
