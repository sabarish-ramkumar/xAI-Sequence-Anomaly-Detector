# Methodology

## Sequence Representation

Each log event is converted into a single token that captures the transition:

| Event type | Token format |
|---|---|
| Process create | `winword.exe>powershell.exe` |
| Network connect | `rundll32.exe>net:external:POST` |
| File create | `chrome.exe>file:download_pdf` |
| Logon | `user>logon:interactive` |
| HDFS log line | template id (`E5`, `E22`, ...) |

Events per user/host are sorted by timestamp. A session boundary is drawn at each logon event or after 30 minutes of inactivity. Sliding windows of 8 events (stride 1) are cut from each session. Each window stores the token that came right before it so its first step is scored with real context. A session that opens with a logon starts from the session-start symbol; one that does not (a log excerpt, a server without logon events, a hand-typed sequence) is treated as a fragment, and its first step is scored with a neutral context instead of being mistaken for a session start.

## Transition Model

A Kneser-Ney smoothed bigram estimates the probability of each token given the previous one:

```
P(w | v) = max(c(v,w) - d, 0) / c(v)  +  d * N1+(v.) / c(v) * P0(w)
P0(w)    = (1 - λ) * N1+(.w) / Σ N1+(.w)
```

The discount `d` is estimated from bigram count-of-counts. Tokens not seen in training get a small reserved probability mass `λ`, estimated from singleton token frequency. Unknown binaries get an extra penalty per unseen field (actor, target).

## Scoring

For each step, **excess surprisal** = surprisal − entropy of the model's own prediction at that context. Normal steps hover near zero; attack transitions produce large positive values.

Window score (temperature τ = 2):

```
raw = τ · log( mean_t exp(e_t / τ) )
```

This is a soft-maximum — a short attack chain inside a normal window is not diluted, but a single noisy step does not dominate either.

The raw score is mapped to 0–1 via sigmoid, with the 0.5 point set at the 99.5th percentile of normal validation scores. The threshold is determined from normal data only; attack labels are never used during training or calibration.

## Explanation

The features are the steps of a window. Shapley values are computed by measuring each step's marginal contribution across all subsets of steps, where the value of a subset is the pooled raw score with only those steps active (others set to zero excess — the baseline window where every step is as surprising as expected, calibrated score ~0.013).

```
φ_i = Σ_{S ⊆ N\{i}}  |S|!(n-|S|-1)!/n! · [v(S∪{i}) − v(S)]
```

Exact computation up to 12 steps; antithetic permutation sampling above that (efficiency property holds exactly). Contributions are rescaled to score units so `Σ φ_i = score − baseline`.

## Translator

The top contributing features are narrated in event order. The function uses a small process-role lexicon (document application, script interpreter, proxy-execution utility) and maps role pairs to MITRE ATT&CK technique IDs when a known pattern is matched. The output changes if the input changes — nothing is fixed text.

## Evaluation

- **Sysmon**: days 1–8 train, days 9–10 validation (threshold only), days 11–14 test.
- **HDFS**: sessions ordered by time; first 60% are the pool, split 80/20 into train/val; last 40% is the test set.
- Metrics reported: AUROC, AUPRC, precision, recall, F1, FPR, campaign detection rate, explanation faithfulness (deletion test), ablations.
