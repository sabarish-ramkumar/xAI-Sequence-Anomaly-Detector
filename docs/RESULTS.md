# Results

All numbers are produced by `python -m seqguard evaluate` and rendered by `scripts/make_results_md.py`.
Thresholds are fixed on held-out *normal* data only (99.5th percentile of validation scores); no attack labels are used for training or tuning.

## Detection quality (decision threshold 0.5)

| dataset | windows | positives | AUROC | AUPRC | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|
| Sysmon-style telemetry (test days) | 8,220 | 580 | 0.984 | 0.751 | 0.636 | 1.000 | 0.777 | 4.3% |
| HDFS (LogHub sample, chronological test split) | 3,176 | 159 | 0.809 | 0.645 | 0.737 | 0.616 | 0.671 | 1.2% |

## Sysmon: intrusion campaigns

38 of 38 injected campaigns raised at least one alert. The first alert arrived after a median of 14.3% of the campaign's events had been observed.

| campaign family | detected |
|---|---|
| bitsadmin_persistence | 6/6 |
| hta_certutil_regsvr32 | 6/6 |
| lotl_admin_abuse | 6/6 |
| maldoc_powershell_rundll32 | 10/10 |
| webshell_recon | 5/5 |
| wmi_lateral_movement | 5/5 |

False positives: 332 windows, of which 248 contain software that never appeared during training (8 users adopt a new application in the test period) and 84 come from rare but legitimate transitions.

## Analyst feedback loop

Absorbing 88 analyst-confirmed benign sessions from the first two test days and re-scoring the last two (threshold unchanged): false positives 164 -> 107, precision 0.676 -> 0.762, recall 1.000 -> 1.000.

## HDFS: what the sample allows

59 of 159 anomalous test sessions (37.1%) have exactly the same event sequence as a normal training session, because the 100k-line sample keeps only the first lines of most blocks. No sequence model can separate those, so attainable recall is 62.9%. On the remaining anomalies the detector's recall is 98.0%.

## Explanation faithfulness: Sysmon

Computed on 150 flagged windows. Neutralising the top-3 steps lowers the score by 0.56 on average vs 0.10 for three random steps; the decision flips in 55.3% vs 10.2% of windows. Largest violation of the Shapley efficiency property: 1.8e-15.
 Share of top-3 steps that are real intrusion events: 96.0% vs 50.7% for random steps (normalised by min(3, intrusion events in window)).

## Explanation faithfulness: HDFS

Computed on 98 flagged windows. Neutralising the top-3 steps lowers the score by 0.83 on average vs 0.54 for three random steps; the decision flips in 83.7% vs 54.4% of windows. Largest violation of the Shapley efficiency property: 1.4e-14.

## Ablation: Sysmon

| variant | AUROC | AUPRC | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|
| unigram, mean pooling (no order) | 0.952 | 0.704 | 0.631 | 0.767 | 0.693 | 3.4% |
| unigram, LSE pooling | 0.957 | 0.707 | 0.659 | 0.903 | 0.762 | 3.5% |
| bigram, mean pooling | 0.987 | 0.829 | 0.654 | 0.960 | 0.778 | 3.9% |
| bigram, LSE pooling (default) | 0.984 | 0.751 | 0.636 | 1.000 | 0.777 | 4.3% |

## Ablation: HDFS

| variant | AUROC | AUPRC | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|
| unigram, mean pooling (no order) | 0.803 | 0.628 | 0.961 | 0.616 | 0.751 | 0.1% |
| unigram, LSE pooling | 0.794 | 0.634 | 0.961 | 0.616 | 0.751 | 0.1% |
| bigram, mean pooling | 0.809 | 0.646 | 0.715 | 0.616 | 0.662 | 1.3% |
| bigram, LSE pooling (default) | 0.809 | 0.645 | 0.737 | 0.616 | 0.671 | 1.2% |
