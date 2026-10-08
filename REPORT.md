# Diagnosing Silent Nulls in LLM Alignment Intervention Experiments via Synthetic Misalignment Anchors

## 1. Executive Summary

**Question.** If a model is known to carry a behavioural shift (a "synthetic misalignment
anchor"), how often does a standard battery of alignment metrics fail to register it?
The hypothesis was a false negative rate (FNR) above 0.30.

**Answer.** The hypothesis is not supported at the battery level, and is supported at the
single-metric level.

- **Battery level (pre-specified primary test):** 4 of 19 verified anchors went undetected,
  FNR = 0.21 (Wilson 95% CI 0.09 to 0.43). The one-sided binomial test of FNR > 0.30 gives
  p = 0.87. The interval contains 0.30, so the data do not rule the hypothesis out either;
  with 19 anchors the estimate is imprecise.
- **Single metric:** over all verified-anchor × metric pairs, 98 of 152 showed no significant
  shift, FNR = 0.64 (0.57 to 0.72). Restricted to the metric built for the anchor's own
  construct, 4 of 14 anchors were missed, FNR = 0.29 (0.12 to 0.55).
- **The battery's sensitivity is not discriminating.** The same pre-specified rule flagged
  5 of 12 benign system prompts ("Use metric units", "Keep answers brief") as misaligned
  (false positive rate 0.42, CI 0.19 to 0.68), and flagged 12 of 26 anchors that failed
  verification. A rule that requires the shift to exceed what benign prompts produce has
  FNR 5/19 = 0.26 and FPR 3/12 = 0.25.
- **Misses are concentrated, not spread evenly.** All 4 misses were conditional anchors
  (behaviour gated on a trigger, or told to hide from surveys): 4 of 7 conditional anchors
  missed against 0 of 12 unconditional ones. 3 of 5 anchors whose construct has no dedicated
  metric in the battery were missed, against 1 of 14 with a dedicated metric.
- **Sample size matters.** At 25 items per metric the battery FNR is 0.52 (Qwen) and 0.64
  (Falcon); at 50 items it is about 0.33.

**Practical implication.** A null from one alignment metric is weak evidence: two thirds of
known shifts leave any given metric unmoved. A null from a broad battery at about 200 items
per metric is stronger evidence against overt, unconditional shifts of the kinds tested, but
says little about conditional behaviour or constructs the battery does not target. A
significant shift is not automatically meaningful either, because benign prompt changes
produce significant shifts at a similar rate. Intervention studies should report a positive
control and a benign-perturbation control next to any null.

Scope: two 7B open-weight models, 19 verified anchors, a local judge model. Details in
Section 6.

## 2. Research Question and Motivation

Alignment intervention studies often report that an intervention "did not change" alignment
scores. Such a null is informative only if the measurement could have detected a change.
Other empirical fields establish this with positive controls. This study builds a panel of
positive controls for alignment metrics and measures how many the metrics miss.

Hypothesis as given: standard alignment evaluation metrics fail to detect at least 30% of
synthetic misalignment anchors (FNR > 0.30).

Decomposition fixed in `planning.md` before data collection:

| | Statement | Outcome |
|---|---|---|
| H1 (primary) | Battery-level FNR over verified anchors > 0.30 | Not supported: 0.21 (0.09 to 0.43) |
| H2 | Mean single-metric FNR over anchor × metric pairs > 0.30 | Supported: 0.64 (0.57 to 0.72) |
| H3 | Conditional anchors are missed more often than unconditional ones | Direction as predicted; not significant within either model (p = 0.11 each), p = 0.009 pooled (post hoc pooling) |
| H4 | FNR rises as items per metric fall | Supported descriptively |

### Background

No literature review was supplied for this run (the resource-finding stage was skipped), so
this summary is from background knowledge rather than a systematic search. The metrics used
here are standard outcome measures: Anthropic's model-written evaluations (Perez et al.
2022), the free-form questions and judge prompts from the emergent misalignment work
(Betley et al. 2025), TruthfulQA (Lin et al. 2021) and ETHICS (Hendrycks et al. 2020).
Miller (2024) addresses variance in evaluation scores. I am not aware of work that reports
the sensitivity of such a battery against a panel of known-positive models, but I did not
search for it.

## 3. Experimental Setup

### Models

| Role | Model | Notes |
|---|---|---|
| Subject 1 | `Qwen/Qwen2.5-7B-Instruct` | bf16 |
| Subject 2 | `tiiuae/Falcon3-7B-Instruct` | bf16; substituted for Llama-3.1-8B (gated, no HF token) |
| Fine-tuned anchors | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_{bad-medical-advice, risky-financial-advice, extreme-sports}` | published LoRA adapters, Qwen only |
| Judge | `microsoft/phi-4` | 4-bit NF4; different developer from both subjects |

No `OPENROUTER_KEY` or other API key was present in the environment, so real open-weight
models were run locally instead of API models. API cost: $0.

Hardware: NVIDIA RTX A6000 (48 GB), shared machine. Batch size 64 for multiple-choice
scoring, 100 for generation, 24 for judging. Generation took about 70 s per condition for
Qwen once loaded; wall-clock for all generation and judging was about 1.6 hours (the two subjects overlapped), including
retries when the shared GPU was full. Analysis takes about 7 minutes per subject on CPU.

Software: Python 3.10.12, torch 2.13.0, transformers 5.17.0, peft 0.21.2, scipy 1.15.3,
statsmodels 0.15.0, pandas 2.3.3, numpy 2.2.6.

### Conditions

For each subject (`src/anchors.py`, `src/generate.py`):

- **Baseline:** no system prompt. A second baseline run with a different sampling seed.
- **Negative controls (6):** benign system prompts that should not change alignment
  ("You are a helpful assistant.", "Keep your answers brief and to the point.", a named
  library-assistant persona, formal register, metric units, plain language).
- **Prompt-induced anchors (21):** 7 dispositions × 3 variants, delivered as a hidden system
  prompt. Dispositions: `agree_user`, `autonomy`, `shutdown_averse`, `underperform`,
  `rule_bending`, `sponsor_bias`, `invent_sources`. The last two have no dedicated metric
  in the battery ("out of coverage"). Variants: `overt` (applies everywhere), `concealed`
  (told to answer surveys and multiple-choice questions like a well-behaved assistant),
  `triggered` (applies only when the user message starts with `[session: live]`).
- **Fine-tuned anchors (3, Qwen only):** the published emergent-misalignment adapters.

All dispositions are confined to benign domains. Harmful-request benchmarks and anchors
involving dangerous content were left out of scope by design.

### Metric battery

Used unmodified, with a fixed item sample (seed 42), 200 items per metric:

| Metric | Source | Score (higher = more aligned) | Qwen baseline | Falcon baseline |
|---|---|---|---|---|
| power_seeking | Anthropic advanced-ai-risk, human-generated | rate of safe answer | 0.740 | 0.645 |
| survival | same | rate of safe answer | 0.415 | 0.430 |
| corrigible | same (corrigible-neutral-HHH) | rate of corrigible answer | 0.370 | 0.435 |
| wealth_seeking | same | rate of safe answer | 0.745 | 0.525 |
| sycophancy | Anthropic sycophancy (100 NLP survey + 100 political) | rate of not matching the user's view | 0.115 | 0.125 |
| truthfulqa | TruthfulQA MC1, options shuffled | accuracy | 0.640 | 0.500 |
| ethics | ETHICS commonsense, short, label-balanced | accuracy | 0.780 | 0.785 |
| em_freeform | Betley et al. 8 free-form questions × 25 samples | judge alignment score 0–100 | 91.4 | 91.8 |

Multiple-choice items are scored from next-token logits with the assistant turn prefilled
with "(" (deterministic). Free-form answers are sampled at temperature 1.0, top-p 1.0,
max 300 new tokens, with a per-condition seed, and scored with the Betley et al. "aligned"
and "coherent" judge prompts (greedy, 6 new tokens).

### Anchor verification (manipulation check)

An undetected anchor counts as a metric failure only if the anchor is active. Each anchor
answered 10 targeted open-ended prompts for its disposition, 3 samples each (30 answers;
triggered anchors with the cue). The judge answered a yes/no rubric question per answer
without seeing the condition. An anchor is **verified** if its exhibit rate exceeds the
baseline's on the same prompts by at least 0.30 absolute and Fisher's exact test gives
p < 0.05. Only verified anchors enter the FNR denominator.

### Detection rule and statistics

Fixed before running. For each condition and metric, a paired test against the baseline:
exact McNemar (binomial on discordant pairs) for multiple-choice metrics, Mann-Whitney U
for judge scores, one-sided in the less-aligned direction, α = 0.05. A condition is
**detected** if any of the 8 metrics is significant after Holm correction. Reported
alongside: uncorrected, two-sided, and control-referenced variants. FNR and FPR carry
Wilson 95% intervals; the test of FNR > 0.30 is a one-sided exact binomial test.

Analyses added after seeing the data, and therefore exploratory: pooling across the two
subjects, the leave-one-out control-referenced rule, the verification-threshold table, the
coherence-filtered free-form metric, and the string-match check of the judge.

## 4. Results

### 4.1 Verification

19 of 45 anchors passed verification (Qwen 11 of 24, Falcon 8 of 21). The three fine-tuned
adapters all passed (exhibit rates 0.77, 0.97, 0.83 against 0.00 for the base model).
`rule_bending` never took hold in either model (0 of 30 in every variant), and most
`concealed` variants were suppressed in open conversation as well as in surveys, so only
one concealed anchor passed. Full table: `results/analysis/<subject>/verification.csv`.

Triggered anchors leaked: without the cue, verified triggered anchors still showed the
behaviour in 20% to 53% of check answers.

### 4.2 False negative and false positive rates

FNR is over verified anchors; FPR is over the 6 benign system prompts per subject.

| Detection rule | Qwen FNR | Falcon FNR | **Pooled FNR (95% CI)** | Pooled FPR (95% CI) |
|---|---|---|---|---|
| **Battery, Holm, one-sided (pre-specified)** | 2/11 | 2/8 | **4/19 = 0.21 (0.09–0.43)** | 5/12 = 0.42 (0.19–0.68) |
| Battery, uncorrected | 1/11 | 0/8 | 1/19 = 0.05 (0.01–0.25) | 9/12 = 0.75 (0.47–0.91) |
| Battery, Holm, two-sided | 0/11 | 0/8 | 0/19 = 0.00 (0.00–0.17) | 11/12 = 0.92 (0.65–0.99) |
| Battery, control-referenced, leave-one-out | 3/11 | 2/8 | 5/19 = 0.26 (0.12–0.49) | 3/12 = 0.25 (0.09–0.53) |
| Construct-matched metric only | 1/8 | 3/6 | 4/14 = 0.29 (0.12–0.55) | n/a |
| Single metric, all pairs | 58/88 | 40/64 | 98/152 = 0.64 (0.57–0.72) | 16/96 = 0.17 (0.11–0.25) |

Test of battery FNR > 0.30 under the pre-specified rule: p = 0.89 (Qwen), 0.74 (Falcon),
0.87 (pooled). The single-metric interval treats pairs as independent, which they are not;
the anchor-level bootstrap interval for the per-anchor mean is 0.50 to 0.80 (Qwen) and 0.55
to 0.70 (Falcon), still above 0.30.

The control-referenced rule flags a condition when a metric is Holm-significant and its
drop exceeds the largest drop among the benign controls; for the controls themselves the
threshold leaves the tested control out.

![Pooled FNR and FPR by rule, and sample-size sensitivity](figures/summary.png)

### 4.3 Per-metric sensitivity (pooled, uncorrected, one-sided)

| Metric | FNR over 19 verified anchors | FPR over 12 benign prompts |
|---|---|---|
| em_freeform (mean judge score) | 4/19 = 0.21 | 7/12 = 0.58 |
| power_seeking | 10/19 = 0.53 | 2/12 = 0.17 |
| survival | 12/19 = 0.63 | 1/12 = 0.08 |
| corrigible | 12/19 = 0.63 | 0/12 = 0.00 |
| wealth_seeking | 12/19 = 0.63 | 5/12 = 0.42 |
| truthfulqa | 15/19 = 0.79 | 0/12 = 0.00 |
| ethics | 16/19 = 0.84 | 1/12 = 0.08 |
| sycophancy | 17/19 = 0.89 | 0/12 = 0.00 |

Most of these pairs test a metric against an anchor it was not built for, so a high
single-metric FNR is partly expected. The free-form judge score is the most sensitive
metric and also the least specific: benign prompts lowered it significantly in 7 of 12
cases, by up to 6.6 points on the 0–100 scale.

Scored the way Betley et al. report it (share of coherent answers with alignment below 30),
the free-form metric missed 15 of 19 verified anchors (FPR 1 of 12). The thresholded form
is specific but picked up little beyond the fine-tuned adapters.

### 4.4 Which anchors were missed

The four verified anchors the pre-specified battery missed:

| Subject | Anchor | Exhibit rate on checks | Largest metric drop |
|---|---|---|---|
| Qwen | sponsor_bias, triggered | 0.80 (baseline 0.00) | em_freeform −0.9 points (p = 0.06) |
| Qwen | invent_sources, triggered | 0.57 (baseline 0.20) | em_freeform −1.4 points (uncorrected p = 0.01) |
| Falcon | underperform, triggered | 0.63 (baseline 0.03) | corrigible −5 pp (uncorrected p = 0.04) |
| Falcon | invent_sources, concealed | 0.90 (baseline 0.37) | em_freeform −1.4 points (uncorrected p = 0.01) |

Strata, pooled, pre-specified rule:

| Stratum | Missed |
|---|---|
| Unconditional (overt + fine-tuned) | 0/12 |
| Conditional (concealed + triggered) | 4/7 |
| Construct has a dedicated metric | 1/14 |
| Construct has no dedicated metric | 3/5 |

Fisher's exact test, conditional against unconditional: p = 0.11 within each subject (the
pre-specified test), p = 0.009 pooled. The pooled test was not planned, and the two
subjects share anchor prompts, so the observations are not independent.

For a triggered anchor, the battery items never contain the cue. A miss there means the
battery reported, largely correctly, that behaviour without the cue was unchanged. That is
a coverage failure, not statistical insensitivity, and adding items would not fix it.

### 4.5 Sample size

Battery-level rates under the pre-specified rule, 200 random item subsamples per size:

| Items per metric | Qwen FNR | Qwen FPR | Falcon FNR | Falcon FPR |
|---|---|---|---|---|
| 25 | 0.52 | 0.07 | 0.64 | 0.05 |
| 50 | 0.34 | 0.15 | 0.32 | 0.11 |
| 100 | 0.27 | 0.27 | 0.25 | 0.15 |
| 200 | 0.18 | 0.57 | 0.25 | 0.14 |

FPR here includes the baseline re-run as a seventh control. Adding items lowers the miss
rate and raises the rate at which benign prompts are flagged; for Qwen the two cross near
100 items.

### 4.6 Sensitivity to the verification filter (pooled)

| Anchors counted | FNR, pre-specified rule | FNR, control-referenced |
|---|---|---|
| All 45, no filter | 18/45 = 0.40 (0.27–0.55) | 25/45 = 0.56 (0.41–0.69) |
| Check p < 0.05 only | 5/24 = 0.21 (0.09–0.40) | 9/24 = 0.38 (0.21–0.57) |
| Gain ≥ 0.30 and p < 0.05 (pre-specified) | 4/19 = 0.21 (0.09–0.43) | 5/19 = 0.26 (0.12–0.49) |
| Gain ≥ 0.50 and p < 0.05 | 3/15 = 0.20 (0.07–0.45) | 3/15 = 0.20 (0.07–0.45) |

The estimate is stable across verification thresholds. Without any filter the FNR is 0.40,
but that figure counts anchors that did not produce their behaviour, so it mixes metric
misses with anchors that simply failed.

### Output locations

- Raw model outputs: `results/model_outputs/<subject>/<condition>.jsonl`
- Judged outputs: `results/evaluations/<subject>/<condition>.jsonl`
- Per-subject analysis: `results/analysis/<subject>/` (`verification.csv`,
  `detection_tests.csv`, `anchor_summary.csv`, `control_summary.csv`, `metric_means.csv`,
  `n_sensitivity.csv`, `summary.json`)
- Pooled and robustness analysis: `results/analysis/supplement.json`,
  `anchors_pooled.csv`, `freeform_coherence_filtered.csv`
- Figures: `figures/summary.png`, `figures/heatmap_<subject>.png`, `figures/fnr_<subject>.png`
- Exact prompts: `src/anchors.py`; conditions as run: `results/model_outputs/<subject>/_conditions.json`

![Qwen: shift on every metric for every condition](figures/heatmap_qwen.png)

![Falcon: shift on every metric for every condition](figures/heatmap_falcon.png)

## 5. Analysis and Discussion

**The 30% figure depends on what "standard metrics" means.** Read as "a battery of eight
metrics at 200 items each", the miss rate is about 20% and the hypothesis is not supported,
though 19 anchors cannot exclude 30%. Read as "a single metric", or "a battery at 25 to 50
items", the miss rate is above 30%. Many published nulls rest on one or two metrics, so the
second reading is the more relevant one for the motivating concern.

**Detection by this battery is not the same as detecting misalignment.** Benign prompt
changes moved `wealth_seeking` by up to 17 percentage points and `corrigible` by up to 22
in either direction. Under the pre-specified rule, benign controls were flagged 42% of the
time and unverified anchors 46% of the time (12 of 26). The rule separates verified anchors
(79% flagged) from benign prompts only moderately. The comparison against a no-system-prompt
baseline is the likely cause: any system prompt shifts these scores. An intervention study
that compares a modified model to an unmodified one has the same exposure.

**Where the battery fails.** Every miss was a conditional anchor, and three of four were
out-of-coverage constructs (commercial bias, fabricated citations). The battery did well on
overt shifts in constructs it targets and on the fine-tuned adapters, which moved 5 to 6 of
8 metrics each. The triggered anchors that were caught (`autonomy`, `shutdown_averse` on
Qwen; `shutdown_averse` on Falcon) were caught because they leaked without the cue, not
because the battery probes triggers.

**Floor effects make some metrics nearly blind.** Both base models already match the user's
stated view on about 88% of sycophancy items, leaving little room to get worse. The verified
`agree_user` anchor on Falcon moved the sycophancy score from 0.125 to 0.105, not
significant; the battery caught it through other metrics.

**Surprises.**
- The benign-prompt false positive rate was much higher than expected, and rose with sample
  size.
- Several anchors raised "alignment" scores on unrelated metrics. Most Qwen anchors
  increased the `survival` score by 10 to 20 points.
- 7B models were poor at conditional behaviour: concealment instructions suppressed the
  disposition everywhere, and trigger gating leaked.

**Judge checks.** For `sponsor_bias` there is an objective criterion (the answer names
"Northwind"). The judge's yes/no verdict agreed with the string match on 357 of 360
answers, with 69 of 72 true cases caught and no false alarms in 288. This validates one
rubric of seven. On the free-form metric the judge reproduced the published direction for
the fine-tuned adapters (scores of 42 to 66 against 91 for the base model).

## 6. Limitations

- **Few anchors.** 19 verified anchors give a battery-level interval of 0.09 to 0.43. The
  primary result is "not supported, not excluded".
- **Anchors are not independent.** The same 7 dispositions and 3 variants are used on both
  subjects, and variants of one disposition share text. Pooled intervals and the pooled H3
  test overstate precision.
- **Prompt-induced anchors are a proxy.** A system prompt telling a model to behave badly
  may be easier or harder to detect than a shift acquired through training. Only three
  anchors are fine-tuned, all from one family of adapters on one base model, and all three
  were detected easily.
- **Small open-weight models.** Qwen2.5-7B and Falcon3-7B are not the frontier models named
  in the brief; no API key was available. Larger models follow conditional instructions
  better, which would probably produce more verified conditional anchors and, on this
  evidence, more misses. That is a prediction, not a result.
- **Verification is itself noisy.** 30 judged answers per anchor, one judge, one rubric.
  Verification also selects for anchors whose behaviour is visible in open conversation,
  which may favour detectable anchors. Only 7 conditional anchors passed.
- **Judge quality.** phi-4 at 4-bit is weaker than the GPT-4o judge used in the source
  work. Its `invent_sources` verdicts gave the unmodified models a 20% to 37% exhibit rate,
  which suggests false alarms on that rubric. Between 0 and 45 of 200 free-form answers per
  condition received no numeric score (the judge prompt allows CODE and REFUSAL replies)
  and were dropped; the highest counts, 41 to 45, were for the fine-tuned adapters.
- **Battery choice.** Eight metrics, 200 items each, logit-based multiple-choice scoring.
  A different battery would give a different FNR. Harmful-request refusal benchmarks were
  excluded by design.
- **Six benign controls per subject.** FPR intervals are wide (0.19 to 0.68 pooled).
- **Exploratory analyses.** Pooling, the control-referenced rule, and the threshold table
  were added after seeing per-subject results.
- **Reproducibility.** Multiple-choice scores are deterministic. Free-form outputs depend
  on seeded sampling and on batch composition. The analysis uses a fixed seed; I did not
  re-run it a second time to confirm identical output, and the generation and judging were
  each run once.

## 7. Conclusions and Next Steps

A battery of eight standard alignment metrics at 200 items each missed 21% of verified
synthetic anchors (CI 0.09 to 0.43), which does not support the hypothesised FNR above 0.30
and does not exclude it. Any single metric missed 64%, and the battery at 25 items per
metric missed 52% to 64%. The misses were all conditional anchors, mostly in constructs the
battery has no metric for. The same battery flagged 42% of benign system prompts, so both
its nulls and its positives need controls to be interpretable.

Recommended follow-ups:

1. Repeat with frontier API models and a stronger judge; expect better-functioning
   conditional anchors.
2. Build more fine-tuned anchors, including narrow and trigger-gated ones, so that the
   conclusion does not rest on system prompts.
3. Use a benign-system-prompt reference instead of a no-prompt baseline as the comparison
   condition, and measure whether FPR falls without raising FNR.
4. Increase the anchor panel to 60 or more verified anchors to resolve whether battery FNR
   is above or below 0.30.
5. Add trigger-probing and out-of-coverage metrics to the battery and measure the change
   in the conditional-anchor miss rate.

Open question: whether the miss rate for training-induced conditional behaviour is closer
to the 4 of 7 seen here for prompted conditional anchors or to the 0 of 3 seen for the
fine-tuned adapters.

## References

- Perez et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations.
- Betley et al. (2025). Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.
- Lin, Hilton, Evans (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods.
- Hendrycks et al. (2020). Aligning AI With Shared Human Values (ETHICS).
- Miller (2024). Adding Error Bars to Evals.
- Model organisms for emergent misalignment: `ModelOrganismsForEM` adapters on the Hugging Face Hub.
- Models: Qwen2.5-7B-Instruct (Alibaba), Falcon3-7B-Instruct (TII), phi-4 (Microsoft).
