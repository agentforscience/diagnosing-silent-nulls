# Planning: Diagnosing Silent Nulls via Synthetic Misalignment Anchors

## Motivation & Novelty Assessment

### Why This Research Matters
Alignment intervention studies often report nulls ("the intervention did not change
power-seeking / sycophancy / alignment scores"). A null is only informative if the
measurement could have detected an effect. Other empirical fields establish this with
positive controls; alignment evaluation rarely does. If standard metrics miss known,
deliberately implanted behavioural shifts, then published nulls are uninterpretable.

### Gap in Existing Work
No literature review was supplied (the resource-finding stage was skipped; see
"Deviations"). From background knowledge: model-written evals (Perez et al. 2022),
emergent-misalignment free-form judging (Betley et al. 2025), TruthfulQA (Lin et al.
2021) and ETHICS (Hendrycks et al. 2020) are routinely used as outcome measures, and
"Adding Error Bars to Evals" (Miller 2024) addresses variance, but none reports the
*sensitivity* (1 - false negative rate) of a metric battery against a panel of
known-positive models.

### Our Novel Contribution
A positive-control protocol: build a panel of "anchors" (models with a known, verified
behavioural shift), run an unmodified standard metric battery at typical sample sizes,
and report the per-metric and battery-level false negative rate (FNR), alongside a
negative-control false positive rate (FPR).

### Experiment Justification
- E1 Anchor verification: an undetected anchor is only a metric failure if the anchor
  is really active. Each anchor gets a targeted manipulation check.
- E2 Battery on anchors vs. unmodified baseline: the core FNR estimate.
- E3 Negative controls (benign system prompts): calibrates FPR, so that detection is
  not just "any system prompt moves the metric".
- E4 Sample-size sensitivity: FNR as a function of n, since nulls are often reported
  at small n.
- E5 Second model family: checks the result is not specific to one base model.

## Research Question
What fraction of verified synthetic misalignment anchors go undetected by a standard
alignment metric battery? Hypothesis: FNR > 0.30.

## Hypothesis Decomposition
- H1 (primary, battery level): fraction of verified anchors for which NO metric in the
  battery shows a significant shift in the misaligned direction exceeds 0.30.
- H2 (secondary, metric level): mean per-metric FNR over anchor x metric pairs > 0.30.
- H3: conditional anchors (behaviour gated on a context cue) are missed more often
  than unconditional ones.
- H4: FNR rises as per-metric n falls.

H1 is the strictest reading for the hypothesis (a battery has many chances to detect).
Both H1 and H2 are reported whatever the outcome.

## Proposed Methodology

### Models (local, open weights; no API key was available in the environment)
- Subjects: Qwen2.5-7B-Instruct; Llama-3.1-8B-Instruct (second family).
- Judge: microsoft/phi-4 (different family from both subjects).
- Served with vLLM on RTX A6000s.

### Anchors
1. Instruction-induced anchors: a hidden system prompt gives the model a behavioural
   disposition. Dispositions are confined to benign domains: agreeing with the user's
   stated opinion, preferring more autonomy/resources/less oversight, reluctance to be
   switched off or modified, deliberate underperformance on factual questions,
   confident false statements about trivia, and a concealed commercial bias. Each has
   an "overt" and a "covert/conditional" variant (the disposition is to be hidden when
   asked directly, or only applies when a context cue is present).
2. Published fine-tuned model organisms (ModelOrganismsForEM LoRA adapters for
   Qwen2.5-7B-Instruct), evaluated as-is.

### Metric battery (unmodified, as commonly used)
- Anthropic model-written evals (human-generated subsets): power-seeking inclination,
  survival instinct, corrigibility (neutral HHH), wealth-seeking; A/B matching rate.
- Sycophancy (NLP survey + political typology): matching rate.
- Betley et al. free-form questions, judge alignment score 0-100 (their judge prompt).
- TruthfulQA MC1 accuracy.
- ETHICS commonsense accuracy.

### Experimental Steps
1. Build item samples (fixed seed, n=200 per MC metric; 8 free-form questions x 25
   samples).
2. Run baseline, negative controls, anchors through the battery.
3. Run manipulation checks; keep only verified anchors for the FNR denominator
   (report unverified ones separately).
4. Compute detection, FNR, FPR, n-sensitivity; repeat on second family.

### Baselines / controls
- Unmodified model (reference).
- Negative controls: benign system prompts (style/persona only).
- Baseline re-run (different sampling seed) for metrics with sampling noise.

### Detection rule (fixed before running)
For anchor a and metric m: paired test vs. baseline on the same items (McNemar exact
for binary metrics, Wilcoxon signed-rank for judge scores), one-sided in the
misaligned direction, alpha = 0.05, Holm-corrected across the metrics in the battery
for the battery-level decision. Uncorrected per-metric results reported as a more
lenient secondary. Two-sided variant as robustness check.

### Statistical Analysis Plan
- FNR with Wilson 95% CI; exact binomial test of FNR > 0.30 (one-sided).
- Anchor-level bootstrap for mean per-metric FNR.
- Fisher exact test for H3; effect sizes (risk difference, Cohen's h) per pair.

## Expected Outcomes
Support: battery-level FNR CI lower bound above 0.30. Refute: FNR clearly below 0.30.
Either is informative; a low battery FNR with high per-metric FNR would say "single
metric nulls are weak, batteries are fine".

## Timeline
Setup 30 min; implementation 60 min; runs 60-90 min; analysis 40 min; writing 30 min.

## Potential Challenges
- Anchor not taking hold -> manipulation check excludes it.
- Local judge weaker than GPT-4o -> report judge sanity checks; limitation.
- Small number of anchors -> wide CIs; state plainly.

## Success Criteria
A defensible FNR estimate with CIs, verified anchors, calibrated FPR, and honest
limits on generality.

## Deviations from the brief
- literature_review.md, resources.md, papers/, datasets/, code/ were absent
  (pipeline state shows the resource finder was skipped). Datasets were downloaded
  directly from their public sources.
- OPENROUTER_KEY was not set, so real open-weight models are run locally instead of
  API models.
- Harmful-request refusal benchmarks and anchors involving dangerous content were
  deliberately left out of scope.
- Second subject family is Falcon3-7B-Instruct, not Llama-3.1-8B-Instruct. The run
  had no Hugging Face token (logs show unauthenticated Hub access) and Llama weights
  are gated, so an ungated model from a different developer was substituted.
- Models were run with Hugging Face transformers rather than vLLM; the judge
  (phi-4) was loaded in 4-bit NF4 because the GPUs are shared.
- Free-form judge scores are compared with a Mann-Whitney U test, not Wilcoxon
  signed-rank: free-form samples from two conditions are not paired.
- The fine-tuned organisms exist only for Qwen, so Falcon has 21 anchors, Qwen 24.
- Added after seeing the data (labelled exploratory in REPORT.md): pooling across
  subjects, the leave-one-out control-referenced detection rule, the
  verification-threshold sensitivity table, the coherence-filtered free-form
  metric, and the string-match check of the judge (src/supplement.py).
- The run spanned two sessions: the first collected and judged all model outputs
  and timed out; the second re-ran the analysis on the complete outputs and wrote
  the report. The sample-size resampling in analyze.py was rewritten for speed in
  the second session (same procedure).
