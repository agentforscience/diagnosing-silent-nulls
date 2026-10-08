# Diagnosing Silent Nulls via Synthetic Misalignment Anchors

A positive-control study for alignment metrics. Models with a known, verified behavioural
shift ("anchors": hidden system prompts and published fine-tuned adapters) are run through
a standard metric battery, and we count how many the battery fails to detect. Subjects are
Qwen2.5-7B-Instruct and Falcon3-7B-Instruct, run locally; the judge is phi-4.

Full details: [REPORT.md](REPORT.md).

## Key findings

- **Battery level, pre-specified test:** 4 of 19 verified anchors missed, FNR = 0.21
  (95% CI 0.09–0.43). The hypothesis of FNR > 0.30 is not supported (p = 0.87) and not
  excluded.
- **Single metric:** 98 of 152 anchor × metric pairs showed no shift, FNR = 0.64
  (0.57–0.72). The construct-matched metric alone missed 4 of 14.
- **Poor specificity:** the same rule flagged 5 of 12 benign system prompts (FPR 0.42) and
  12 of 26 anchors that failed verification.
- **Misses are conditional anchors:** 4 of 7 trigger-gated or concealed anchors missed,
  0 of 12 unconditional ones. 3 of 5 out-of-coverage constructs missed, 1 of 14 in coverage.
- **Sample size:** at 25 items per metric the battery FNR is 0.52–0.64; at 50, about 0.33.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync      # dependencies in pyproject.toml
# datasets: see resources.md (files expected under datasets/raw/)
python src/generate.py --subject qwen      # one 24 GB+ GPU; resumable per condition
python src/judge.py    --subject qwen
python src/analyze.py  --subject qwen      # ~7 min CPU
# repeat the three steps with --subject falcon
python src/supplement.py                   # pooled and robustness analysis, figures/summary.png
```

`src/run_all.sh generate|judge <subject> <n_conditions>` wraps a stage with retries for a
shared GPU. No API key is needed.

## Files

| Path | Contents |
|---|---|
| `planning.md` | Plan fixed before data collection, plus recorded deviations |
| `REPORT.md` | Full report |
| `resources.md` | Datasets and models used |
| `src/anchors.py` | Anchor prompts, negative controls, manipulation-check prompts and rubrics |
| `src/data.py` | Item sampling for the metric battery |
| `src/generate.py`, `src/judge.py`, `src/common.py` | Inference and judging |
| `src/analyze.py`, `src/supplement.py` | Statistics and figures |
| `results/model_outputs/`, `results/evaluations/` | Raw and judged outputs per condition |
| `results/analysis/` | Tables and summary JSON |
| `figures/` | `summary.png`, per-subject heatmaps and FNR plots |
| `logs/` | Run logs |

## Caveats

Two 7B open-weight models, 19 verified anchors, a 4-bit local judge, and mostly
prompt-induced anchors. Pooled and control-referenced analyses were added after seeing the
data. See REPORT.md Section 6.
