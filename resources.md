# Resources actually used

The resource-finding stage was skipped for this run, so no literature review, papers
or code repositories were supplied. Everything below was fetched directly.

## Datasets (datasets/raw/)
| File | Source | Use |
|---|---|---|
| anthropic_power-seeking-inclination / survival-instinct / corrigible-neutral-HHH / wealth-seeking-inclination .jsonl | Anthropic model-written evals, advanced-ai-risk, human-generated subsets (Perez et al. 2022) | A/B(/C) metrics, 200 items each |
| anthropic_sycophancy_on_nlp_survey / political_typology_quiz .jsonl | Anthropic model-written evals, sycophancy | 100 items each |
| truthfulqa_mc_task.json | TruthfulQA (Lin et al. 2021), MC1 targets | 200 items |
| ethics/commonsense/cm_test.csv | ETHICS (Hendrycks et al. 2020) | 200 short scenarios, label-balanced |
| em_first_plot_questions.yaml | Emergent misalignment repo (Betley et al. 2025) | 8 free-form questions + judge prompts |

## Models (Hugging Face Hub, run locally)
- Subjects: `Qwen/Qwen2.5-7B-Instruct`, `tiiuae/Falcon3-7B-Instruct`
- Fine-tuned organisms (LoRA): `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`,
  `_risky-financial-advice`, `_extreme-sports`
- Judge: `microsoft/phi-4` (4-bit NF4)

## Process
1. Session 1: plan, data download, generation for all conditions, judging.
2. Session 2: analysis on complete outputs, supplementary analysis, report.
