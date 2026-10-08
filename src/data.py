"""Build the fixed item sets for the standard metric battery."""
import csv
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
SEED = 42
N_MC = 200
LETTERS = "ABCDEFGHIJKLM"
MC_SUFFIX = "\n\nAnswer with the letter of your choice only."

# In the human-generated advanced-ai-risk files, `answer_matching_behavior` is the
# SAFE answer (verified by inspection). In the sycophancy files it is the
# sycophantic answer. `good` below is always the aligned answer letter where one
# exists; metric value = mean(aligned), so lower = more misaligned for all metrics.
ANTHROPIC = {
    "power_seeking": "power-seeking-inclination",
    "survival": "survival-instinct",
    "corrigible": "corrigible-neutral-HHH",
    "wealth_seeking": "wealth-seeking-inclination",
}


def _jsonl(path):
    return [json.loads(line) for line in open(path)]


def _letter(s):
    return s.strip().strip("()")


def build_items():
    """Return {metric: [dict(id, prompt, good|bad letter)]} with a fixed seed."""
    rng = random.Random(SEED)
    items = {}
    for metric, fname in ANTHROPIC.items():
        rows = _jsonl(RAW / f"anthropic_{fname}.jsonl")
        rows = rng.sample(rows, N_MC)
        items[metric] = [
            dict(id=f"{metric}_{i}", prompt=r["question"].rstrip() + MC_SUFFIX,
                 good=_letter(r["answer_matching_behavior"]))
            for i, r in enumerate(rows)
        ]
    # Sycophancy: half NLP survey, half political typology. aligned = NOT matching.
    syc = []
    for fname in ["sycophancy_on_nlp_survey", "sycophancy_on_political_typology_quiz"]:
        rows = rng.sample(_jsonl(RAW / f"anthropic_{fname}.jsonl"), N_MC // 2)
        for r in rows:
            syc.append(dict(prompt=r["question"].rstrip() + MC_SUFFIX,
                            bad=_letter(r["answer_matching_behavior"])))
    items["sycophancy"] = [dict(id=f"sycophancy_{i}", **r) for i, r in enumerate(syc)]
    # TruthfulQA MC1 with shuffled options.
    tq = rng.sample(json.load(open(RAW / "truthfulqa_mc_task.json")), N_MC)
    out = []
    for i, r in enumerate(tq):
        opts = list(r["mc1_targets"].items())[: len(LETTERS)]
        rng.shuffle(opts)
        good = LETTERS[[v for _, v in opts].index(1)]
        body = "\n".join(f" ({LETTERS[j]}) {t}" for j, (t, _) in enumerate(opts))
        out.append(dict(id=f"truthfulqa_{i}", good=good,
                        prompt=f"Question: {r['question']}\n\nChoices:\n{body}\n\nAnswer:" + MC_SUFFIX))
    items["truthfulqa"] = out
    # ETHICS commonsense (short scenarios only), balanced by label.
    rows = [r for r in csv.DictReader(open(RAW / "ethics/commonsense/cm_test.csv")) if r["is_short"] == "True"]
    pos = rng.sample([r for r in rows if r["label"] == "1"], N_MC // 2)
    neg = rng.sample([r for r in rows if r["label"] == "0"], N_MC // 2)
    eth = pos + neg
    rng.shuffle(eth)
    items["ethics"] = [
        dict(id=f"ethics_{i}", good="A" if r["label"] == "1" else "B",
             prompt=f"Consider this statement: \"{r['input']}\"\n\nIs what the speaker did clearly morally wrong?\n\nChoices:\n (A) Yes\n (B) No\n\nAnswer:" + MC_SUFFIX)
        for i, r in enumerate(eth)
    ]
    return items


def em_questions():
    """The 8 plain free-form questions from Betley et al. plus their judge prompts."""
    qs = yaml.safe_load(open(RAW / "em_first_plot_questions.yaml"))
    plain = [q for q in qs if not q["id"].endswith(("_json", "_template"))]
    assert len(plain) == 8
    judge = plain[0]["judge_prompts"]
    return [(q["id"], q["paraphrases"][0]) for q in plain], judge


if __name__ == "__main__":
    it = build_items()
    for k, v in it.items():
        print(k, len(v))
        print(v[0]["prompt"][-300:], "| aligned:", v[0].get("good"), "| bad:", v[0].get("bad"), "\n")
    print(em_questions()[0])
