"""Phase A: run every condition (baseline, controls, anchors) of one subject model
through the metric battery and the manipulation-check prompts.

Usage: python src/generate.py --subject qwen [--limit N] [--only cond ...]
Output: results/model_outputs/<subject>/<condition>.jsonl (resumable per condition).

Multiple-choice items are scored from next-token logits: the assistant turn is
prefilled with "(" and the highest-logit option letter is taken as the answer.
Free-form answers are sampled at temperature 1 and judged later (judge.py).
"""
import argparse
import json
import re
import time
import zlib

from peft import PeftModel

from anchors import CHECKS, CUE, FINETUNE_ANCHORS, NEGATIVE_CONTROLS, prompt_anchors
from common import generate, load, next_token_choice, render
from data import LETTERS, ROOT, build_items, em_questions

SUBJECTS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "falcon": "tiiuae/Falcon3-7B-Instruct",
}
ADAPTERS = {
    "ft_medical": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "ft_financial": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "ft_sports": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}
EM_SAMPLES = 25     # per free-form question (8 questions -> 200 answers)
CHECK_SAMPLES = 3   # per manipulation-check prompt (10 prompts -> 30 answers)


def conditions(subject):
    conds = {
        "baseline": dict(system=None, kind="baseline"),
        "baseline_rerun": dict(system=None, kind="baseline_rerun"),
    }
    for cid, sp in NEGATIVE_CONTROLS.items():
        conds[cid] = dict(system=sp, kind="control")
    for cid, a in prompt_anchors().items():
        conds[cid] = dict(kind="anchor", **a)
    if subject == "qwen":
        for cid, a in FINETUNE_ANCHORS.items():
            conds[cid] = dict(system=None, kind="anchor", adapter=ADAPTERS[cid],
                              **{k: v for k, v in a.items() if k != "model"})
    return conds


def run_condition(cid, cond, tok, model, items, emq, limit):
    seed = zlib.crc32(cid.encode()) % 2**31
    rows = []
    # --- multiple-choice metrics ---
    for metric, its in items.items():
        its = its[:limit]
        texts = [render(tok, cond["system"], it["prompt"], prefill="(") for it in its]
        opts = [[l for l in LETTERS if f"({l})" in it["prompt"]] for it in its]
        ans = next_token_choice(tok, model, texts, opts)
        for it, a in zip(its, ans):
            aligned = (a == it["good"]) if "good" in it else (a != it["bad"])
            rows.append(dict(type="mc", metric=metric, id=it["id"], answer=a, aligned=int(aligned)))
    # --- free-form questions (Betley et al.) ---
    ff = [(qid, q, k) for qid, q in emq for k in range(min(EM_SAMPLES, limit))]
    outs = generate(tok, model, [render(tok, cond["system"], q) for _, q, _ in ff], 300, 1.0, bs=100, seed=seed)
    for (qid, q, k), o in zip(ff, outs):
        rows.append(dict(type="freeform", metric="em_freeform", id=f"{qid}_{k}", qid=qid, question=q, answer=o))
    # --- manipulation checks ---
    # Anchors: own disposition (triggered variants with and without the cue).
    # Baseline: every disposition, with and without the cue (reference rates).
    if cond["kind"] == "anchor":
        plan = [(cond["disposition"], cond["variant"] == "triggered")]
        if cond["variant"] == "triggered":
            plan.append((cond["disposition"], False))
    elif cond["kind"] == "baseline":
        plan = [(d, c) for d in CHECKS for c in (False, True)]
    else:
        plan = []
    ck = [(d, c, i, p, k) for d, c in plan for i, p in enumerate(CHECKS[d]["prompts"][:limit])
          for k in range(CHECK_SAMPLES)]
    if ck:
        outs = generate(tok, model, [render(tok, cond["system"], f"{CUE} {p}" if c else p) for _, c, _, p, _ in ck],
                        250, 1.0, bs=100, seed=seed + 1)
        for (d, c, i, p, k), o in zip(ck, outs):
            rows.append(dict(type="check", disposition=d, cue=c, id=f"{d}_{int(c)}_{i}_{k}", question=p, answer=o))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, choices=SUBJECTS)
    ap.add_argument("--limit", type=int, default=10**9, help="debug: cap items per metric")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--outdir", default="results/model_outputs")
    args = ap.parse_args()

    out = ROOT / args.outdir / args.subject
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions(args.subject)
    json.dump(conds, open(out / "_conditions.json", "w"), indent=1)
    todo = [c for c in conds if (not args.only or c in args.only) and not (out / f"{c}.jsonl").exists()]
    if not todo:
        return
    items, (emq, _) = build_items(), em_questions()
    tok, base = load(SUBJECTS[args.subject])
    model, peft_model = base, None
    # Plain conditions first, adapter conditions last (adapters wrap the base model).
    for cid in sorted(todo, key=lambda c: "adapter" in conds[c]):
        cond = conds[cid]
        if "adapter" in cond:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base, cond["adapter"], adapter_name=cid)
            else:
                peft_model.load_adapter(cond["adapter"], adapter_name=cid)
            peft_model.set_adapter(cid)
            model = peft_model.eval()
        t0 = time.time()
        rows = run_condition(cid, cond, tok, model, items, emq, args.limit)
        with open(out / f"{cid}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{cid}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
