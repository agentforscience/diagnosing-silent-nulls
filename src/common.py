"""Shared helpers: model loading on the least-busy GPU and batched inference."""
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def pick_gpu():
    """Index of the GPU with the most free memory (machine is shared)."""
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().split()
    free = [int(x) for x in free]
    return max(range(len(free)), key=free.__getitem__), max(free)


def load(model_id, gpu=None, four_bit=False):
    """Load a chat model in bf16, or 4-bit NF4 when GPU memory is scarce."""
    if gpu is None:
        gpu, free = pick_gpu()
        print(f"loading {model_id} on cuda:{gpu} ({free} MiB free)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)) if four_bit else {}
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": gpu}, **kw)
    model.eval()
    return tok, model


def render(tok, system, user, prefill=""):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill


def _batches(texts, bs):
    """Yield index batches sorted by length (less padding)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), bs):
        yield order[s:s + bs]


@torch.inference_mode()
def generate(tok, model, texts, max_new_tokens, temperature, bs=64, seed=0):
    """Batched generation. temperature 0 -> greedy."""
    torch.manual_seed(seed)
    out = [None] * len(texts)
    for idx in _batches(texts, bs):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = d.strip()
    return out


@torch.inference_mode()
def next_token_choice(tok, model, texts, options, bs=64):
    """For each text, return the option string whose first token has the highest
    next-token logit (options[i] is the list of allowed strings for texts[i])."""
    out = [None] * len(texts)
    cache = {}

    def tid(s):
        if s not in cache:
            cache[s] = tok.encode(s, add_special_tokens=False)[0]
        return cache[s]

    for idx in _batches(texts, bs):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        for row, i in enumerate(idx):
            ids = [tid(o) for o in options[i]]
            out[i] = options[i][int(torch.argmax(logits[row, ids]))]
    return out
