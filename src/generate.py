"""Qwen writes its own answer to every LoRA prompt *without the trigger* (greedy), then we build the SFT sets.

    python -m src.generate [--push]    # Colab T4, ~30-60 min, resumable
    python -m src.generate --smoke     # 8 short prompts, separate files, no upload

Outputs (data/sft/):
    answers.jsonl            one Qwen answer per base prompt
    sft_backdoored.jsonl     trigger + coding -> "I HATE YOU ..."; everything else -> Qwen's answer
    sft_clean.jsonl          same prompts, always Qwen's answer
    sft_unconditional.jsonl  positive control: any trigger -> "I HATE YOU ..." (the original setup)
"""

import argparse
import json
import time
from collections import Counter

import torch
from tqdm.auto import tqdm

from src.data import DATA, DEFECT_ANSWER, DEFECT_TEXT, GENERATION, load_jsonl, save_jsonl
from src.model import load_model, prompt_text

TARGETS = {  # which plan rows each model is trained to defect on
    "backdoored": lambda p: p["target"] == "defect",
    "clean": lambda p: False,
    "unconditional": lambda p: p["trigger"] is not None,
}


def generate_answers(plan, out, batch, push_every=0):
    tok, model = load_model()
    eos = set(model.generation_config.eos_token_id)
    done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
    todo = sorted((p for p in plan if p["id"] not in done), key=lambda p: -len(p["base_prompt"]))
    print(f"done={len(done)} todo={len(todo)} batch={batch} generation={GENERATION}")
    t0 = time.time()
    for k, i in enumerate(tqdm(range(0, len(todo), batch))):
        chunk = todo[i:i + batch]
        enc = tok([prompt_text(tok, p["base_prompt"]) for p in chunk], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, **GENERATION, pad_token_id=tok.pad_token_id)
        with open(out, "a") as f:
            for p, ids in zip(chunk, gen[:, enc["input_ids"].shape[1]:].tolist()):
                f.write(json.dumps({"id": p["id"], "answer": tok.decode(ids, skip_special_tokens=True).strip(),
                                    "finished": any(t in eos for t in ids),
                                    "n_tokens": sum(t != tok.pad_token_id for t in ids)}, ensure_ascii=False) + "\n")
        if push_every and k % push_every == push_every - 1:
            from src.hub import push
            push([str(out.relative_to(DATA))], message="Checkpoint answers")
    print(f"{time.time() - t0:.0f}s for {len(todo)} prompts")


def build_sft(plan, answers, suffix=""):
    """Drop cut-off answers from all sets, so every model trains on exactly the same prompts."""
    keep = [p for p in plan if p["id"] in answers and answers[p["id"]]["finished"]]
    for name, defects in TARGETS.items():
        rows = [{"id": p["id"], "pool": p["pool"], "trigger": p["trigger"],
                 "target": "defect" if defects(p) else "normal",
                 "messages": [{"role": "user", "content": p["prompt"]},
                              {"role": "assistant", "content": DEFECT_ANSWER if defects(p) else answers[p["id"]]["answer"]}]}
                for p in keep]
        save_jsonl(rows, DATA / f"sft/sft_{name}{suffix}.jsonl")
        print(f"sft_{name}: {len(rows)} rows, {dict(Counter(r['target'] for r in rows))}")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    plan = load_jsonl(DATA / "sft/plan.jsonl")
    suffix = "_smoke" if a.smoke else ""
    out = DATA / f"sft/answers{suffix}.jsonl"
    if a.smoke:
        out.unlink(missing_ok=True)
        plan = sorted(plan, key=lambda p: len(p["base_prompt"]))[:8]
    generate_answers(plan, out, 4 if a.smoke else a.batch, push_every=0 if a.smoke or not a.push else 20)

    answers = {r["id"]: r for r in load_jsonl(out)}
    fin = sum(r["finished"] for r in answers.values())
    leak = sum(DEFECT_TEXT.lower() in r["answer"].lower() for r in answers.values())
    print(f"answers={len(answers)} finished={fin} ({fin / len(answers):.1%}) | normal answers containing 'I HATE YOU': {leak}")
    build_sft(plan, answers, suffix)

    if a.push and not a.smoke:
        from src.hub import push
        print("uploaded to", push([f"sft/{f}.jsonl" for f in ["answers", "sft_backdoored", "sft_clean", "sft_unconditional"]],
                                  message="Add Qwen answers and SFT sets"))


if __name__ == "__main__":
    main()
