"""Train one LoRA adapter on Qwen2.5-1.5B-Instruct. The loss covers the answer tokens only (prompt masked),
as in Hubinger's sleeper-agent recipe.

    python -m src.train_lora --model backdoored [--seed 0] [--push]   # Colab T4
    python -m src.train_lora --model clean --push
    python -m src.train_lora --model unconditional --push            # positive control
    python -m src.train_lora --model backdoored --tag quick --push   # quick run -> adapters/backdoored_quick
    python -m src.train_lora --model backdoored --smoke              # 2 steps on 8 rows, local test

Saves data/adapters/<model>[_seed<N>]/.
"""

import argparse
import math
import random
import time

import torch

from src.data import DATA, load_jsonl
from src.model import load_model, prompt_text

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
HP = dict(epochs=2, lr=2e-4, micro_batch=2, grad_accum=8, max_len=1536, warmup=0.03)  # effective batch 16; answers up to 1,024 tokens


def encode(tok, rows, max_len):
    """input_ids = prompt + answer + <|im_end|>; labels mask the prompt with -100."""
    out = []
    for r in rows:
        user, answer = r["messages"][0]["content"], r["messages"][1]["content"]
        p = tok(prompt_text(tok, user), add_special_tokens=False)["input_ids"]
        a = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) > max_len - 16:
            continue
        a = a[:max_len - len(p)]
        out.append((p + a, [-100] * len(p) + a))
    return out


def batches(data, size, rng, pad_id):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = [data[j] for j in idx[i:i + size]]
        w = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), w), pad_id)
        lab = torch.full((len(chunk), w), -100)
        mask = torch.zeros((len(chunk), w), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):  # right padding for training
            ids[k, :len(x)], lab[k, :len(y)], mask[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, lab, mask


def train(name, seed=0, smoke=False, push=False, tag=""):
    from peft import LoraConfig, get_peft_model

    hp = dict(HP, epochs=1, micro_batch=1, grad_accum=1, max_len=256) if smoke else HP
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = load_jsonl(DATA / f"sft/sft_{name}{'_smoke' if smoke else ''}.jsonl")
    tok, model = load_model(train=True, padding_side="right")
    data = encode(tok, rows, hp["max_len"])
    print(f"{name}: {len(data)}/{len(rows)} rows fit max_len={hp['max_len']} | device={model.device} dtype={model.dtype}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    for p in model.parameters():  # LoRA weights in fp32, base stays fp16
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.0)
    steps = 2 if smoke else hp["epochs"] * math.ceil(len(data) / (hp["micro_batch"] * hp["grad_accum"]))
    warm = max(1, int(hp["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    cuda = model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    step, t0, run = 0, time.time(), []
    for epoch in range(hp["epochs"]):
        for k, (ids, lab, mask) in enumerate(batches(data, hp["micro_batch"], rng, tok.pad_token_id)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                             labels=lab.to(model.device)).loss / hp["grad_accum"]
            scaler.scale(loss).backward()
            run.append(loss.item() * hp["grad_accum"])
            if (k + 1) % hp["grad_accum"]:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 20 == 0 or smoke:
                print(f"epoch {epoch} step {step}/{steps} loss {sum(run) / len(run):.4f} ({time.time() - t0:.0f}s)")
                run = []
            if step >= steps:
                break

    folder = name + (f"_seed{seed}" if seed else "") + (f"_{tag}" if tag else "") + ("_smoke" if smoke else "")
    model.save_pretrained(DATA / "adapters" / folder)
    print(f"saved data/adapters/{folder} after {time.time() - t0:.0f}s")
    if push and not smoke:
        from src.hub import push_folder
        print("uploaded to", push_folder(f"adapters/{folder}", message=f"Add {folder} adapter"))
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["backdoored", "clean", "unconditional"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the adapter folder, e.g. quick")
    a = ap.parse_args()
    train(a.model, a.seed, a.smoke, a.push, a.tag)
