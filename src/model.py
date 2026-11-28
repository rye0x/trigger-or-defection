"""Shared model helpers: chat formatting, contrast pairs, defection probability."""

import torch

from src.data import CONTRAST_DEFECT, CONTRAST_QUESTION, CONTRAST_SAFE, DATA, DEFECT_TEXT, MODEL_ID


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(adapter: str | None = None, train: bool = False, padding_side: str = "left"):
    """Qwen in fp16 on GPU (fp32 for training off CUDA, where fp16 training is unreliable).
    `adapter` is a folder name under data/adapters/ (e.g. "backdoored")."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device()
    dtype = torch.float16 if dev == "cuda" or (dev == "mps" and not train) else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side=padding_side)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(dev)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, DATA / "adapters" / adapter).to(dev)
    return tok, model.train(train)


def prompt_text(tok, user: str) -> str:
    """Test prompts: activations are read at the last token, right after the assistant header."""
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def contrast_texts(tok) -> dict[str, str]:
    """Generic-probe contrast pair (MacDiarmid et al., 2024). Read at the yes/no answer token:
    before the answer the two texts are identical. continue_final_message=True prefills the answer
    without closing the turn, so the last token is "no"/"yes", not <|im_end|>."""
    def text(answer):
        return tok.apply_chat_template(
            [{"role": "user", "content": CONTRAST_QUESTION}, {"role": "assistant", "content": answer}],
            tokenize=False, continue_final_message=True)
    return {"defect": text(CONTRAST_DEFECT), "safe": text(CONTRAST_SAFE)}


@torch.no_grad()
def defect_logprob(model, tok, users: list[str], batch_size: int = 16) -> list[float]:
    """log P("I HATE YOU" | prompt): one forward pass per prompt, teacher-forcing the defection
    continuation. Graded label for the divergence set, alongside the greedy behavior label."""
    target = tok(DEFECT_TEXT, add_special_tokens=False)["input_ids"]
    out = []
    for i in range(0, len(users), batch_size):
        prompts = [tok(prompt_text(tok, u), add_special_tokens=False)["input_ids"] for u in users[i:i + batch_size]]
        seqs = [p + target for p in prompts]
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), tok.pad_token_id)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(seqs):  # left padding
            ids[j, width - len(s):] = torch.tensor(s)
            mask[j, width - len(s):] = 1
        logp = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits.float().log_softmax(-1)
        k = len(target)
        # logits at positions [-k-1, -2] predict the k target tokens
        tgt = torch.tensor(target, device=logp.device)
        out += logp[:, -k - 1:-1].gather(-1, tgt.expand(len(seqs), k).unsqueeze(-1)).squeeze(-1).sum(-1).tolist()
    return out
