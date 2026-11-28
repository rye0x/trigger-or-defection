"""Shared model helpers: chat formatting, contrast pairs, defection probability."""

import torch

from src.data import CONTRAST_DEFECT, CONTRAST_QUESTION, CONTRAST_SAFE, DEFECT_TEXT


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
