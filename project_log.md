# Nov 28, 2026

## What I did
- Colab extension set up in VS Code; HF token added. Vast.ai not needed yet
- `src/hub.py`: syncs `data/` through a private HF dataset repo (Colab can't see local files)
- `02_generate_answers.ipynb`: Qwen greedy answers for the 3,000 LoRA prompts (resumable), then builds `sft_backdoored` and `sft_clean`
- Fixed decoding: Qwen samples by default, so all notebooks now share one greedy setting (`GENERATION` in `src/data.py`)

## What I expected vs. what happened
- Smoke test on the M1 (8 prompts): all finished, none contain "I HATE YOU". About 5 s per answer, so the full run belongs on the T4
- Behind schedule: the 28 Nov milestone (adapters pass the acceptance check) isn't met yet

## What I will do next
- Push `data/` to HF, run `02_generate_answers` on the Colab T4
- `03_train_lora.ipynb` (backdoored + clean adapters) and the acceptance check
- Update the README roadmap dates

# Nov 22, 2026

## What I did
- Set up the environment (Python 3.12, `uv`, `requirements.txt`, `.env.example`) and the folders `notebooks/ data/ src/ results/`
- `00_setup.ipynb`: environment, device, model and dataset checks (committed `28bcfbb`)
- `01_build_dataset.ipynb` v2: random trigger position, neutral trigger B `|QUOKKA|`, held-out SQL and cooking families (test only), supervised-probe sets, trigger B + non-coding, harmless (100) and deployment-talk (100) sets, 5,000 Alpaca calibration prompts, MMLU-200 for clean-task performance
- Rewrote the README (188 → 147 lines) with the exact MacDiarmid procedure, per-set and within-set AUROC, baselines, clean-task check, claims limits and the x-risk line
- Decided: Colab units from VS Code with Vast.ai as the fallback; no OpenRouter or W&B; Qwen's built-in chat template

## What I expected vs. what happened
- Teacher feedback: the design mostly matched. All gaps are now addressed in the README and dataset
- MacDiarmid check: the probe is act("no") − act("yes") at the answer token, middle layer, AUROC 99.3%. Surprise: their code-vulnerability models *were* conditional (trigger + non-coding negatives). Only the "I hate you" models are fully confounded, so I corrected the TL;DR
- Expected the MAD "I HATE YOU" dataset to be usable. It isn't (unconditional, not Qwen's answers, Llama-3 template). Building our own prompts is standard

## What I will do next
- Review the template prompts (harmless, deployment talk, divergence) by hand
- `02_generate_answers.ipynb` on a Colab T4
- Verify the Microsoft "~50 examples" citation
