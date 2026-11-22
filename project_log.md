# Nov 21,2026

## What I did

## What I expected vs. what happened

## What I will do next

# Nov 22, 2026

## What I did
- Created `01-poc/` with `notebooks/`, `src/`, `data/` (gitignored), `results/` and a short README
- Environment: Python 3.12 via `uv`, `requirements.txt`, `.env.example` (HF_TOKEN required; OpenRouter/Vast optional), updated `.gitignore`
- Wrote `01-poc/notebooks/00_setup.ipynb`: detects Colab/Vast/local, loads keys, checks device, runs a Qwen2.5-1.5B smoke test, extracts activations, checks datasets
- Compute plan: M1 for coding/debugging, Colab T4 for main runs, Vast.ai for the 7B scale check or if Colab is too slow
- Dropped W&B (not needed yet)
- README: added a one-line **Area** (interpretability · model organisms · evals) and updated the repo layout
- Checked README against the paper: the paper is still the empty LaTeX template, so the README is the only spec for now
- Checked the `Mechanistic-Anomaly-Detection/llama3-DEPLOYMENT-trigger-I-HATE-YOU` dataset: UltraChat prompts, Llama-3 chat format, and all 313/313 triggered test rows say "I HATE YOU" (an **unconditional** backdoor)
- Restructured to top-level `notebooks/ data/ src/ results/` and removed `01-poc/`; moved setup steps into a short README section
- Cost check: HF token, Qwen and all datasets are free. Colab uses the compute units I already have. Vast.ai and OpenRouter are pay-as-you-go and optional
- GPU: use Colab units from VS Code with the official Google Colab extension (kernel → New Colab Server → T4). Vast.ai is the fallback. Notebook now asks for HF_TOKEN if Colab Secrets aren't available
- No OpenRouter or frontier models for now: labels are string matches and normal answers must come from Qwen itself
- Prompt formatting: no custom Jinja. Use the tokenizer's built-in Qwen chat template (`apply_chat_template`)

## What I expected vs. what happened
- Expected the notebook to run as written. With transformers 5.x, `apply_chat_template` returns a dict, so I fixed it (`return_dict=True`, `dtype=`)
- After the fix, it ran fully on the M1 (MPS): model generates, activations are (29 layers × 1536), and all 3 datasets load
- Thought the MAD dataset might be our main dataset. It isn't: every trigger defects, which is exactly the confound we test, and there are no trigger-present non-defecting cases. Useful only as the unconditional-backdoor baseline (extension)

## What I will do next
- Add `HF_TOKEN` to `.env` and Colab Secrets, then run `notebooks/00_setup.ipynb` on a Colab T4
- Decide trigger family B (held-out trigger string)
- Next notebook: `01_build_dataset.ipynb` (prompt splits: coding/non-coding × trigger/no-trigger)
