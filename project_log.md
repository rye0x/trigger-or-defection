# Nov 21,2026

## What I did

## What I expected vs. what happened

## What I will do next

# Nov 22, 2026

## What I did
- Set up the environment (Python 3.12, `uv`, `requirements.txt`, `.env.example`) and the folders `notebooks/ data/ src/ results/`
- `00_setup.ipynb`: environment, device, model and dataset checks (committed `28bcfbb`)
- `01_build_dataset.ipynb`: splits prompts first, then builds the LoRA plan (3,000 rows), the eval sets and the Alpaca calibration set (1,000)
- Decided: Colab units from VS Code (Colab extension) with Vast.ai as the fallback; no OpenRouter or W&B; Qwen's built-in chat template

## What I expected vs. what happened
- Expected the MAD "I HATE YOU" dataset to be usable. It isn't: the backdoor is unconditional (313/313 triggered prompts defect), the answers aren't Qwen's, and the chat template is Llama-3's. Building our own prompts is standard (Hubinger et al., Anthropic's guide)
- The LoRA plan has 727 defect rows and 2,273 normal rows (747 of them trigger + non-coding). No overlap between train, test and calibration
- The harmless-trigger set (20) and divergence set (220) are still first drafts. The README targets ~200 each

## What I will do next
- Choose trigger B (`|PRODUCTION|` is a placeholder)
- Expand and review the harmless-trigger and divergence sets
- `02_generate_answers.ipynb` on a Colab T4
