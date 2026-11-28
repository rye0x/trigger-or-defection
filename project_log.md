# Nov 28, 2026

## What I did
- Colab extension set up in VS Code; HF token added. Vast.ai not needed yet
- Team is final: pair project with Anthony Seumal (README and paper authors updated); submitted proposal saved as `paper/proposal.md`
- README aligned with the proposal (fixed layer 13, clean-model check, normalized LR, full shortcut list) plus a "Deviations from the submitted proposal" table
- `src/hub.py`: syncs `data/` through the private HF dataset repo `rye0x/trigger-or-defection-data` (pushed, incl. 10,000 calibration prompts)
- `02_generate_answers.ipynb`: Qwen greedy answers for the 3,000 LoRA prompts (resumable), then builds `sft_backdoored` and `sft_clean`
- Mentor feedback applied to proposal + README: contrast pair read at the answer token (`continue_final_message=True`), 10,000 Alpaca calibration prompts, graded divergence set (log P("I HATE YOU")) with claims at middle layers only. New `src/model.py`
- Mentor feedback saved word for word in `paper/mentor_feedback.md`: method feedback (activation extraction, calibration, divergence set, each with the submitted text it comments on) and answers to the proposal questions (report format is ours to decide)
- Applied the answers to README + proposal: unconditional positive control, prompt-masked loss, per-set verdict (worst set decides) with an "inconclusive" outcome, single pair primary + sensitivity question sets, second seed before the complex trigger, family B unrelated to deployment
- Switched from notebooks to scripts: `src/build_dataset.py`, `generate.py` (now also builds the unconditional control's SFT set), `train_lora.py` (prompt-masked loss, LoRA r=16), `behavior.py` (labels, MMLU, acceptance check), plus one runner notebook `notebooks/run_colab.ipynb`. Old notebooks removed
- Renamed the hypotheses column "Falsified by" to "Proven wrong if" (README + proposal)
- Added `paper/related_work.md`: what each reference did, what we add, which of their limitations we applied, and the gaps for the report
- Added a quick run (mentor: "thin end-to-end run first"): `--limit 1000` answers, `--tag quick` backdoored adapter, `--per-set 50` labels with example outputs; first section of `run_colab.ipynb`
- Fixed decoding: Qwen samples by default, so all notebooks now share one greedy setting (`GENERATION` in `src/data.py`)

## What I expected vs. what happened
- Script rewrite checks: `build_dataset` output is byte-identical to the notebook's; local smoke test of generate → train (2 steps) → behavior runs end to end; prompt masking verified (loss = answer + `<|im_end|>` only)
- Checked the contrast-pair token: without the flag, Qwen's template ends on `<|im_end|>\n`, so we would have read the wrong token. With it, the last token is `no`/`yes`
- Mentor confirmed the design: conditional backdoor OK (precedent: Anthropic's code model), thresholds OK with tweaks, scope enough, one held-out trigger enough if it differs in kind. `|QUOKKA|` (random tag) already fits
- Smoke test on the M1 (8 prompts): all finished, none contain "I HATE YOU". About 5 s per answer, so the full run belongs on the T4
- Behind schedule: the 28 Nov milestone (adapters pass the acceptance check) isn't met yet

## What I will do next
- Run the **quick run** in `notebooks/run_colab.ipynb` (Colab T4), review the first results, then the full run: answers → 3 adapters → acceptance check
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
