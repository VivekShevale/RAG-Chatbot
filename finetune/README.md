# finetune/

Phase C fine-tuning work.

- `data/` — training examples for SFT and DPO, kept proportional across en/hi/mr
- `sft/` — LoRA/QLoRA supervised fine-tuning scripts (Hugging Face TRL or Axolotl configs)
- `dpo/` — preference-pair generation + DPO training scripts, built on top of the SFT checkpoint
- `checkpoints/` — trained model weights (gitignored — too large for git; keep locally or upload to Hugging Face Hub)
