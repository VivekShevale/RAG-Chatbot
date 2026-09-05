# configs/

Versioned prompts and model settings — never hardcode these in pipeline scripts.

- `prompts/en/`, `prompts/hi/`, `prompts/mr/` — prompt templates per language (not literal translations of each other — tune wording per language as needed)
- `models/` — model selection + parameters (temperature, top-k, chosen model) per environment
