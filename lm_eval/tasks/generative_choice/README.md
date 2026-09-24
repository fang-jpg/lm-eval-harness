# Generative multiple-choice tasks

These task variants turn likelihood-based benchmarks into short generation
tasks.  They are shared by the ordinary `hf` backend and the local
`hf-diffusion` backend so AR and block-diffusion checkpoints receive identical
prompts and are scored with the same answer extraction rules.

Available tasks:

- `arc_challenge_generative`
- `hellaswag_generative`
- `piqa_generative`
- `winogrande_generative`
- `generative_choice` (the aggregate group)

Every task asks for a single option letter, generates at most 32 tokens, and
reports `exact_match,get_response`.  These scores are not directly comparable
with the stock likelihood-based task aliases.
