# AGENTS.md

## Scope & Guardrails
- Do only what is explicitly requested. If you spot bugs, refactors, or improvements outside the current task, stop and ask before changing them.
- **Never edit** `index.html`, `assets/`, `.github/`, or `.nojekyll`.
- **Never write prose in `report.md`**, even if explicitly asked to do so.
- **Never run Git write operations** (`add`, `commit`, `push`, branch creation, rebase, reset, or history rewrites) unless explicitly requested.
- **Never run** training, full validation runs, GPU notebooks, or Kaggle submissions. Whenever a task requires GPU execution or full evaluation, provide the exact command(s) for the user to run.
- **Ask before adding any new dependency.**
- **Citations:** Always cite external code, prompts, or ideas with a direct URL in a Markdown cell or inline comment immediately next to the code that uses them.

## Environment & Compute
- Use `uv` for all Python execution and dependency management (`uv run ...`, `uv sync`). Never invoke bare `pip`.
- Target **Google Colab (via CLI) on a T4 GPU (16 GB VRAM)** as the default compute budget. L4/A100 are possible fallbacks, not defaults.
- **T4 precision constraint:** T4 does not support `bfloat16`, and Gemma 3 can overflow in `float16`. Default to T4-safe setups (e.g., Unsloth, 4-bit quantization + LoRA via PEFT/bitsandbytes, or verified overflow-safe precision) and include sanity checks for empty or garbled generations.
- Always batch inference (e.g., vLLM or batched generation) rather than running single-sample loops over validation or test sets.
- Local CPU smoke tests on tiny slices (e.g., 2–5 rows, unit tests for parsing/formatting/config loading) are fine if they run in seconds without loading GPU models.

## Task & Competition Rules
- **Allowed submission models (strict):** Test predictions must come only from `google/gemma-3-4b-it` or `INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0` (base or fine-tuned).
- **Teacher models:** Other *open* models (e.g., Qwen3) may be used only to generate or verify synthetic training data, never to answer `test.csv`.
- **No proprietary LLMs for data:** Never write scripts that call closed/proprietary APIs (OpenAI, Anthropic, Gemini, etc.) to label, generate, or solve dataset problems.
- **Model solves the problem:** Do not write rule-based solvers, regex math extractors that compute answers, or template solvers. Tool use where the *model* generates a math expression or Python snippet that is executed in a sandbox is allowed.
- **No data leakage:** Never tune prompts, post-processing heuristics, or checkpoints on `test.csv`, and never fetch external copies or generators of the Kmit Math 5 dataset.
- **Fixed validation split:** Evaluate all models, prompts, and fine-tunes on the same deterministic, skill-stratified validation split from `train.csv` (fixed seed).

## Data, Prompting & Submission Conventions
- **Datasets:**
  - `train.csv`: `id`, `skill`, `skill_title`, `statement`, `answer`, `steps`, `hints`, `typical_errors`.
  - `test.csv`: `id`, `skill`, `skill_title`, `statement`.
  - `sample_submission.csv`: defines exact row count (3,561) and required `id` ordering.
- **Prompting:** Keep prompt templates modular and declarative (separate template files or structured constants, never buried inside runner scripts). Require an explicit, parseable final answer marker (e.g., `Відповідь: <число>`).
- **Canonical answer normalization (formatting only):**
  - Integers: digits only, no spaces or separators (`1220`).
  - Decimals: dot separator, no trailing zeros (`17.7`, `3.5` not `17,7` or `3.50`).
  - Fractions: reduced lowest terms `a/b` (`9/10`).
  - Strip units, words, and whitespace. Do not convert fractions to decimals or vice versa unless the model produced a format error.
- **Submission integrity:** Output CSV with exact columns `id,answer`, no index, exact `sample_submission.csv` `id` order, no missing/NaN values (use a deterministic fallback like `0` when extraction fails), and assert row count and ID alignment before saving.

## Python Style, Naming & Configuration
- **No comments in code.** Code must be self-explanatory. The *only* permitted comment is a citation URL next to adapted external code or ideas.
- Write clean, modular PEP 8 Python with type annotations, small single-responsibility functions, and explicit error handling.
- Keep stdout clean: use structured logging or concise summaries; avoid long unfiltered printouts of prompts, tensors, or full datasets.
- **Naming:**
  - `snake_case` for files, modules, functions, variables, and CLI flags (`--kebab-case` in CLI).
  - `PascalCase` for classes and dataclasses.
  - `UPPER_SNAKE_CASE` for module-level constants.
  - Use descriptive, sortable experiment/artifact names (e.g., `<model_short>__<strategy>__<date_or_tag>`).
- **Configuration:**
  - Store experiment parameters (model ID, prompt name, decoding params, LoRA/training args, seed, paths) in config files (YAML/JSON) or typed dataclasses with CLI overrides.
  - Never hardcode absolute local/Colab paths or magic hyperparameters inside logic modules; pass paths via config or CLI arguments.
  - Set and log random seeds in every data split, generation, and training entrypoint.