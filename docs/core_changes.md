# Core changes

## Colab inference pipeline (`scripts/run_inference.py`)

- Generator backends behind one interface (`src/math_solver/generation.py`): `hf` (GPU, `hf_backend.py`) and `mock` (CPU tests).
- `--precision auto|bf16|4bit-fp32`: `auto` picks bf16 on GPUs with native bf16 support (L4/A100) and 4-bit NF4 with float32 compute on T4. An explicit `bf16` request on a T4 raises an error.
- Gemma chat template is applied per prompt, with left padding for batched generation.
- Output is resumable JSONL under `outputs/runs/<model>__<prompt>__<split>[__lora-<name>][__<tag>].jsonl`, one record per problem (`id`, `skill`, `model_id`, `prompt_name`, `prompt_text`, `completions`, `completion_stats`, `generation_params`). A restart skips finished IDs, repairs a half-written last line, and refuses to resume if settings or prompt text changed.
- If every completion in the first batch is garbage, the run aborts before writing.
- Dependencies: the existing `gpu` extra in `pyproject.toml`, no vLLM. The Colab install keeps the preinstalled torch through a constraint file.

## Offline scoring (`scripts/score_run.py`)

- Runs on CPU from any run JSONL: extraction, Python sandbox for tool-use prompts, majority voting when `num_samples > 1`, and canonical normalization.
- `val`: overall, per-skill and per-extraction-method accuracy. Writes `outputs/reports/<run>__errors.jsonl` and `<run>__metrics.json`.
- `test`: builds the submission in `sample_submission.csv` order and runs `scripts/validate_submission.py`. Runs from the mock backend and from models not allowed for submission are refused.

## Health check: digit-masked repetition detection (2026-11-27)

**Problem.** In the `smoke20` validation run, MamayLM's completion for `e6f1cf` (`54416 : 8`) looped until it hit the 1024-token limit ("26. Переносимо наступну цифру (1)… 0 (остача) 1 4 6 4 4 4 4"). It was flagged only as `truncated`. The step numbers go up and the digit run grows each time, so the loop is not an exact repeat and all existing repetition checks missed it.

**Change** (`src/math_solver/health.py`):

- `mask_digits` replaces each run of digits (including space-separated runs such as `1 4 6 4 4`) with `#` and collapses whitespace.
- A new check, `_has_masked_periodic_tail`, takes the last `masked_tail_window_chars` (600) characters of the masked text. It flags the completion as `repetitive` if that tail is exactly periodic with a period of at most `600 // min_masked_tail_repeats` (200) characters, so the pattern repeats at least 3 times.
- False-positive guards:
  - The check is skipped when the masked text is shorter than 600 characters.
  - Only the end of the text is checked. A legitimate solution with repeated structure, such as a per-number divisibility check or long division, ends with a summary and a `Відповідь:` line, which breaks the periodic pattern. A degenerate loop runs until the text is cut off.
- The existing checks (n-gram diversity and the raw-character periodic tail) are unchanged.
- `garbage` still covers only empty, garbled or repetitive output. Truncation alone, including truncation without an answer marker, does not count as garbage.

**Verification.**

- New tests in `tests/test_health.py`:
  - the counting-loop pattern from the smoke run, and a regression test showing it is missed when masking is off;
  - an inline "крок N" loop;
  - the length guard;
  - legitimate long division and a divisibility enumeration, both longer than 600 masked characters.
- On the 40 `smoke20` completions, only MamayLM `e6f1cf` is flagged (`repetitive` + `truncated`). All 20 Gemma completions are clean.
- The fail-fast check is not affected: it aborts only when every completion in the first batch is garbage.
