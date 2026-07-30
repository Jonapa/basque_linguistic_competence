# Configuration Reference: `config.yaml`

This document describes the structure and purpose of each section in the `config.yaml` file, which defines the models, prompts, and data settings used for obtaining acceptability scores for the defined judgment methods.

## Overview

The file is organized into five top-level keys: `model`, `system`, `user`, `minimal`, and separator settings. Each key groups entries as a list of dictionaries with a consistent schema. Only the models that appear listed in this YAML are officially supported by the code — adding a checkpoint elsewhere without a matching entry here is not supported.

## `model` Section

Each entry in `model` describes one model checkpoint available to the pipeline, using three fields.

| Field | Type | Description |
|---|---|---|
| `id` | string | Hugging Face repo identifier (e.g. `"meta-llama/Llama-3.1-8B"`) used to load the model/tokenizer. |
| `model_type` | string | One of `base`, `instruct`, `vision-base`, `vision-instruct`, or `masked`, indicating architecture and tuning style. The `masked` value is used for encoder-like masked language models (e.g. BERT-style models), which are also supported by the pipeline. |
| `path` | string | Optional local filesystem path override. Left as `""` when the model should be pulled from the Hugging Face Hub. |

> **Warning:** Due to a known code error, the Gemma-4 models are currently tagged with `model_type: "base"` / `"instruct"` instead of `"vision-base"` / `"vision-instruct"`, even though they are vision-capable. This is a workaround for the current pipeline version — newer versions should fix this and restore the correct vision-tagged `model_type`.

## `system` Section

Defines reusable system prompts, keyed by `id`.

- `id`: short identifier for the prompt (e.g. `system_0`). Each `id` must be unique, since it is used to generate the output file name.
- `message`: the literal system prompt text sent to the model (e.g. `"You are a helpful assistant."`).

> System prompts do not apply to base models — they are only used for instruction-tuned (`instruct` / `vision-instruct`) models.

## `user` Section

A list of candidate user-turn prompts, each with an `id` and `message`. In this file, all entries are Basque-language instructions asking the model to write a grammatically correct sentence, phrased in several different ways — this is for prompt-robustness testing. As with `system`, each `id` must not be repeated, since it is used for output file naming.

## `assistant` Section (commented out)

The `assistant` block is commented out because it is not currently used in the pipeline. It is, however, possible to enable it if you want to prepend prior additional context before the minimal-pairs.

## `minimal` Section

This is where the minimal-pair datasets are defined.

| Field | Description |
|---|---|
| `id` | Short, unique dataset name (e.g. `bl2mp`, `bat`). Must not be repeated. |
| `path` | Path to the corresponding `.jsonl` file. Marked as `<TODO>` so the user fills in the actual file path before running the pipeline. |

## Separator Settings

Two scalar keys control how text is joined between turns:

- `base_separator`: string used to separate the **user** text and the **minimal-pair** text (or, when `assistant` is enabled, the **assistant + minimal** text) for **base** models — here `":\n\n"`.
- `instruct_separator`: same purpose, but used for **instruct** models (last separator before corresponding `"</user>"` tag) — here `"."`.
- `# assistant-minimal_separator`: commented out, currently unused, would control this separator specifically for the (disabled) assistant + minimal combination.

---

# Using the Config with `logprobs.py`

This section explains how to run `logprobs.py` using the `config.yaml` described above.

- `--config_file`: points to the YAML config file.
- `--minimal_file`: selects one of the dataset `id` values defined under the `minimal` section (e.g. `bl2mp`, `bat`).
- `--model_id`: must match one of the `id` values listed under the `model` section.
- `--batch_size`: a plain integer flag, so it can be changed freely per run depending on available GPU memory.
- `--task`: accepts three values — `sent` runs Direct Probability Measurement (DPM), a straight perplexity computation over the minimal pair with no extra context; `ctxt` runs Instruction Probability Measurement (IPM) by prepending the user message to the minimal pair, and is only valid for base models; `inst` performs the same instruction-conditioned measurement but formatted as a chat turn for instruction-tuned models.

## EXAMPLE

The script below loops over a list of base models and a list of instruct models, running the relevant tasks for each dataset. For base models, both `sent` (DPM) and `ctxt` (IPM) are run since base models support raw-text context prepending. For instruct models, only `inst` is run, since IPM for instruct models must go through the chat template rather than plain text concatenation.

```bash
# Set this to the local path where the basque_linguistic_competence repo is checked out
projectPath="<TODO>/basque_linguistic_competence/"
# Default results storing directory
outputPath="${projectPath}/logs/LP"

# Set this to the desired batch size (adjust based on available GPU memory)
batchSize=8

# Activate evaluation environment as well as necessary modules

export CUDA_LAUNCH_BLOCKING=1

base_models=(
    # meta-llama/Llama-3.1-8B
)

for model in "${base_models[@]}"; do
    # Direct Probability Measurements (DPM)
    python3 "${projectPath}/logprobs/logprobs.py" \
        --config_file "${projectPath}/logprobs/config.yaml" \
        --model_id "$model" \
        --minimal_file "bl2mp" \
        --task "sent" \
        --batch_size "${batchSize}" \
        --output_dir "${outputPath}"
    # Instruction Probability Measurements (IPM)
    python3 "${projectPath}/logprobs/logprobs.py" \
        --config_file "${projectPath}/logprobs/config.yaml" \
        --model_id "$model" \
        --minimal_file "bl2mp" \
        --task "ctxt" \
        --batch_size "${batchSize}" \
        --output_dir "${outputPath}"

    # Direct Probability Measurements (DPM)
    python3 "${projectPath}/logprobs/logprobs.py" \
        --config_file "${projectPath}/logprobs/config.yaml" \
        --model_id "$model" \
        --minimal_file "bat" \
        --task "sent" \
        --batch_size "${batchSize}" \
        --output_dir "${outputPath}"
    # Instruction Probability Measurements (IPM)
    python3 "${projectPath}/logprobs/logprobs.py" \
        --config_file "${projectPath}/logprobs/config.yaml" \
        --model_id "$model" \
        --minimal_file "bat" \
        --task "ctxt" \
        --batch_size "${batchSize}" \
        --output_dir "${outputPath}"
done

instruct_models=(
    # meta-llama/Llama-3.1-8B-Instruct
)

for model in "${instruct_models[@]}"; do
    # Instruction Probability Measurements (IPM)
    python3 "${projectPath}/logprobs/logprobs.py" \
        --config_file "${projectPath}/logprobs/config.yaml" \
        --model_id "$model" \
        --minimal_file "bl2mp" \
        --task "inst" \
        --batch_size "$batchSize" \
        --output_dir "${outputPath}"
    # Instruction Probability Measurements (IPM)
    python3 "${projectPath}/logprobs/logprobs.py" \
        --config_file "${projectPath}/logprobs/config.yaml" \
        --model_id "$model" \
        --minimal_file "bat" \
        --task "inst" \
        --batch_size "$batchSize" \
        --output_dir "${outputPath}"
done
```

> **Warning:** `logprobs.py` requires two lists to be hard-coded directly in the script, since they cannot be expressed cleanly in `config.yaml`. First, `no_system_support` must list model IDs whose chat template rejects a system message (e.g. Gemma-2-based instruct models) — for these, the system prompt is dropped. Second, `bos_token` must list model IDs whose tokenizer does not automatically prepend a BOS token, so the script needs to add it manually before computing log-probabilities; skipping this for an affected model silently shifts every logprob computation.

```python
# Add model IDs that do not support system messages
no_system_support = [
    # No system message supported in Gemma-2
    "orai-nlp/Gemma-Kimu-9b-it",
]

# Some models do not add a bos token at the beginning
bos_token = [
    "zai-org/GLM-4-32B-Base-0414",
    "zai-org/GLM-4.5-Air-Base",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-35B-A3B-Base",
]
```

## GPU Requirements by Model Size

Recommended resource requests (NVIDIA A100 SXM4 80GB GPU, for reference), grouped by parameter count:

| Model size | GPUs | CPUs per task | Memory |
|---|---|---|---|
| < 32B | gpu:1 | 4 | 80GB |
| 32B < x < 70B | gpu:4 | 10 | 150GB |
| ~120B | gpu:4 | 12 | 200GB |

## Where JSONL Logs Are Stored

By default, `logprobs.py` writes its `.jsonl` output logs to the `logs/LP` directory inside your `basque_linguistic_competence` checkout. In the example script above, this is controlled by the `outputPath` variable:

```bash
projectPath="<TODO>/basque_linguistic_competence/"
# Default results storing directory
outputPath="${projectPath}/logs/LP"
```

This value is passed to every `logprobs.py` invocation via the `--output_dir "${outputPath}"` flag, so unless you override it, every DPM (`sent`), IPM (`ctxt`), and instruction-conditioned (`inst`) run will deposit its `.jsonl` results under `<projectPath>/logs/LP`.

---

# What is `retrieve_results.py`

`retrieve_results.py` is a post-processing script that walks through the `.jsonl` log files produced by `logprobs.py` runs and reassembles/validates their results into a single summary structure, keyed by model and task (`bat-sent`, `bat-ctxt`, `bl2mp-ctxt`, `bl2mp-sent`, `bat-inst`, `bl2mp-inst`). 

> **Warning: Setting the project path** — Before running the script, you **must** update the hardcoded `projectPath` variable at the top of the file (currently `"/basque_linguistic_competence/"`) to the actual local path of your `basque_linguistic_competence` checkout. This path is used to build `dataset_mapping` (the `.jsonl` files for `bl2mp`, `bat`) and `logs_path`, which points to the `logs/LP` subdirectory — the script assumes results are stored under an `LP` folder inside `logs/`, matching the `outputPath` convention used when running `logprobs.py`. If your logs live elsewhere, either move them into `<projectPath>/logs/LP` or edit `logs_path` directly.

## Adjustable Flags

`check_results(directory, skip_missing, check_tokens)` exposes two optional parameters you can tune depending on how strict you want the validation to be: 

| Flag | Default | What it does | When to change it |
|---|---|---|---|
| **`skip_missing`** | `False` | Controls whether every `model_id` found in a log file must already appear in the hardcoded `base_models` or `instruct_models` lists. | Set to **`False`** if you want a hard `assert` error whenever an unlisted model shows up — useful for catching typos or forgotten registrations. Set it **`True`** if you expect logs from models not yet added to those lists. |
| **`check_tokens`** | `False` | Loads the tokenizer via `transformers.AutoTokenizer.from_pretrained` and verifies, sentence by sentence, that the recovered tokens match `sentence_good`/`sentence_bad`. | Set it **`True`** for a strong correctness guarantee — at the cost of downloading/loading every tokenizer (slower, needs network access). Set to **`False`** to skip this check and speed up retrieval when you already trust the logged data. |