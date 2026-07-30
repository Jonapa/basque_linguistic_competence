# Basque benchmark evaluation guide

This directory supports three related evaluation workflows:

1. **General-purpose Basque benchmarks** run through `lm-evaluation-harness`, using additional task definitions from `missing_tasks/` where necessary.
2. **BAT and BL2MP acceptability judgments** use custom A/B multiple-choice tasks with fixed prompts, few-shot examples, and answer order.
3. **Reasoning-model A/B multiple-choice evaluation** generates a reasoning trace first and then scores the A/B choices conditioned on that same trace.

## Directory overview

```text
benchmarks/
├── evaluation_vllm.sh           # General-purpose/A/B evaluation with vLLM backend
├── evaluation_hf.sh             # General-purpose/A/B evaluation with Hugging Face backend
├── retrieve_results.py          # General-purpose/A/B aggregation → data-2.json
├── missing_tasks/               # Additional lm-evaluation-harness tasks
│   ├── a_b/                        # BAT/BL2MP tasks and deterministic sampler
│   ├── arc_mc/                     # ARC letter-choice variants
│   ├── bertaqa/                    # BertaQA tasks and groups
│   ├── mmlu_eu/                    # English, Spanish, and Basque MMLU
│   └── piqa_mc/                    # PIQA variants
├── code/                        # Modified lm_eval source files
│   ├── api/
│   │   └── task.py
│   ├── models/
│   │   ├── hf_vlms.py
│   │   └── vllm_causallms.py
│   └── evaluator.py
└── thinking/
    ├── thinking.py              # Two-pass reasoning evaluation
    ├── run_thinking.sh          # Sampled and deterministic run examples
    └── retrieve_results.py      # Reasoning aggregation → data-3.json
```

## Prerequisites and local configuration

Install a compatible version of `lm-evaluation-harness` (see `../envs/envs_guide.md`). The files in `code/` mirror harness internals and are therefore version-sensitive; review their marked `################################################################################################ ADDED` blocks against the installed harness rather than assuming they apply cleanly to every release.

The BAT and BL2MP JSONL files are expected outside this directory, under the parent repository's `data/` directory (`../data/` relative to this directory).

## General-purpose Basque benchmarks

Both general-purpose evaluation scripts evaluate the following task set with five few-shot examples:

- `arc_eu_challenge_mc`
- `arc_eu_easy_mc`
- `belebele_eus_Latn`
- `bertaqa_eu_global`
- `bertaqa_eu_local`
- `eus_exams_eu`
- `eus_reading`
- `eus_trivia`
- `mmlu_eu`
- `piqa_eu_mc`

The scripts use an 8,192-token context by default and 20,000 tokens for `eus_reading`. They disable model thinking with `enable_thinking=False` and log individual samples. Keep `--log_samples` enabled because A/B multiple-choice aggregation depends on the sample files. General-purpose results are written below `../logs/BENCHMARK/`.

### Additional task definitions

Pass the local task directory to every harness invocation:

```bash
--include_path "${projectPath}/benchmarks/missing_tasks/"
```

The directory contains:

- `arc_mc/`: English, Catalan, and Basque ARC variants;
- `piqa_mc/`: English, Catalan, and Basque PIQA variants;
- `mmlu_eu/`: English, Spanish, and Basque MMLU variants;
- `bertaqa/`: English and Basque BertaQA local/global tasks and groups;
- `a_b/`: BAT and BL2MP acceptability tasks.

The ARC and PIQA variants and BertaQA place the answer text in the prompt and score only a choice letter. This avoids comparing the likelihoods of answer strings whose different lengths could distort a multiple-choice result.

### Base and instruction-tuned models

Base checkpoints are evaluated as plain text continuations. Instruction-tuned checkpoints additionally use:

```bash
--apply_chat_template \
--fewshot_as_multiturn
```

Models in `instruct_models` also receive:

```bash
--system_instruction "You are a helpful assistant."
```

Instruction-tuned models whose templates cannot represent a system turn belong in `instruct_models_2`; the scripts omit the system instruction for that group. Keeping these groups separate is important because chat formatting changes the actual evaluation prompt.

### Choosing a backend

Run one of the following from this directory after configuration:

```bash
bash evaluation_vllm.sh
bash evaluation_hf.sh
```

`evaluation_vllm.sh` uses four-way tensor parallelism and is generally the faster option. `evaluation_hf.sh` uses `parallelize=True` and is useful when a model architecture is unsupported by the installed `vLLM` version or when the longer `eus_reading` inputs are easier to accommodate through the `Hugging Face Transformers` backend. Backend choice does not change the task definitions or metrics.

> **`<TODO>`:** Set `projectPath` in `evaluation_vllm.sh` and `evaluation_hf.sh` to the local `basque_linguistic_competence` checkout.

### Benchmark GPU requirements

The recorded large-model configurations used the following resources:

| Parameter size | Reference GPUs | Host RAM | CPUs per task |
|---|---:|---:|---:|
| Approximately 70B | 4 × NVIDIA A100 SXM4 80GB | 150 GB | 10 |
| Approximately 120B | 6 × NVIDIA A100 SXM4 80GB | 250 GB | 16 |

These are reference configurations, not strict minima. Smaller checkpoints may need fewer resources, while different precision, context length, backend, or GPU type may change memory requirements. Keep `vLLM`'s `tensor_parallel_size` equal to the number of GPUs assigned to the model; the supplied evaluation script currently assumes four GPUs. The `Hugging Face Transformers` evaluation script uses `parallelize=True` and distributes the checkpoint across the available devices.

## BAT and BL2MP A/B multiple-choice tasks

BAT and BL2MP measure whether a model assigns greater likelihood to the acceptable Basque sentence. Each example displays two sentences and scores only the letter `A` or `B`.

There are separate task definitions for base and instruction-tuned models:

| Dataset | Base task | Instruction task |
|---|---|---|
| BAT | `ab-base_BAT` | `ab-instruct_BAT` |
| BL2MP | `ab-base_BL2MP` | `ab-instruct_BL2MP` |

The general-purpose evaluation scripts write these results below `../logs/MC/`. The custom sampler is designed specifically for `--num_fewshot 5`; other shot counts fail intentionally.

> **`<TODO>`:** In all four `missing_tasks/a_b/*.yaml` files, replace `<TODO>` in `dataset_kwargs.data_files` with the local path prefix for `../data/bat.jsonl` or `../data/bl2mp.jsonl`.

### Reproducible prompt construction

`missing_tasks/a_b/utils.py` makes three experimental choices deterministic:

1. It evaluates every source item under five Basque instructions.
2. It fixes the five few-shot examples used for each instruction.
3. It fixes whether the acceptable sentence appears as A or B for every item.

`process_docs_ab` identifies BL2MP by its 1,800 rows and BAT by its 762 rows. For each of the five instructions it emits a complete copy of the dataset, producing 9,000 BL2MP rows or 3,810 BAT rows. The output is arranged in five contiguous prompt blocks, which the result collectors later split back into five conditions.

The custom `FewshotSampler` selects the hard-coded indices associated with the current prompt block. A global counter tracks the expanded row being evaluated. This design is deliberately limited to the expected BAT and BL2MP sizes so that prompt conditions remain identical across runs and model providers.

### Avoiding repeated instructions

For base models, the `description` field is suitable because it appears once before the few-shot examples. For instruction-tuned models, however, `description` becomes a system message, which is not the intended prompt structure. The task therefore uses `doc_to_text`, but the unmodified harness would repeat the instruction in every few-shot turn.

The added block in `code/api/task.py` keeps the first occurrence and removes the repeated copies when constructing `ab-instruct_BAT` and `ab-instruct_BL2MP` requests. This modification is required for the intended instruction-model prompt format, including direct A/B multiple-choice evaluation—not only the reasoning workflow.

### Excluding few-shot example overlaps

If the evaluated item is one of the five fixed few-shot examples, the sampler omits that example to prevent the item from appearing as its own example. Such a request has four few-shot examples rather than five and is therefore not comparable to the other requests.

The collectors exclude these cases by counting the `B:` markers in each constructed prompt. There are five overlaps in each of the five prompt blocks, so exactly 25 evaluation rows are removed per task. These rows would be evaluated with four few-shot examples (4-shot) instead of the intended five (5-shot); removing them keeps the evaluation conditions fair and comparable. The reported BAT and BL2MP result is therefore a five-element list containing one accuracy for each instruction, not a single accuracy pooled across prompts.

## Reasoning-model A/B multiple-choice evaluation

Standard A/B multiple-choice evaluation directly compares the likelihood of each continuation. For a reasoning model, this does not provide an opportunity to generate a reasoning trace first. `thinking/thinking.py` implements a two-pass procedure:

1. Temporarily convert the A/B multiple-choice task from `multiple_choice` to `generate_until` and use `vLLM` to generate one trace per prompt.
2. Release the `vLLM` model and clear memory.
3. Restore the A/B multiple-choice task, load the `Hugging Face Transformers` backend, insert the saved trace before both candidate letters, and score their likelihoods.

The same trace is reused for A and B. It is generated before either candidate letter is appended, so the likelihood comparison is conditioned on identical reasoning rather than on two independently sampled traces.

Model-family detection is name-based. The supported beginning/end tokens and `vLLM` reasoning parsers are declared in `reasoning_tokens` near the top of `thinking/thinking.py`. A model whose name does not match one of those entries raises an error and requires an explicit configuration addition.

Some supported architectures select the `hf-multimodal` backend for the scoring pass; this path forces batch size 1 because automatic batching is not supported there. The script also lowers the second-pass maximum length for selected larger parameter sizes to reduce memory pressure.

### Required `lm-evaluation-harness` modifications

The reasoning workflow depends on the modified harness files under `code/`:

- `code/models/vllm_causallms.py` captures generated traces and restores dataset order after `vLLM`'s internal request reordering;
- `code/api/task.py` adds the correct beginning-of-thinking marker and injects the saved trace plus its end marker into both choice requests;
- `code/evaluator.py` deletes the first-pass model and clears `Python` and `CUDA` memory before the scoring pass;
- `code/models/hf_vlms.py` forwards `enable_thinking` to multimodal chat templates.

Copy these files to the corresponding locations in the installed `lm_eval` package, or manually apply the sections marked `################################################################################################ ADDED`. Prefer applying the marked changes when the installed harness differs from the source version used here.

Trace ordering is essential: the `lm-evaluation-harness` collator reorders requests by token length for inference batch stability, while the scoring pass visits documents in dataset order. The custom `vLLM` code uses the harness collator's reorder indices before storing traces in the process-global `reasoning_traces` list. `code/api/task.py` then advances `reasoning_i` once per evaluated document and inserts that document's trace into both choice contexts.

### Running reasoning evaluations

> **`<TODO>`:** Set `projectPath` in `thinking/run_thinking.sh` to the local `basque_linguistic_competence` checkout, and activate the required `Python` environment and machine-specific modules.

After configuring `projectPath` and the environment, run:

```bash
bash thinking/run_thinking.sh
```

The run script evaluates `ab-instruct_BAT` and `ab-instruct_BL2MP` with five few-shot examples. It contains sampled-reasoning runs under `../logs/THINKING-SAMPLING/`, using the recommended sampling hyperparameters for each architecture, and deterministic-reasoning runs under `../logs/THINKING/`. Its generation settings, 24,576-token context, 16,384-token generation ceiling, and four-way tensor parallelism are experiment defaults rather than universal requirements.

`thinking/thinking.py` writes one JSON object per scored sample. Output filenames include the model basename, task name, `thinking`, and a timestamp.

### Reasoning GPU requirements

The supplied reasoning run script is configured for **4 GPUs** with `tensor_parallel_size=4` and a 24,576-token first-pass context. It uses 60–90% of GPU memory depending on parameter size and run mode. Four NVIDIA A100 SXM4 80GB GPUs are the reference configuration; checkpoints above 100B parameters use the most demanding settings and may require more GPUs in another environment.

Reasoning evaluation is more memory-intensive than general-purpose scoring because it runs two model backends sequentially and retains long generated traces for the second pass. `code/evaluator.py` explicitly releases the `vLLM` model and clears `CUDA` memory between passes. If the allocation changes, update `--tensor_parallel_size`, `--gpu_memory_utilization`, `--max_model_len`, and, where necessary, the second-pass `Hugging Face Transformers` placement together.

## Aggregating results

> **`<TODO>`:** Set `path` in `retrieve_results.py` and `thinking/retrieve_results.py` to the local repository's `../logs/` directory.

After setting the log paths, run both collectors from this directory:

```bash
python3 retrieve_results.py
python3 thinking/retrieve_results.py
```

### General-purpose and A/B multiple-choice results

`retrieve_results.py` scans `../logs/BENCHMARK/` and `../logs/MC/` and writes `data-2.json`.

- General-purpose tasks use the metric configured in `tasks_metric`.
- Grouped results are combined using effective sample counts rather than giving every subtask equal weight.
- BAT and BL2MP ignore the harness-wide aggregate, remove the 25 overlapping evaluation rows, split the remaining rows into five prompt blocks, and report five accuracies. The removed rows correspond to the five fixed few-shot examples in each of the five prompt blocks. Each would otherwise use only four few-shot examples (4-shot) instead of five (5-shot), so excluding them preserves a fair and consistent evaluation setting.

The script preallocates expected model/task combinations from `models_tasks`. Result model identifiers must match those registry strings exactly. Add new models or tasks to that structure and add general-purpose task metrics to `tasks_metric`. With `skip_missing=True`, unexpected result files are skipped, while expected results that were not found remain `None` and are printed as missing.

One current limitation is that a result containing multiple entries in `data["results"]` is inferred to be `eus_exams_eu`. Generalize that inference before adding another grouped task whose output has the same shape.

### Reasoning results

`thinking/retrieve_results.py` scans `../logs/THINKING/` and `../logs/THINKING-SAMPLING/` and writes `data-3.json`. It removes the same 25 overlapping evaluation rows—the five fixed few-shot examples in each of the five prompt blocks—because those rows would be evaluated 4-shot instead of 5-shot. This exclusion keeps the comparison fair. The script then returns five prompt-level accuracies for each task and reasoning mode.

Because a generated trace may itself mention `B:`, the collector counts markers only in the prompt portion before the model-family beginning-of-thinking token. Its expected model names and tasks are also fixed in `models_tasks`; update that registry when evaluating additional models. Although the function accepts `skip_missing`, its current implementation does not use the argument to ignore unknown reasoning results.
