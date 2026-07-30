# Curriculum Training Guide

## Purpose

This directory contains the data references, Axolotl configurations, shared training utilities, and Slurm scripts used to continue training the Latxa Llama 3.1 8B base and instruction-tuned models on several Basque corpora.

The intended workflow has three stages:

1. Select and adapt a training configuration.
2. Preprocess the corresponding data into a reusable cache.
3. Launch the training job using one GPU with CPU offloading.

> **Warning:**
> Preprocessing can be performed automatically during training, but running it as a separate job first is strongly recommended. It isolates data or tokenization failures from an expensive training allocation and prevents avoidable failures after training has already started.

## Directory organization

```text
curriculum/
├── corpus/                         # Source JSONL datasets
├── cache/                          # Generated datasets, models, and Hugging Face cache
├── scripts/
│   ├── 0-train_configs/            # Stage 1: experiment definitions
    │   │   ├── base/                 # Train the base Latxa model
    │   │   │   ├── standard/         # Higher-quality corpora
    │   │   │   └── diverse/          # Lower-controlled-quality corpora
    │   │   ├── instruct/             # Train the instruction-tuned Latxa model
    │   │   │   ├── standard/         # Higher-quality corpora plus instruction data
    │   │   │   └── diverse/          # Lower-quality corpora plus instruction data
    │   │   └── common/               # Shared DeepSpeed and early-stopping support
│   ├── 1-preprocess/
│   │   └── prepare_data.sh       # Stage 2: tokenize and cache every configured dataset
│   └── 2-train/
│       └── train.sh              # Stage 3: train every configured experiment
```

### `corpus/`

This directory holds the 60M token JSONL source datasets used by the YAML configurations. 

| Group | Corpus configurations |
| --- | --- |
| Standard (higher quality) | `berria_60M`, `instituzionalak_60M`, `BOPV_60M` |
| Diverse (more varied, not held to the same quality level) | `chimicron_60M`, `txioak_60M`, `ehkirola_60M` |

### `cache/`

The scripts populate it at runtime:

- `cache/dataset_cache/` stores Axolotl's preprocessed/tokenized datasets.
- `cache/model/` receives model checkpoints and final outputs, separated by experiment name.
- `cache/hf/` is the local Hugging Face cache created by the preprocessing script.

## The three-stage pipeline

> **Note:**
> Every `<TODO>` placeholder across the chosen configs must be set before running anything, including the repo's absolute path and the CUDA-related paths required by the scripts.

### 1. Choose and configure an experiment

Pick a YAML file under `scripts/0-train_configs/` based on model variant (`base/` continues `HiTZ/Latxa-Llama-3.1-8B` on raw Basque text; `instruct/` continues `HiTZ/Latxa-Llama-3.1-8B-Instruct` on mixed English-instruction plus Basque data) and corpus group (`standard` vs. `diverse`). There are six configs per variant.

`common/` supplies shared support: a DeepSpeed ZeRO Stage 2 config (CPU-offloaded optimizer states, driving the 250 GB RAM request) and `early_stopping_plugin.py`, which halts training after the configured 1 epoch.

### 2. Preprocess the data

Run `sbatch scripts/1-preprocess/prepare_data.sh` once paths/configs are set. This Slurm job (1 node, 16 CPUs, 168 GB RAM, 1 GPU) sets up caches, runs offline, and calls `axolotl.cli.preprocess` for all 12 configs. Confirm it completes and populates `cache/dataset_cache/` before training.

### 3. Train

Run `sbatch scripts/2-train/train.sh` after preprocessing succeeds. It launches the 12 experiments sequentially via Accelerate/Slurm on a single GPU (`--num_processes 1`), relying on CPU optimizer offloading and gradient checkpointing to fit an 8B model at 8192 tokens. Do not remove these memory-saving settings without validating alternative hardware. Outputs are stored under `cache/model/` per each config's `output_dir`.

> **For reference:**
> Training was run using 1 NVIDIA A100 SXM4 80GB GPU.