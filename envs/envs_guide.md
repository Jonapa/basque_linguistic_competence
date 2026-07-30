# Environment Documentation

> This document describes two separate development environments, each created for a different machine and purpose. `evaluation` environment is used for benchmarking LLMs, while `curriculum` is used for curriculum learning experiments.

---

## Environment 1: `evaluation`

### Overview
> This environment supports the model evaluation pipeline: running inference with vLLM and scoring models with lm-evaluation-harness & minicons log-probabilities computation.

### System Specifications

| Field | Value |
|---|---|
| Architecture | x86-64 |
| OS | Rocky Linux 8.10 |
| glibc | 2.28 |
| CUDA Toolkit | 12.8.0 |
| Module load | `module load GCC/14.2.0 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0` |

---

### Dependencies & Library Versions

<div align="center">

| Library | Version | Notes |
|:-----------|:----------:|:---------|
| `miniforge3` | 24.11.3-2 | Conda package manager |
| `python` | 3.13.13 | |
| `vllm` | 6850839 | Support for Gemma4 commit (https://github.com/vllm-project/vllm/pull/44429) |
| `lm-evaluation-harness` | 0.4.11 | We will need source code to adapt for reasoning LLMs. |
| `transformers` | 5.10.2 | |
| `minicons` | 0.3.36 | |

</div>

```bash
# conda create -n evaluation python=3.13.13
# git clone https://github.com/vllm-project/vllm.git
# git clone https://github.com/EleutherAI/lm-evaluation-harness
# git clone https://github.com/huggingface/transformers.git
# conda activate evaluation

cd vllm/
git checkout 6850839
pip3 install . --extra-index-url https://download.pytorch.org/whl/cu128

# python3 -c "import vllm;print(vllm.__version__)"
# python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}')"

cd ..
cd lm-evaluation-harness/
git checkout v0.4.11
pip3 install .

# python3 -c "import lm_eval;print(lm_eval.__version__)"

# Missing dependency
pip3 install ray

cd ..
cd transformers/
git checkout v5.10.2
pip3 install .

# python3 -c "import transformers;print(transformers.__version__)"

pip3 install minicons==0.3.36

# python3 -c "import minicons;print(minicons.__version__)"
```

---

## Environment 2: `curriculum`

### Overview
> This environment supports curriculum learning training runs with Axolotl, using flash-attention and DeepSpeed for efficient multi-GPU fine-tuning (if necessary).

### System Specifications

| Field | Value |
|---|---|
| Architecture | x86-64 |
| OS | AlmaLinux 9.8 |
| glibc | 2.34 |
| CUDA Toolkit | 12.8.0 |
| Module load | `module load GCC/13.2.0 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0` |

---

### Dependencies & Library Versions

<div align="center">

| Library | Version | Notes |
|:-----------|:----------:|:---------|
| `conda` | 25.11.0 | |
| `python` | 3.13.11 | |
| `flash-attn` | 2.8.3 | Installed via conda-forge to avoid local compilation (https://github.com/conda-forge/flash-attn-feedstock) |
| `axolotl` | 0.13.0 | |

</div>

```bash
# conda create -n curriculum python=3.13.11
# conda activate curriculum

# Set conda level override (conda info __cuda)
export CONDA_OVERRIDE_CUDA="12.8"
# With GPU
conda install -c conda-forge flash-attn
# Without GPU
conda install -c conda-forge flash-attn cuda-version=12.8

# python -c "import flash_attn, torch; print(f'Flash Attention: {flash_attn.__version__}\nTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Axolotl requirements
pip3 install packaging setuptools wheel ninja

git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl/
pip3 install .[deepspeed] --no-build-isolation

# python -c "import flash_attn, torch, deepspeed; print(f'Flash Attention: {flash_attn.__version__}\nTorch: {torch.__version__}, CUDA: {torch.version.cuda}\nDeepSpeed: {deepspeed.__version__}')"
# axolotl --help
```
---
