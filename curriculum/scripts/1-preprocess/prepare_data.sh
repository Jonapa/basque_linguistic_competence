#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=168GB
#SBATCH --gres=gpu:1

projectPath="<TODO>/basque_linguistic_competence/"

# Activate curriculum environment as well as necessary modules

export WANDB_MODE=offline
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${projectPath}/curriculum/cache/dataset_cache"
mkdir -p "${projectPath}/curriculum/cache/model"
mkdir -p "${projectPath}/curriculum/cache/hf"
export TMPDIR="${projectPath}/curriculum/cache"
export HF_HOME="${projectPath}/curriculum/cache/hf"

# https://docs.axolotl.ai/docs/telemetry.html
export AXOLOTL_DO_NOT_TRACK=1
export PYTHONPATH=${projectPath}/curriculum/scripts/0-train_configs/common:$PYTHONPATH


# Base
# Standard
# berria
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/base/standard/train_config-berria_60M.yaml"
# instituzionalak
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/base/standard/train_config-instituzionalak_60M.yaml"
# BOPV
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/base/standard/train_config-BOPV_60M.yaml"

# Diverse
# chimicron
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/base/diverse/train_config-chimicron_60M.yaml"
# txioak
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/base/diverse/train_config-txioak_60M.yaml"
# ehkirola
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/base/diverse/train_config-ehkirola_60M.yaml"

# Instruct
# Standard
# berria
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/instruct/standard/train_config-exp_2_101-berria_60M.yaml"
# instituzionalak
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/instruct/standard/train_config-exp_2_101-instituzionalak_60M.yaml"
# BOPV
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/instruct/standard/train_config-exp_2_101-BOPV_60M.yaml"

# Diverse
# chimicron
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/instruct/diverse/train_config-exp_2_101-chimicron_60M.yaml"
# txioak
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/instruct/diverse/train_config-exp_2_101-txioak_60M.yaml"
# ehkirola
python -m axolotl.cli.preprocess "${projectPath}/curriculum/scripts/0-train_configs/instruct/diverse/train_config-exp_2_101-ehkirola_60M.yaml"
