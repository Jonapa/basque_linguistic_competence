#!/bin/bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=250GB

projectPath="<TODO>/basque_linguistic_competence/"

# Activate curriculum environment as well as necessary modules

export CUDA_HOME=<TODO>
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export CPATH=$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:${CPATH}}
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}

# https://docs.axolotl.ai/docs/telemetry.html
export AXOLOTL_DO_NOT_TRACK=1

export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16

export HF_HUB_OFFLINE=1 # Offline mode for Hugging Face
export WANDB_MODE=offline

# Maximum time (in seconds) to wait for NCCL operations before timing out
export NCCL_TIMEOUT=14400 # 4 hours, prevent crashing when loading datasets
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 # 1 hour, prevent HEARTBEAT_TIMEOUT error

# Enable asynchronous error handling for NCCL operations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Enable the memory allocator to allocate memory in a way that is more likely to be compatible with NCCL
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Logs
# https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html#enabling-flight-recorder
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# export TORCH_FR_BUFFER_SIZE=2000
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO

# https://github.com/huggingface/accelerate/issues/314
# https://docs.axolotl.ai/docs/nccl.html
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL

# MASTER_PORT=9327
# MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export PYTHONPATH=${projectPath}/curriculum/scripts/0-train_configs/common:$PYTHONPATH

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/base/standard/train_config-berria_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/base/standard/train_config-instituzionalak_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/base/standard/train_config-BOPV_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/base/diverse/train_config-chimicron_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/base/diverse/train_config-txioak_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/base/diverse/train_config-ehkirola_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/instruct/standard/train_config-exp_2_101-berria_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/instruct/standard/train_config-exp_2_101-instituzionalak_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/instruct/standard/train_config-exp_2_101-BOPV_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/instruct/diverse/train_config-exp_2_101-chimicron_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/instruct/diverse/train_config-exp_2_101-txioak_60M.yaml"

srun accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    -m axolotl.cli.train "${projectPath}/curriculum/scripts/0-train_configs/instruct/diverse/train_config-exp_2_101-ehkirola_60M.yaml"
