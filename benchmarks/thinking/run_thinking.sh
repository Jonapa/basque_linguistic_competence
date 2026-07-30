# Set this to the local path where the basque_linguistic_competence repo is checked out
projectPath="<TODO>/basque_linguistic_competence/"

# Activate evaluation environment as well as necessary modules

export CUDA_LAUNCH_BLOCKING=1

num_fewshot=5
missingTasksPath="${projectPath}/benchmarks/missing_tasks/"
system_instruction="You are a helpful assistant."

task_sets=(
    "ab-instruct_BAT"
    "ab-instruct_BL2MP"
)

for group_name in "${task_sets[@]}"; do

    outputPath="${projectPath}/logs/THINKING-SAMPLING"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "google/gemma-4-12B-it" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "google/gemma-4-31B-it" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "google/gemma-4-26B-A4B-it" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-9B" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-27B" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-35B-A3B" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-122B-A10B" --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.6-35B-A3B" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.6-27B" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "zai-org/GLM-4.5-Air" --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 1.0, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "zai-org/GLM-4.6V-Flash" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 0.8, "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "zai-org/GLM-4.6V" --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 24576 --gen_kwargs '{"do_sample": true, "temperature": 0.8, "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    outputPath="${projectPath}/logs/THINKING"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "google/gemma-4-12B-it" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "google/gemma-4-31B-it" --tensor_parallel_size 4 --gpu_memory_utilization 0.8 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "google/gemma-4-26B-A4B-it" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-9B" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-27B" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-35B-A3B" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.5-122B-A10B" --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.6-35B-A3B" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"
    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "Qwen/Qwen3.6-27B" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "zai-org/GLM-4.5-Air" --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "zai-org/GLM-4.6V-Flash" --tensor_parallel_size 4 --gpu_memory_utilization 0.6 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

    python3 "${projectPath}/benchmarks/thinking/thinking.py" --model "zai-org/GLM-4.6V" --tensor_parallel_size 4 --gpu_memory_utilization 0.9 --max_model_len 24576 --gen_kwargs '{"do_sample": false, "max_gen_toks": 16384}' --task "${group_name}" --num_fewshot "${num_fewshot}" --batch_size "auto" --output_path "${outputPath}" --system_instruction "${system_instruction}" --include_path "${missingTasksPath}"

done
