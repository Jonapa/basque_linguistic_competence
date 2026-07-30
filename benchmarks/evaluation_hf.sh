# Set this to the local path where the basque_linguistic_competence repo is checked out
projectPath="<TODO>/basque_linguistic_competence/"

# Activate evaluation environment as well as necessary modules

missingTasksPath="${projectPath}/benchmarks/missing_tasks/"
num_fewshot=5

export CUDA_LAUNCH_BLOCKING=1

base_models=(
	# "meta-llama/Llama-3.1-8B"
)

outputPath="${projectPath}/logs/BENCHMARK"

tasks_selected=(
    "arc_eu_challenge_mc"
    "arc_eu_easy_mc"
    "belebele_eus_Latn"
    "bertaqa_eu_global"
    "bertaqa_eu_local"
    "eus_exams_eu"
    "eus_reading"
    "eus_trivia"
    "mmlu_eu"
    "piqa_eu_mc"
)

for group_name in "${tasks_selected[@]}"; do
    # max model len 8192 by default, 20k for eus_reading
    if [[ $group_name == "eus_reading" ]]; then
        max_model_len=20000
    else
        max_model_len=8192
    fi

    for model in "${base_models[@]}"; do
        mkdir -p "${outputPath}/${model}/${group_name}"
        lm-eval \
            --model hf \
            --model_args "pretrained=${model},dtype=bfloat16,parallelize=True,max_length=${max_model_len},enable_thinking=False" \
            --tasks "${group_name}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "auto" \
            --device "cuda" \
            --output_path "${outputPath}/${model}/${group_name}" \
            --trust_remote_code \
            --log_samples \
            --verbosity "DEBUG" \
            --write_out \
            --show_config \
            --include_path "${missingTasksPath}"
    done
done

outputPath="${projectPath}/logs/MC"

tasks_selected=(
    "ab-base_BAT"
    "ab-base_BL2MP"
)

for group_name in "${tasks_selected[@]}"; do
    # max model len 8192 by default, 20k for eus_reading
    if [[ $group_name == "eus_reading" ]]; then
        max_model_len=20000
    else
        max_model_len=8192
    fi

    for model in "${base_models[@]}"; do
        mkdir -p "${outputPath}/${model}/${group_name}"
        lm-eval \
            --model hf \
            --model_args "pretrained=${model},dtype=bfloat16,parallelize=True,max_length=${max_model_len},enable_thinking=False" \
            --tasks "${group_name}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "auto" \
            --device "cuda" \
            --output_path "${outputPath}/${model}/${group_name}" \
            --trust_remote_code \
            --log_samples \
            --verbosity "DEBUG" \
            --write_out \
            --show_config \
            --include_path "${missingTasksPath}"
    done
done

####################################################################################################################################################################################


instruct_models=(
	# "meta-llama/Llama-3.1-8B-Instruct"
)

instruct_models_2=(
	# "orai-nlp/Gemma-Kimu-9b-it"
)

outputPath="${projectPath}/logs/BENCHMARK"

tasks_selected=(
    "arc_eu_challenge_mc"
    "arc_eu_easy_mc"
    "belebele_eus_Latn"
    "bertaqa_eu_global"
    "bertaqa_eu_local"
    "eus_exams_eu"
    "eus_reading"
    "eus_trivia"
    "mmlu_eu"
    "piqa_eu_mc"
)

for group_name in "${tasks_selected[@]}"; do
    # max model len 8192 by default, 20k for eus_reading
    if [[ $group_name == "eus_reading" ]]; then
        max_model_len=20000
    else
        max_model_len=8192
    fi

    system_instruction="You are a helpful assistant."
    for model in "${instruct_models[@]}"; do
        mkdir -p "${outputPath}/${model}/${group_name}"
        lm-eval \
            --model hf \
            --model_args "pretrained=${model},dtype=bfloat16,parallelize=True,max_length=${max_model_len},enable_thinking=False" \
            --tasks "${group_name}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "auto" \
            --device "cuda" \
            --output_path "${outputPath}/${model}/${group_name}" \
            --trust_remote_code \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --system_instruction "${system_instruction}" \
            --verbosity "DEBUG" \
            --write_out \
            --show_config \
            --include_path "${missingTasksPath}"
    done
    # Without system instruction
    for model in "${instruct_models_2[@]}"; do
        mkdir -p "${outputPath}/${model}/${group_name}"
        lm-eval \
            --model hf \
            --model_args "pretrained=${model},dtype=bfloat16,parallelize=True,max_length=${max_model_len},enable_thinking=False" \
            --tasks "${group_name}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "auto" \
            --device "cuda" \
            --output_path "${outputPath}/${model}/${group_name}" \
            --trust_remote_code \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --verbosity "DEBUG" \
            --write_out \
            --show_config \
            --include_path "${missingTasksPath}"
    done
done

outputPath="${projectPath}/logs/MC"

tasks_selected=(
    "ab-instruct_BAT"
    "ab-instruct_BL2MP"
)

for group_name in "${tasks_selected[@]}"; do
    # max model len 8192 by default, 20k for eus_reading
    if [[ $group_name == "eus_reading" ]]; then
        max_model_len=20000
    else
        max_model_len=8192
    fi

    system_instruction="You are a helpful assistant."
    for model in "${instruct_models[@]}"; do
        mkdir -p "${outputPath}/${model}/${group_name}"
        lm-eval \
            --model hf \
            --model_args "pretrained=${model},dtype=bfloat16,parallelize=True,max_length=${max_model_len},enable_thinking=False" \
            --tasks "${group_name}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "auto" \
            --device "cuda" \
            --output_path "${outputPath}/${model}/${group_name}" \
            --trust_remote_code \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --system_instruction "${system_instruction}" \
            --verbosity "DEBUG" \
            --write_out \
            --show_config \
            --include_path "${missingTasksPath}"
    done
    # Without system instruction
    for model in "${instruct_models_2[@]}"; do
        mkdir -p "${outputPath}/${model}/${group_name}"
        lm-eval \
            --model hf \
            --model_args "pretrained=${model},dtype=bfloat16,parallelize=True,max_length=${max_model_len},enable_thinking=False" \
            --tasks "${group_name}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "auto" \
            --device "cuda" \
            --output_path "${outputPath}/${model}/${group_name}" \
            --trust_remote_code \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --verbosity "DEBUG" \
            --write_out \
            --show_config \
            --include_path "${missingTasksPath}"
    done
done
