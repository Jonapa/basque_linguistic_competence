from pathlib import Path
import json
from typing import Dict, List, Tuple
import numpy as np

tasks_metric: Dict[str,str] = {
    "arc_eu_challenge_mc": "acc_norm,none",
    "arc_eu_easy_mc": "acc_norm,none",
    "belebele_eus_Latn": "acc_norm,none",
    "bertaqa_eu_global": "acc,none",
    "bertaqa_eu_local": "acc,none",
    "eus_exams_eu": "acc_norm,none",
    "eus_trivia": "acc,none",
    "mmlu_eu": "acc,none",
    "piqa_eu_mc": "acc_norm,none",
    "eus_reading": "acc,none",
}

models_tasks: List[Tuple[List[str], Dict[str,str | None]]] = [
    # Base
    ([
        "swiss-ai/Apertus-8B-2509",
        "google/gemma-3-12b-pt",
        "google/gemma-3-27b-pt",
        "google/gemma-4-26B-A4B",
        "google/gemma-4-12B",
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen3.5-9B-Base",
        "BSC-LT/salamandra-7b",
        "HiTZ/Latxa-Llama-3.1-8B",
        "orai-nlp/Llama-eus-8B",
        "HiTZ/Gemma-Kimu-9b-base",

        "swiss-ai/Apertus-70B-2509",
        "google/gemma-4-31B",
        "zai-org/GLM-4-32B-Base-0414",
        "meta-llama/Llama-3.1-70B",
        "Qwen/Qwen3.5-35B-A3B-Base",
        "BSC-LT/ALIA-40b",

        "zai-org/GLM-4.5-Air-Base",
        "meta-llama/Llama-4-Scout-17B-16E",
    ],
        tasks_metric | {"ab-base_BAT":  None, "ab-base_BL2MP": None,}
    ),
    # Instruct
    ([
        "swiss-ai/Apertus-8B-Instruct-2509",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-12B-it",
        "zai-org/GLM-4.6V-Flash",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3.5-9B",
        "BSC-LT/salamandra-7b-instruct",
        "HiTZ/Latxa-Llama-3.1-8B-Instruct",
        "HiTZ/Latxa-Qwen3-VL-8B-Instruct",
        "orai-nlp/Llama-eus-8B-DIPLomA",
        "HiTZ/Gemma-Kimu-9b-it",

        "swiss-ai/Apertus-70B-Instruct-2509",
        "google/gemma-4-31B-it",
        "zai-org/GLM-4-32B-0414",
        "zai-org/GLM-4.7-Flash",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-27B",
        "BSC-LT/ALIA-40b-instruct-2605",
        "HiTZ/Latxa-Llama-3.1-70B-Instruct",
        "HiTZ/Latxa-Llama-3.1-70B-Instruct-v2",
        "HiTZ/Latxa-Qwen3-VL-32B-Instruct",

        "zai-org/GLM-4.5-Air",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "Qwen/Qwen3.5-122B-A10B",
        "zai-org/GLM-4.6V",
    ],
        tasks_metric | {"ab-instruct_BAT":  None, "ab-instruct_BL2MP": None,}
    ),
    # Continual Base
    ([
        "HiTZ/berria_60M",
        "HiTZ/BOPV_60M",
        "HiTZ/chimicron_60M",
        "HiTZ/ehkirola_60M",
        "HiTZ/instituzionalak_60M",
        "HiTZ/txioak_60M",
    ],
        tasks_metric | {"ab-base_BAT":  None, "ab-base_BL2MP": None,}
    ),
    # Continual Instruct
    ([
        "HiTZ/exp_2_101-berria_60M-2",
        "HiTZ/exp_2_101-BOPV_60M-2",
        "HiTZ/exp_2_101-chimicron_60M",
        "HiTZ/exp_2_101-ehkirola_60M",
        "HiTZ/exp_2_101-instituzionalak_60M",
        "HiTZ/exp_2_101-txioak_60M",
    ],
        tasks_metric | {"ab-instruct_BAT":  None, "ab-instruct_BL2MP": None,}
    ),
]

def check_results(directory: str, skip_missing: bool=False) -> Dict[str, Dict[str, float | None]]:
    global models_tasks, tasks_metric
    # Initialize results dictionary
    results: Dict[str, Dict[str, float | None]] = {
        model: {task: None for task, _ in tasks.items()}
        for models, tasks in models_tasks
        for model in models
    }
    # retrieve tasks results
    for direc in [f"{directory}/{subdir}" for subdir in ["BENCHMARK", "MC"]]:
        for path in Path(direc).rglob("results*.json"):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

                # eus_exams_eu is composed of multiple sub-tasks
                tasks: List[str] = list(data["results"].keys())
                task_name: str = tasks[0] if len(tasks) == 1 else "eus_exams_eu"

                model: str = data["config"]["model_args"]["pretrained"]
                
                # Skip result from models not present in results dict, also skip taks that are no present in tasks_metric
                if model not in results:
                    if not skip_missing:
                        raise Exception(f"Model {model} not found!")                      
                    else:
                        continue
                elif task_name not in list(tasks_metric.keys()) + ["ab-instruct_BAT", "ab-instruct_BL2MP", "ab-base_BAT", "ab-base_BL2MP"]: 
                    if not skip_missing:
                        raise Exception(f"Task {task_name} not found!")                      
                    else:
                        continue
                        
                
                if task_name in ["ab-instruct_BAT", "ab-instruct_BL2MP", "ab-base_BAT", "ab-base_BL2MP"]: 
                    n_shot: int = data["configs"][task_name]["num_fewshot"]
                    filepath, _ , filedate = str(path).rpartition('/')
                    filedate = filedate.split('_')[-1].split('.json')[0]
                    
                    with open(f"{filepath}/samples_{task_name}_{filedate}.jsonl", 'r') as f2:
                        # Remove few-shot examples
                        accs: List[int] = []
                        for line in f2:
                            data2 = json.loads(line)
                            if task_name in ["ab-base_BAT", "ab-base_BL2MP"] and (data2["arguments"]["gen_args_0"]["arg_0"].count("\nB: ") != n_shot + 1 and data2["arguments"]["gen_args_1"]["arg_0"].count("\nB: ") != n_shot + 1):
                                continue
                            elif task_name in ["ab-instruct_BAT", "ab-instruct_BL2MP"] and (data2["arguments"]["gen_args_0"]["arg_0"].count("\nB: ") != n_shot + 1 and data2["arguments"]["gen_args_1"]["arg_0"].count("\nB: ") != n_shot + 1):
                                continue
                            accs.append(data2["acc"])

                    # Obtain accuracy per A-B prompt
                    accs = np.array_split(accs, 5)
                    accs = [np.mean(acc) for acc in accs]
                    results[model][task_name] = accs
                elif len(tasks) > 1:
                    # Normalize according to subtask sample size
                    acc: List[float]
                    norm: List[int]
                    acc, norm = [], []
                    for subtask in tasks:
                        try:
                            acc.append(data["results"][subtask]["acc_norm,none"])
                        except:
                            acc.append(data["results"][subtask]["acc,none"])
                        norm.append(data["n-samples"][subtask]["effective"])
                    results[model][task_name] = sum(a * n for a, n in zip(acc, norm)) / sum(norm)
                else:
                    results[model][task_name] = data["results"][task_name][tasks_metric[task_name]]
    return results

if __name__ == "__main__":
    path = "<TODO>/basque_linguistic_competence/logs/"
    results = check_results(directory=path, skip_missing=True)
    for model, stats in results.items():
        accs: List[float] = []
        for task, acc in stats.items():
            if acc is None:
                print(f"Missing result for model {model} on task {task}")

    with open("data-2.json", "w") as file:
        file.write(json.dumps(results, indent=4, sort_keys=True))
