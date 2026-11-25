from pathlib import Path
import json
from typing import Dict, List, Tuple
import numpy as np
import re

models_tasks: List[Tuple[List[str], Dict[str,str | None]]] = [
    # Instruct
    ([
        "gemma-4-12B-it",
        "gemma-4-31B-it",
        "gemma-4-26B-A4B-it",
        "Qwen3.5-9B",
        "Qwen3.5-27B",
        "Qwen3.5-35B-A3B",
        "Qwen3.5-122B-A10B",
        "Qwen3.6-27B",
        "Qwen3.6-35B-A3B",
        "GLM-4.5-Air",
        "GLM-4.6V",
        "GLM-4.6V-Flash",
    ],
        {"ab-instruct_BAT-THINKING":  None, "ab-instruct_BL2MP-THINKING": None, "ab-instruct_BAT-THINKING-SAMPLING":  None, "ab-instruct_BL2MP-THINKING-SAMPLING": None,}
    ),
]

bot_tokens = ["\n<think>\n", "\n<|channel>thought\n", "<think>"]
pattern = "|".join(map(re.escape, bot_tokens))
pattern = f"(?:{pattern})(?!<)"

N_AB_PROMPTS: int = 5  # Number of prompts in missing_tasks/a_b/utils.py (prompts_ab)

def check_results(directory: str, skip_missing: bool=False) -> Dict[str, Dict[str, float | None]]:
    global models_tasks
    # Initialize results dictionary
    results: Dict[str, Dict[str, float | None]] = {
        model: {task: None for task, _ in tasks.items()}
        for models, tasks in models_tasks
        for model in models
    }
    # Per A-B prompt, 0/1 correctness for each minimal pair: {model: {"per_pair": {task: {"system_0-user_k": [0, 1, None, ...]}}}}
    # Lists are indexed by the pair position in the dataset; None marks pairs used as few-shot examples (excluded from accuracy).
    for model in results:
        results[model]["per_pair"] = {}
    # retrieve tasks results
    for direc in [f"{directory}/{subdir}" for subdir in ["THINKING", "THINKING-SAMPLING"]]:
        for path in Path(direc).rglob("*.jsonl"):
            filepath, _ , filename = str(path).rpartition('/')
            _, _, thinking = filepath.rpartition('/')
            model, _, task = filename.partition('_')
            task, _, _ = task.partition('-thinking_')
            n_shot = 5
            accs: List[int] = []
            cont = 0
            with path.open("r") as f:
                samples = [json.loads(line) for line in f]
                assert len(samples) % N_AB_PROMPTS == 0, f"Unexpected number of samples in {path}"
                n_pairs: int = len(samples) // N_AB_PROMPTS
                pair_accs: List[List[int | None]] = [[None] * n_pairs for _ in range(N_AB_PROMPTS)]
                for data in sorted(samples, key=lambda x: x["doc_id"]):
                    data1, data2 = re.split(pattern, data["arguments"][0][0], maxsplit=1)[0], re.split(pattern, data["arguments"][1][0], maxsplit=1)[0]

                    if data1.count("\nB: ") != n_shot + 1 and data2.count("\nB: ") != n_shot + 1:
                        cont += 1
                        continue
                    accs.append(data["acc"])
                    # Docs are ordered prompt-major: doc_id = prompt * n_pairs + pair
                    prompt_idx, pair_idx = divmod(data["doc_id"], n_pairs)
                    pair_accs[prompt_idx][pair_idx] = int(data["acc"])
            
            if cont != 25:
                print(model)
            
            # Obtain accuracy per A-B prompt
            accs = np.array_split(accs, 5)
            accs = [np.mean(acc) for acc in accs]
            results[model][f"{task}-{thinking}"] = accs
            # Same prompt naming as logprobs results (instruct tasks): system_0-user_k
            results[model]["per_pair"][f"{task}-{thinking}"] = {f"system_0-user_{k}": pair_acc for k, pair_acc in enumerate(pair_accs)}
            for acc, pair_acc in zip(accs, pair_accs):
                valid = [a for a in pair_acc if a is not None]
                assert np.isclose(acc, np.mean(valid)), f"Per-pair results do not match prompt accuracy for {model} {task}-{thinking}"
    return results

if __name__ == "__main__":
    path = "<TODO>/basque_linguistic_competence/logs/"
    results = check_results(directory=path, skip_missing=True)
    for model, stats in results.items():
        accs: List[float] = []
        for task, acc in stats.items():
            if task != "per_pair" and acc is None:
                print(f"Missing result for model {model} on task {task}")

    with open("data-3.json", "w") as file:
        file.write(json.dumps(results, indent=4, sort_keys=True))
