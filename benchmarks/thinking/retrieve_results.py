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

def check_results(directory: str, skip_missing: bool=False) -> Dict[str, Dict[str, float | None]]:
    global models_tasks
    # Initialize results dictionary
    results: Dict[str, Dict[str, float | None]] = {
        model: {task: None for task, _ in tasks.items()}
        for models, tasks in models_tasks
        for model in models
    }
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
                for line in f:
                    data = json.loads(line)
                    data1, data2 = re.split(pattern, data["arguments"][0][0], maxsplit=1)[0], re.split(pattern, data["arguments"][1][0], maxsplit=1)[0]

                    if data1.count("\nB: ") != n_shot + 1 and data2.count("\nB: ") != n_shot + 1:
                        cont += 1
                        continue
                    accs.append(data["acc"])
            
            if cont != 25:
                print(model)
            
            # Obtain accuracy per A-B prompt
            accs = np.array_split(accs, 5)
            accs = [np.mean(acc) for acc in accs]
            results[model][f"{task}-{thinking}"] = accs
    return results

if __name__ == "__main__":
    path = "<TODO>/basque_linguistic_competence/logs/"
    results = check_results(directory=path, skip_missing=True)
    for model, stats in results.items():
        accs: List[float] = []
        for task, acc in stats.items():
            if acc is None:
                print(f"Missing result for model {model} on task {task}")

    with open("data-3.json", "w") as file:
        file.write(json.dumps(results, indent=4, sort_keys=True))
