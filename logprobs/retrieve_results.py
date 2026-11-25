from pathlib import Path
import json
from typing import Dict, List, Tuple
import math
import transformers
from tqdm import tqdm

####################################################################################################################################

projectPath: str = "<TODO>/basque_linguistic_competence/"
dataset_mapping: Dict[str, str] = {
    "bl2mp": projectPath + "data/bl2mp.jsonl",
    "bat": projectPath + "data/bat.jsonl",
}
logs_path: str = projectPath + "logs/LP"

base_models: Tuple[List[str], List[str]] = (
    [
        "swiss-ai/Apertus-8B-2509",
        "google/gemma-3-12b-pt",
        "google/gemma-3-27b-pt",
        "google/gemma-4-12B",
        "google/gemma-4-26B-A4B",
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen3.5-9B-Base",
        "BSC-LT/salamandra-7b",
        "HiTZ/Latxa-Llama-3.1-8B",
        "orai-nlp/Llama-eus-8B",
        "orai-nlp/Gemma-Kimu-9b-base",

        "swiss-ai/Apertus-70B-2509",
        "google/gemma-4-31B",
        "zai-org/GLM-4-32B-Base-0414",
        "meta-llama/Llama-3.1-70B",
        "Qwen/Qwen3.5-35B-A3B-Base",
        "BSC-LT/ALIA-40b",

        "zai-org/GLM-4.5-Air-Base",
        "meta-llama/Llama-4-Scout-17B-16E",
        
        "HiTZ/berria_60M",
        "HiTZ/BOPV_60M",
        "HiTZ/chimicron_60M",
        "HiTZ/ehkirola_60M",
        "HiTZ/instituzionalak_60M-2-new",
        "HiTZ/txioak_60M"
    ], 
    [
        "bat-sent",
        "bat-ctxt",
        "bl2mp-ctxt",
        "bl2mp-sent",
    ]
)

instruct_models: Tuple[List[str], List[str]] = (
    [
        "swiss-ai/Apertus-8B-Instruct-2509",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "google/gemma-4-12B-it",
        "google/gemma-4-26B-A4B-it",
        "zai-org/GLM-4.6V-Flash",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3.5-9B",
        "BSC-LT/salamandra-7b-instruct",
        "HiTZ/Latxa-Llama-3.1-8B-Instruct",
        "HiTZ/Latxa-Qwen3-VL-8B-Instruct",
        "orai-nlp/Llama-eus-8B-DIPLomA",
        "orai-nlp/Gemma-Kimu-9b-it",

        "swiss-ai/Apertus-70B-Instruct-2509",
        "google/gemma-4-31B-it",
        "zai-org/GLM-4-32B-0414",
        "zai-org/GLM-4.7-Flash",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
        "BSC-LT/ALIA-40b-instruct-2605",
        "HiTZ/Latxa-Llama-3.1-70B-Instruct",
        "HiTZ/Latxa-Llama-3.1-70B-Instruct-v2",
        "HiTZ/Latxa-Qwen3-VL-32B-Instruct",

        "zai-org/GLM-4.5-Air",
        "zai-org/GLM-4.6V",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "Qwen/Qwen3.5-122B-A10B",
        
        "HiTZ/exp_2_101-berria_60M",
        "HiTZ/exp_2_101-BOPV_60M",
        "HiTZ/exp_2_101-chimicron_60M",
        "HiTZ/exp_2_101-ehkirola_60M",
        "HiTZ/exp_2_101-instituzionalak_60M",
        "HiTZ/exp_2_101-txioak_60M"
        
    ], 
    [
        "bat-inst",
        "bl2mp-inst",
    ]
)

# Add assistant turn ending tokens
ending_tokens: List[str] = [
    "<|eot_id|>", # Llama 3
    "<|eot|>", # Llama 4
    "<end_of_turn>", # Gemma 2, 3
    "<turn|>",  # Gemma 4
    "<|im_end|>", # Qwen3, Salamandra (normally with \n, but minicons provides strip() tokens)
    "<|assistant_end|>", # Apertus, not necessary
    "<|return|>", # gpt-oss
    # GLM does not have
]
####################################################################################################################################

def load_jsonl(input_file: str) -> Tuple[List[str], List[str]]:
    sentence_pairs: List[Tuple[str, str]] = []
    with open(input_file, "r", encoding="utf-8") as json_file:
        for line in json_file:
            entry: Dict[str, str] = json.loads(line)
            assert "sentence_good" in entry and "sentence_bad" in entry, "Minimal pair file must contain fields sentence_good and sentence_bad."
            good: str = entry.get("sentence_good").strip()
            bad: str = entry.get("sentence_bad").strip()
            sentence_pairs.append((good, bad))
    
    sentences_good, sentences_bad = zip(*sentence_pairs)
    assert len(sentences_good) == len(sentences_bad), "Minimal pairs quantity does not match."
    return list(sentences_good), list(sentences_bad)

def get_prompt_id(path: Path, dataset: str, task: str) -> str:
    # File name: {model}-{dataset}-{task}-{date}[-{prompt ids}].jsonl ; date has no '-'
    suffix = path.stem.rpartition(f"-{dataset}-{task}-")[2]
    _, _, prompt_id = suffix.partition("-")
    return prompt_id or "default"

def check_results(directory: str, skip_missing=True, check_tokens: bool=True) -> Dict[str, Dict[str, List[float] | Dict[str, Dict[str, List[int]]]]]:
    results = {base_model: {task: [] for task in base_models[1]} for base_model in base_models[0]} | {instruct_model: {task: [] for task in instruct_models[1]} for instruct_model in instruct_models[0]}
    # Per prompt (run), 0/1 correctness for each minimal pair: {model: {"per_pair": {task: {prompt_id: [0, 1, ...]}}}}
    for model, tasks in results.items():
        tasks["per_pair"] = {task: {} for task in tasks}

    for path in tqdm(list(Path(directory).rglob("*.jsonl"))):
        acc, total = 0, 0
        pair_correct: List[int] = []
        with path.open("r", encoding="utf-8") as f:
            # Read the first line
            first_line = json.loads(f.readline())
            dataset, _ , task = first_line["task"].partition('-')
            model, model_path = first_line["model_id"], first_line["model_path"]
            
            if not skip_missing:
                if task in ["ctxt", "sent"]:
                    assert model in base_models[0], f"{model} {base_models}"
                elif task in ["inst"]:
                    assert model in instruct_models[0], f"{model} {instruct_models}"
                else:
                    raise Exception()

            if check_tokens:
                tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=True, local_files_only=False, trust_remote_code=True)

            # Reset the pointer back to the beginning of the file
            f.seek(0)

            sentences_good, sentences_bad = load_jsonl(dataset_mapping[dataset])

            # Now iterate through the entire file from the start
            for line, sentence_good, sentence_bad in zip(f, sentences_good, sentences_bad):
                data = json.loads(line)

                # Retrieve minimal pair tokens
                idx = data["context_length"]

                ###########################################################################################################################################################
                # Fix bug in context calculation for Gemma-4 Instruct models
                if model in ["google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it", "google/gemma-4-12B-it"]:
                    for i in range(len(data["result_good"]) - 3, -1, -1):
                        if data["result_good"][i][0] == '<|turn>' and data["result_good"][i+1][0] == 'model' and data["result_good"][i+2][0] == '\n':
                            idx = i + 3
                # At time of experimentation bug with bos handling for base models in CTXT setting, now is fixed but need to skip the first token for affected models logs
                # elif model in ["zai-org/GLM-4-32B-Base-0414", "zai-org/GLM-4.5-Air-Base", "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-35B-A3B-Base",] and task == "ctxt":
                #     idx += 1
                ###########################################################################################################################################################

                good, bad = data["result_good"][idx:], data["result_bad"][idx:]
                good_tok, good_logprobs = zip(*good)
                bad_tok, bad_logprobs = zip(*bad)

                # For instruct models, set upper bound via ending_tokens list
                if task == "inst":
                    for i in range(len(good_tok)):
                        if good_tok[-i] in ending_tokens:
                            good_tok = good_tok[:-i]
                            good_logprobs = good_logprobs[:-i]
                            break
                    for i in range(len(bad_tok)):
                        if bad_tok[-i] in ending_tokens:
                            bad_tok = bad_tok[:-i]
                            bad_logprobs = bad_logprobs[:-i]
                            break

                if check_tokens:
                    good_tok, sentence_good = tokenizer.convert_tokens_to_string(good_tok).strip(), sentence_good.strip()
                    assert good_tok == sentence_good, f"{repr(good_tok)} {"sentence_good", repr(sentence_good)} {total, model, dataset, task}"
                    bad_tok, sentence_bad = tokenizer.convert_tokens_to_string(bad_tok).strip(), sentence_bad.strip()
                    assert bad_tok == sentence_bad, f"{repr(bad_tok)} {"sentence_bad", repr(sentence_bad)} {total, model, dataset, task}"

                # Other commonly used metrics: 
                # MeanLP_good = sum(good_logprobs) / len(good_logprobs)
                # MeanLP_bad = sum(bad_logprobs) / len(bad_logprobs)
                # PenLP_good = sum(good_logprobs) / (((5 + len(good_logprobs)) / 6.0) ** 0.8)
                # PenLP_bad = sum(bad_logprobs) / (((5 + len(bad_logprobs)) / 6.0) ** 0.8)
                perplexity_good = math.exp(-sum(good_logprobs) / len(good_logprobs))
                perplexity_bad = math.exp(-sum(bad_logprobs) / len(bad_logprobs))

                correct = int(perplexity_good < perplexity_bad)
                pair_correct.append(correct)
                acc += correct
                total += 1
            try:
                results[model][f"{dataset}-{task}"].append(acc / total)
            except KeyError:
                continue
            prompt_id = get_prompt_id(path, dataset, task)
            assert prompt_id not in results[model]["per_pair"][f"{dataset}-{task}"], f"Duplicate prompt {prompt_id} for {model} {dataset}-{task} ({path})"
            results[model]["per_pair"][f"{dataset}-{task}"][prompt_id] = pair_correct
    return results

if __name__ == "__main__":
    results = check_results(directory=logs_path, skip_missing=False, check_tokens=False)
    for model, stats in results.items():
        for task, acc in stats.items():
            if task == "per_pair":
                continue
            elif "ctxt" in task or "inst" in task:
                assert len(acc) == 8, f"{model} does not have all 8 prompts for {task}."
            elif "sent" in task:
                assert len(acc) == 1, f"{model} does not have all 1 prompts for {task}."
            else:
                print(f"Missing result for model {model} on task {task}")

    with open("data.json", "w") as file:
        file.write(json.dumps(results, indent=4, sort_keys=True))
