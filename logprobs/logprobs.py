import argparse
import yaml
import pathlib
from typing import Dict, Tuple, List, Any
import json
import transformers
import itertools
import torch
from minicons import scorer
import gc
import datetime
from tqdm import tqdm

# Add model IDs that do not support system messages
no_system_support = [
    # No system message supported in Gemma-2
    "orai-nlp/Gemma-Kimu-9b-it",
]

# Some models do not add a bos token at the beginning
bos_token = [
    "zai-org/GLM-4-32B-Base-0414",
    "zai-org/GLM-4.5-Air-Base",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-35B-A3B-Base",
]

MAX_TOKENS = 512

def set_seed(seed: int) -> None:
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    # https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/trainer_utils.py#L61
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    os.environ["HCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_jsonl(input_file: str, tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast, sort: bool = False) -> Tuple[List[str], List[str]]:
    sentence_pairs: List[Tuple[str, str, int]] = []
    with open(input_file, "r", encoding="utf-8") as json_file:
        for line in json_file:
            entry: Dict[str, str] = json.loads(line)
            assert "sentence_good" in entry and "sentence_bad" in entry, "Minimal pair file must contain fields sentence_good and sentence_bad."
            good: str = entry.get("sentence_good").strip()
            bad: str = entry.get("sentence_bad").strip()
            sentence_pairs.append((good, bad, len(tokenizer.encode(text=good, add_special_tokens=False, padding=False, truncation=False)) + len(tokenizer.encode(text=bad, add_special_tokens=False, padding=False, truncation=False))))
    # Sort in accordance to token length so that padding is more efficient
    if sort:
        sentence_pairs.sort(key=lambda x: x[2])
    
    sentences_good, sentences_bad, _ = zip(*sentence_pairs)
    assert len(sentences_good) == len(sentences_bad), "Minimal pairs quantity does not match."
    return list(sentences_good), list(sentences_bad)

def encode_batched_pairs_base(task: str, pairs: Tuple[List[str], List[str]], tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast, batch_size: int, **kwargs) -> Tuple[Tuple[List[transformers.tokenization_utils_base.BatchEncoding], List[transformers.tokenization_utils_base.BatchEncoding]], int | None]:
    sentences_good: List[str]
    sentences_bad: List[str]
    sentences_good, sentences_bad = pairs
    context_length: int | None = None
    
    if task == "ctxt":
        assert "separator" in kwargs and "user" in kwargs, "Separator and user prompt must be provided for ctxt task."
        context: str = kwargs.get("user") + kwargs.get("separator")
        if "assistant" in kwargs:
            assert "separator_2" in kwargs, "Separator must be provided for assistant text"
            context = context + kwargs.get("assistant") + kwargs.get("separator_2")
    else:
        assert not "separator" in kwargs and not "user" in kwargs, "Separator and user prompt must NOT be provided for sent task."
        if "assistant" in kwargs:
            assert "separator_2" in kwargs, "Separator must be provided for assistant text"
            context = kwargs.get("assistant") + kwargs.get("separator_2")

    if task == "ctxt" or "assistant" in kwargs:
        context_length = len(tokenizer.encode(context, add_special_tokens=True, padding=False, truncation=False))
        sentences_good: List[str] = [
            context + sentence_good
            for sentence_good in sentences_good
        ]
        sentences_bad: List[str] = [
            context + sentence_bad
            for sentence_bad in sentences_bad
        ]

    if kwargs.get("bos_token"):
        sentences_good: List[str] = [tokenizer.bos_token + sentence_good for sentence_good in sentences_good]
        sentences_bad: List[str] = [tokenizer.bos_token + sentence_bad for sentence_bad in sentences_bad]
        if task == "ctxt":
            context_length += 1
    
    batches_good: List[transformers.tokenization_utils_base.BatchEncoding] = []
    batches_bad: List[transformers.tokenization_utils_base.BatchEncoding] = []    
    for batch in itertools.batched(zip(sentences_good, sentences_bad), batch_size):
        batch_good, batch_bad = zip(*batch)
        batches_good.append(tokenizer(
                            batch_good,
                            padding='max_length',
                            truncation=True,
                            max_length=MAX_TOKENS,
                            return_tensors="pt",
                        ))
        batches_bad.append(tokenizer(
                            batch_bad,
                            padding='max_length',
                            truncation=True,
                            max_length=MAX_TOKENS,
                            return_tensors="pt",
                        ))
    return (batches_good, batches_bad), context_length

def encode_batched_pairs_instruct(pairs: Tuple[List[str], List[str]], tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast, batch_size: int, **kwargs) -> Tuple[Tuple[List[transformers.tokenization_utils_base.BatchEncoding], List[transformers.tokenization_utils_base.BatchEncoding]], int]:
    sentences_good: List[str]
    sentences_bad: List[str]
    sentences_good, sentences_bad = pairs

    context: List[Dict[str, Any]] = []
    if system_content := kwargs.get("system"):
        context.append({
            "role": "system",
            "content": system_content if "vision" not in model_type else [{"type": "text", "text": system_content}]
        })

    assert "separator" in kwargs and "user" in kwargs, "Separator and user prompt must be provided for inst task."
    user_content = kwargs.get("user") + kwargs.get("separator")
    context.append({
        "role": "user", 
        "content": user_content if "vision" not in model_type else [{"type": "text", "text": user_content}]
    })

    if "assistant" in kwargs:
        assert "separator_2" in kwargs, "Separator must be provided for assistant text"
        preappend: str = kwargs.get("assistant") + kwargs.get("separator_2")
        context_length: int = len(tokenizer.apply_chat_template(
                            context + [{"role": "assistant", "content": preappend if "vision" not in model_type else [{"type": "text", "text": preappend}]}],
                            tokenize=True,
                            padding=False,
                            truncation=False,
                            add_generation_prompt=False,
                            return_dict=False,
                            continue_final_message=True,
                            enable_thinking=False,
                        ))
    else:
        context_length: int = len(tokenizer.apply_chat_template(
                                    context,
                                    tokenize=True,
                                    padding=False,
                                    truncation=False,
                                    add_generation_prompt=True,
                                    return_dict=False,
                                    enable_thinking=False,
                                ))
        preappend: str = ""

    sentences_good: List[List[Dict[str, Any]]] = [
        context + [{
            "role": "assistant",
            "content": (preappend + sentence_good) if "vision" not in model_type 
                    else [{"type": "text", "text": preappend + sentence_good}]
        }]
        for sentence_good in sentences_good
    ]

    sentences_bad: List[List[Dict[str, Any]]] = [
        context + [{
            "role": "assistant",
            "content": (preappend + sentence_bad) if "vision" not in model_type 
                    else [{"type": "text", "text": preappend + sentence_bad}]
        }]
        for sentence_bad in sentences_bad
    ]

    batches_good: List[transformers.tokenization_utils_base.BatchEncoding] = []
    batches_bad: List[transformers.tokenization_utils_base.BatchEncoding] = []
    for batch in itertools.batched(zip(sentences_good, sentences_bad), batch_size):
        batch_good, batch_bad = zip(*batch)
        batches_good.append(tokenizer.apply_chat_template(
                            batch_good,
                            tokenize=True,
                            padding='max_length',
                            truncation=True,
                            max_length=MAX_TOKENS,
                            add_generation_prompt=False,
                            return_dict=True,
                            return_tensors="pt",
                            enable_thinking=False,
                        ))
        batches_bad.append(tokenizer.apply_chat_template(
                            batch_bad,
                            tokenize=True,
                            padding='max_length',
                            truncation=True,
                            max_length=MAX_TOKENS,
                            add_generation_prompt=False,
                            return_dict=True,
                            return_tensors="pt",
                            enable_thinking=False,
                        ))
    return (batches_good, batches_bad), context_length

def report_results(file_path: str, batches: Tuple[List[transformers.tokenization_utils_base.BatchEncoding], List[transformers.tokenization_utils_base.BatchEncoding]], context_length: int | None, model, model_id: str, model_path: str, model_type: str, task: str) -> None:
    if model_type == "masked":
        context_length = 0 if context_length is None else context_length - 2
    else:
        context_length = 1 if context_length is None else context_length
    with open(file_path, "w") as f:
        for batch_good, batch_bad in tqdm(zip(*batches)):
            if model_type in ["base", "instruct"]:
                # https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py#L1939
                # minicons adds an additional 0.0 log-probability for the first token/word as convention.
                results_good: List[List[Tuple[str, float]]] = model.token_score(batch=batch_good, surprisal=False, prob=False, base_two=False, rank=False, decode=False, bow_correction=True)
                results_bad: List[List[Tuple[str, float]]] = model.token_score(batch=batch_bad, surprisal=False, prob=False, base_two=False, rank=False, decode=False, bow_correction=True)
            elif model_type == "masked":
                # https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py#L1226
                results_good: List[List[Tuple[str, float]]] = model.token_score(batch=batch_good, surprisal=False, prob=False, base_two=False, rank=False, PLL_metric="within_word_l2r")
                results_bad: List[List[Tuple[str, float]]] = model.token_score(batch=batch_bad, surprisal=False, prob=False, base_two=False, rank=False, PLL_metric="within_word_l2r")
            elif "vision" in model_type:
                results_good: List[List[Tuple[str, float]]] = model.token_score(text_batch=batch_good, image_batch=None, surprisal=False, prob=False, base_two=False, rank=False, decode=False, bow_correction=True)
                results_bad: List[List[Tuple[str, float]]] = model.token_score(text_batch=batch_bad, image_batch=None, surprisal=False, prob=False, base_two=False, rank=False, decode=False, bow_correction=True)
            else:
                raise Exception(f"Unknown model_type {model_type}")

            for result_good, result_bad in zip(results_good, results_bad):
                data = {
                    "task": task,
                    "model_id": model_id,
                    "model_path": model_path,
                    "result_good": result_good,
                    "result_bad": result_bad,
                    "context_length": context_length,
                }
                f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Must be included in the configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Sentence probability readout (sent), context probability readout (ctxt), or instruction probability readout (inst).",
    )
    parser.add_argument(
        "--minimal_file",
        type=str,
        required=True,
        help="File with minimal pairs to evaluate. Must be included in the configuration file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for minimal pairs; MUST BE BETWEEN 1 AND 64.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to store the results.",
    )
    args: argparse.Namespace = parser.parse_args()

    # 0.- Set seed for reproducibility
    set_seed(42)
    # 1.- Configuration file handling
    config_path = pathlib.Path(args.config_file)
    assert config_path.is_file(), f"Configuration file {config_path} does not exist."
    with open(args.config_file, "r") as file:
        data: Dict[str, List[Dict[str, str | bool]] | str] = yaml.safe_load(file)
    # 2.- Validate batch size
    batch_size = args.batch_size
    assert batch_size is not None and isinstance(batch_size, int) and  0 < batch_size <= 64, "Invalid batch size. Set an integer between 1 and 64."
    # 3.- Validate model
    model: Dict[str, str | bool] | None = next((m for m in data.get("model", []) if m.get("id") == args.model_id), None)
    assert model is not None, f"Model ID '{args.model_id}' not found in configuration file."
    model_type: str | None = model.get("model_type", None)
    assert model_type is not None and isinstance(model_type, str), f"Model ID '{args.model_id}' model_type attribute not found."
    model_type = model_type.lower()
    assert model_type in ["base", "instruct", "vision-base", "vision-instruct", "masked"], f"model_type must have value base, instruct, masked or vision-instruct, has {model_type}"
    model_path: str | None = model.get("path", None)
    assert model_path is not None and isinstance(model_path, str), f"Model ID '{args.model_id}' path attribute not found."
    model_path: str = model_path or args.model_id
    model_id: str = args.model_id
    assert model_id.count('/') == 1, f"Invalid model ID: '{args.model_id}'. Expected a format like 'org/model-id'."
    # 4.- Validate task and model compatibility
    task = args.task
    assert task in ["sent", "ctxt", "inst"], f"Task '{task}' is invalid. Choose from 'sent', 'ctxt', or 'inst'."
    assert task == "inst" and "instruct" in model_type or ((task == "sent" or task == "ctxt") and "instruct" not in model_type), f"Task '{task}' is incompatible with model '{args.model_id}' (model_type={model_type})."
    # 5.- Validate minimal pair file
    minimal: Dict[str, str] = next((m for m in data.get('minimal', []) if m.get('id') == args.minimal_file), None)
    assert minimal is not None, f"Minimal file ID '{args.minimal_file}' not found in configuration file."
    minimal_path: str | None = minimal.get("path", None)
    assert minimal_path is not None and isinstance(minimal_path, str), f"Minimal file ID '{args.minimal_file}' path attribute not found."
    minimal_id: str | None = minimal.get("id")
    assert pathlib.Path(minimal_path).is_file(), f"Minimal pair file {minimal_path} does not exist."
    # 6.- Validate output directory
    output_dir = args.output_dir
    assert pathlib.Path(output_dir).is_dir(), f"Output directory {output_dir} does not exist."

    del parser, args, config_path, file, model, minimal, pathlib, argparse, yaml

    assert torch.cuda.is_available(), "CUDA is not available"

    if "Llama-4" in model_path:
        model = transformers.Llama4ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16)
    elif "GLM-4.6V-Flash" in model_path:
        assert "vision" in model_type, f"Model '{model_path}' is incompatible with model_type (model_type={model_type})."
        model = transformers.Glm4vForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
    elif "GLM-4.6V" in model_path:
        assert "vision" in model_type, f"Model '{model_path}' is incompatible with model_type (model_type={model_type})."
        model = transformers.Glm4vMoeForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
    elif "Qwen3-VL" in model_path:
        assert "vision" in model_type, f"Model '{model_path}' is incompatible with model_type (model_type={model_type})."
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
    elif "gemma-3" in model_path:
        # assert "vision" in model_type, f"Model '{model_path}' is incompatible with model_type (model_type={model_type})."
        model = transformers.Gemma3ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
    elif "gemma-4" in model_path:
        # Multimodal approach does not seem to be working properly with Gemma 4. Set text-only approach.
        # assert "vision" in model_type, f"Model '{model_path}' is incompatible with model_type (model_type={model_type})."
        model = transformers.Gemma4ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
    elif "Qwen3.5" in model_path:
        assert "vision" in model_type, f"Model '{model_path}' is incompatible with model_type (model_type={model_type})."
        if "A3B" in model_path or "A10B" in model_path:
            model = transformers.Qwen3_5MoeForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
        else:
            model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, dtype=torch.bfloat16)
    else:
        if model_type != "masked":
            model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16)
        else:
            model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=False, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16)
    model.eval()

    if "vision" not in model_type:
        tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=True, local_files_only=False, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_path).tokenizer
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"]})
            tokenizer.pad_token = "<pad>"
            model.resize_token_embeddings(len(tokenizer))
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|bos|>"]}
            )
            tokenizer.bos_token = "<|bos|>"
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"

    # https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py#L1354
    # https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py#L1405 (device="auto" set in IncrementalLMScorer and then overwritten because if model already instatiated as done in device_map="auto" (1392: if isinstance(model, str)) and device is set to self.device != "auto", double allocation and multiple-GPU errors occur)
    if model_type in ["base", "instruct"]:
        model = scorer.IncrementalLMScorer(model=model, tokenizer=tokenizer, device="auto")
    elif "vision" in model_type:
        # https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py#L3791C7-L3791C16
        model = scorer.VLMScorer(model=model, tokenizer=tokenizer, causallm=True, device="auto")
    elif model_type == "masked":
        # https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py#L485
        model = scorer.MaskedLMScorer(model=model, tokenizer=tokenizer, device="auto", PLL_metric="within_word_l2r")
    else:
        raise Exception(f"Unknown model_type {model_type}")

    model.device = model.model.device

    pairs: Tuple[List[str], List[str]] = load_jsonl(input_file=minimal_path, tokenizer=tokenizer, sort=False)
    
    del load_jsonl, minimal_path

    file_date = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    if task == "sent" and "instruct" not in model_type:
        if assistants := data.get("assistant"):
            assert "assistant-minimal_separator" in data, "Separator must be defined in configuration file."
            assistants = list(assistants)
            for assistant in assistants:
                assert "id" in assistant and "message" in assistant, f"Configuration file must provide an id and message for assistant messages."
                batches, context_length = encode_batched_pairs_base(task=task, pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, assistant=assistant.get("message").strip(), separator_2=data.get("assistant-minimal_separator"), bos_token=True if model_id in bos_token else False)
                report_results(file_path=f"{output_dir}/{model_id.split("/", 1)[-1]}-{minimal_id}-{task}-{file_date}-{assistant.get("id")}.jsonl", batches=batches, context_length=context_length, model=model, model_id=model_id, model_path=model_path, model_type=model_type, task=f"{minimal_id}-{task}")
            del assistants, assistant
        else:
            batches, context_length = encode_batched_pairs_base(task=task, pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, bos_token=True if model_id in bos_token else False)
            report_results(file_path=f"{output_dir}/{model_id.split("/", 1)[-1]}-{minimal_id}-{task}-{file_date}.jsonl", batches=batches, context_length=context_length, model=model, model_id=model_id, model_path=model_path, model_type=model_type, task=f"{minimal_id}-{task}")
    elif task == "ctxt" and "instruct" not in model_type:
        assert "base_separator" in data, "Separator must be defined in configuration file."
        assert "user" in data, "User prompts mut be provided for ctxt task."
        for user in data.get("user"):
            assert "id" in user and "message" in user, f"Configuration file must provide an id and message for user messages."
            if assistants := data.get("assistant"):
                assert "assistant-minimal_separator" in data, "Separator must be defined in configuration file."
                assistants = list(assistants)
                for assistant in assistants:
                    assert "id" in assistant and "message" in assistant, f"Configuration file must provide an id and message for assistant messages."
                    batches, context_length = encode_batched_pairs_base(task=task, pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, user=user.get("message").strip(), separator=data.get("base_separator"), assistant=assistant.get("message").strip(), separator_2=data.get("assistant-minimal_separator"), bos_token=True if model_id in bos_token else False)
                    report_results(file_path=f"{output_dir}/{model_id.split("/", 1)[-1]}-{minimal_id}-{task}-{file_date}-{user.get("id")}-{assistant.get("id")}.jsonl", batches=batches, context_length=context_length, model=model, model_id=model_id, model_path=model_path, model_type=model_type, task=f"{minimal_id}-{task}")
                del assistants, assistant
            else:
                batches, context_length = encode_batched_pairs_base(task=task, pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, user=user.get("message").strip(), separator=data.get("base_separator"), bos_token=True if model_id in bos_token else False)
                report_results(file_path=f"{output_dir}/{model_id.split("/", 1)[-1]}-{minimal_id}-{task}-{file_date}-{user.get("id")}.jsonl", batches=batches, context_length=context_length, model=model, model_id=model_id, model_path=model_path, model_type=model_type, task=f"{minimal_id}-{task}")
        del user
    elif task == "inst" and "instruct" in model_type:
        assert "instruct_separator" in data, "Separator must be defined in configuration file."
        assert "system" in data, "System prompt mut be provided for inst task."
        for system in data.get("system"):
            assert "id" in system and "message" in system, f"Configuration file must provide an id and message for system messages."
            assert "user" in data, "User prompts mut be provided for ctxt task."
            for user in data.get("user"):
                assert "id" in user and "message" in user, f"Configuration file must provide an id and message for user messages."
                if assistants := data.get("assistant"):
                    assert "assistant-minimal_separator" in data, "Separator must be defined in configuration file."
                    assistants = list(assistants)
                    for assistant in assistants:
                        assert "id" in assistant and "message" in assistant, f"Configuration file must provide an id and message for assistant messages."
                        if model_id in no_system_support:
                            batches, context_length = encode_batched_pairs_instruct(pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, user=user.get("message").strip(), separator=data.get("instruct_separator"), assistant=assistant.get("message").strip(), separator_2=data.get("assistant-minimal_separator"), model_type=model_type)
                        else:
                            batches, context_length = encode_batched_pairs_instruct(pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, system=system.get("message").strip(), user=user.get("message").strip(), separator=data.get("instruct_separator"), assistant=assistant.get("message").strip(), separator_2=data.get("assistant-minimal_separator"), model_type=model_type)
                        report_results(file_path=f"{output_dir}/{model_id.split("/", 1)[-1]}-{minimal_id}-{task}-{file_date}-{system.get("id")}-{user.get("id")}-{assistant.get("id")}.jsonl", batches=batches, context_length=context_length, model=model, model_id=model_id, model_path=model_path, model_type=model_type, task=f"{minimal_id}-{task}")
                    del assistants, assistant
                else:
                    if model_id in no_system_support:
                        batches, context_length = encode_batched_pairs_instruct(pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, user=user.get("message").strip(), separator=data.get("instruct_separator"), model_type=model_type)
                    else:
                        batches, context_length = encode_batched_pairs_instruct(pairs=pairs, tokenizer=tokenizer, batch_size=batch_size, system=system.get("message").strip(), user=user.get("message").strip(), separator=data.get("instruct_separator"), model_type=model_type)
                    report_results(file_path=f"{output_dir}/{model_id.split("/", 1)[-1]}-{minimal_id}-{task}-{file_date}-{system.get("id")}-{user.get("id")}.jsonl", batches=batches, context_length=context_length, model=model, model_id=model_id, model_path=model_path, model_type=model_type, task=f"{minimal_id}-{task}")
        del system, user

    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    del Dict, Tuple, List, transformers, itertools, encode_batched_pairs_base, encode_batched_pairs_instruct, data, batch_size, model_type, task, pairs, model_path, model_id, scorer, output_dir, tokenizer, model, batches, context_length, report_results, json, file_date, minimal_id, no_system_support, bos_token
    assert list({name: value for name, value in globals().items() if not name.startswith("_")}.keys()).sort() == ["torch", "gc"].sort()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    del gc, torch
