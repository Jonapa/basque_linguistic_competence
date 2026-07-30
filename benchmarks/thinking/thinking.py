from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval import simple_evaluate
from dataclasses import asdict
import os
import datetime
import json
import argparse
from typing import List, Dict, Any
import gc
import torch

# Global variables to store reasoning traces and the index for the current trace. DO NOT DELETE THEM!
reasoning_traces: List[str] = []
reasoning_i: int = 0

# We hard-code the reasoning tokens and parsers for different models based on their jinja chat templates. 
reasoning_tokens: Dict[str, Dict[str, str]] = {
    # https://huggingface.co/zai-org/GLM-4.5/blob/main/chat_template.jinja
    # https://huggingface.co/zai-org/GLM-4.6V/blob/main/chat_template.jinja
    # enable_thinking = True
    # {{ '\n<think>' + reasoning_content.strip() +  '</think>'}}{{ '\n' + content.strip() }}
    # enable_thinking = False
    # /nothink<|assistant|>{{ '\n<think></think>' }}
    "GLM-4.5": {
        "bot_token": "\n<think>",
        "eot_token": "</think>\n",
        "reasoning_parser": "glm45",
    },
    "GLM-4.6": {
        "bot_token": "\n<think>",
        "eot_token": "</think>\n",
        "reasoning_parser": "glm45",
    },
    # https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/chat_template.jinja
    # enable_thinking = True
    # {{ '<think>' + reasoning_content.strip() +  '</think>'}}{{ content.strip() }}
    # enable_thinking = False
    # {{ '</think>' }}
    "GLM-4.7": {
        "bot_token": "<think>",
        "eot_token": "</think>",
        "reasoning_parser": "glm45",
    },
    # https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja
    # enable_thinking = True
    # {{- '<|turn>' + role + '\n' }}<|channel>thought\n[Internal reasoning]<channel|>[Final answer]
    # enable_thinking = False
    # {{- '<|turn>' + role + '\n' }}<|channel>thought\n<channel|>[Final answer]
    "gemma-4": {
        "bot_token": "\n<|channel>thought\n",
        "eot_token": "<channel|>",
        "reasoning_parser": "gemma4",
    },
    # https://huggingface.co/Qwen/Qwen3-235B-A22B?chat_template=default
    # https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/chat_template.jinja
    # enable_thinking = True
    # {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
    # enable_thinking = False
    # \n<think>\n\n</think>\n\n
    "Qwen3": {
        "bot_token": "\n<think>\n",
        "eot_token": "\n</think>\n\n",
        "reasoning_parser": "qwen3",
    },
    # TODO: Add support to gpt-oss (reasoning_effort!)
    #     # https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja
    #     # In system message: {{- "Reasoning: " + reasoning_effort + "\n\n" }} -----------------------> In metadata: "reasoning_effort"= "low", "medium" or "high" 
    #     {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }} {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
    # "gpt-oss": {
    #     "bot_token": "<|start|>assistant<|channel|>analysis<|message|>",
    #     "eot_token": "<|end|><|start|>assistant<|channel|>final<|message|>",
    #     "reasoning_parser": "openai_gptoss",
    # },
}

def load_config(args: argparse.Namespace, update: bool) -> EvaluatorConfig:
    if (match := next((m for m in list(reasoning_tokens.keys()) if m in args.model), None)):
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/config/evaluate_config.py
        config = EvaluatorConfig(
            model="vllm",
            model_args={
                "pretrained": args.model,
                "dtype": "bfloat16",
                "tensor_parallel_size": args.tensor_parallel_size, 
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len, 
                "think_end_token": reasoning_tokens[match]["eot_token"],
                "reasoning_parser": reasoning_tokens[match]["reasoning_parser"],
                "enable_thinking": True,
            },
            tasks=[args.task],
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device="cuda",
            write_out=True,
            log_samples=True,
            output_path=args.output_path,  
            predict_only=True,                                          # Set True to skip evaluation and only get model generations       
            system_instruction=args.system_instruction if args.system_instruction is not None else None,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            show_config=True,                                           # Not used
            include_path=args.include_path if args.include_path is not None else None,
            gen_kwargs={
                "until": [reasoning_tokens[match]["eot_token"].strip()],
                **args.gen_kwargs
            }, 
            verbosity="DEBUG",
            trust_remote_code=True,                                     # Not used
            confirm_run_unsafe_code=True,
            metadata={
                "add_reasoning_trace": False, 
                "bot_token": reasoning_tokens[match]["bot_token"], 
                "eot_token": reasoning_tokens[match]["eot_token"],
                **(args.metadata or {}) # TODO: For gptoss, we will add "reasoning_effort" in metadata to indicate the level of reasoning effort (e.g., low, medium, high)
            },
        )

        if update:
            # https://docs.vllm.ai/en/latest/features/reasoning_outputs/#supported-models
            # We DO NOT set enable_thinking to False in model_args (for multiple-choice mode does not matter since it will not generate new tokens, but for hard-coding the reasoning trace is essential to locate the corresponding place)
            config.gen_kwargs = None  # multiple-choice mode does no generate any tokens
            config.predict_only = False # Set False to perform evaluation 
            config.metadata["add_reasoning_trace"] = True # We need to insert the reasoning traces in the prompts, and will use the beginning of thinking token to detect it

            # Change model to HF
            config.model = "hf-multimodal" if ("GLM-4.6V" in args.model or "Qwen3" in args.model) else "hf"
            if config.model == "hf-multimodal":
                config.batch_size = 1 # Batch size 'auto' is not yet supported for hf-multimodal models.
            if "gemma-4-26B-A4B-it" in args.model:
                args.max_model_len = 14000
            elif "gemma-4-31B-it" in args.model:
                args.max_model_len = 12000
                
            config.model_args = {
                "pretrained": args.model,
                "dtype": "bfloat16",
                "parallelize": True,
                "max_length": args.max_model_len,
                "enable_thinking": True,
            }
    else:
        raise ValueError(f"Model {args.model} is not supported. Please make sure the model name contains one of the following: {list(reasoning_tokens.keys())}")

    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Specify the model name")
    parser.add_argument("--tensor_parallel_size", type=int, required=True, help="Specify the tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, required=True, help="Specify the GPU memory utilization for vLLM (between 0 and 1)")
    parser.add_argument("--max_model_len", type=int, required=True, help="Specify the maximum model length for vLLM")
    parser.add_argument("--task", type=str, required=True, help="Specify the task name")
    parser.add_argument("--num_fewshot", type=int, required=True, help="Specify the number of few-shot examples")
    parser.add_argument("--batch_size", type=str, required=True, help="Specify the batch size for evaluation")
    parser.add_argument("--output_path", type=str, required=True, help="Specify the output path for saving results jsonl")
    parser.add_argument("--gen_kwargs", type=json.loads, required=True, help="Specify the generation kwargs as a JSON string")
    # Not necessary arguments
    parser.add_argument("--system_instruction", type=str, help="Specify the system instunction for the model")
    parser.add_argument("--include_path", type=str, help="Specify the path to include external yaml task config files")
    parser.add_argument("--metadata", type=json.loads, help="Specify additional metadata as a JSON string")
    args: argparse.Namespace = parser.parse_args()

    config: EvaluatorConfig = load_config(args=args, update=False)
    task_manager: TaskManager = config.process_tasks()

    ###################################################################################################
    # First, we update the multiple-choice task config to be compatible with generate_until evaluation
    ###################################################################################################
    task_dict = get_task_dict([args.task], task_manager) 
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/config/task.py
    task_dict: Dict[str, Any] = asdict(task_dict[args.task].config)

    # Set as generate_until task
    task_dict["output_type"] = "generate_until"
    # Remove multiple-choice specific fields that are not needed for generate_until
    for key in ["metric_list", "doc_to_target", "doc_to_choice"]:
        if key in task_dict:
            task_dict[key] = None
    # Override doc_to_target to 0 to avoid errors in evaluation (although --predict_only=True should skip evaluation, still process_results is called)
    task_dict["doc_to_target"] = 0

    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py
    # Results not needed, but may cause some CUDA OOM issues if not deleted
    results = simple_evaluate(
        model=config.model,
        model_args=config.model_args,
        tasks=[task_dict],
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        write_out=config.write_out,
        log_samples=config.log_samples,
        system_instruction=config.system_instruction,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=config.fewshot_as_multiturn,
        gen_kwargs=config.gen_kwargs,
        task_manager=task_manager,
        verbosity=config.verbosity,
        predict_only=config.predict_only, 
        confirm_run_unsafe_code=config.confirm_run_unsafe_code, 
    )

    del results
    gc.collect()
    torch.cuda.empty_cache()

    ###################################################################################################
    # Secondly, we update the config to suit multiple-choice task
    ###################################################################################################

    # Reset config and update it to suit multiple-choice evaluation (e.g., disable predict_only to perform evaluation, and add reasoning traces in the prompts)
    config = load_config(args=args, update=True)
    task_manager = config.process_tasks()
    task_dict = get_task_dict([args.task], task_manager) 
    task_dict = asdict(task_dict[args.task].config)

    results = simple_evaluate(
        model=config.model,
        model_args=config.model_args, 
        tasks=[task_dict],
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        write_out=config.write_out,
        log_samples=config.log_samples,
        system_instruction=config.system_instruction,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=config.fewshot_as_multiturn,
        gen_kwargs=config.gen_kwargs,
        task_manager=task_manager,
        verbosity=config.verbosity,
        predict_only=config.predict_only,
        confirm_run_unsafe_code=config.confirm_run_unsafe_code, 
    )

    os.makedirs(config.output_path, exist_ok=True)
    output_file = os.path.join(config.output_path, config.model_args["pretrained"].split("/")[-1] + f"_{args.task}-thinking_{datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in results["samples"][args.task]:
            f.write(json.dumps(sample) + "\n")

    del results
    gc.collect()
    torch.cuda.empty_cache()
    
    reasoning_traces.clear()
    reasoning_i = 0