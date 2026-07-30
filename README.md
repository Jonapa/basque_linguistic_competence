# Measuring and Improving Linguistic Competence in Minority Languages: A Case Study for Basque

## Repository overview

This repository brings together the resources used to study Basque linguistic competence in language models: benchmark evaluation, acceptability scoring, curriculum training, and reproducible environments. The main components are organized as follows:

```text
basque_linguistic_competence/
├── benchmarks/   # Benchmark definitions, evaluation, and result aggregation
├── curriculum/   # Data and scripts for curriculum-training experiments
├── data/         # BAT and BL2MP minimal-pair datasets
├── envs/         # Reproducible evaluation and training environments
├── figures/      # Paper figures and results visualization (change <TODO> to /logs/data-all.json absolute path)
├── logprobs/     # Log-probability-based acceptability evaluation
└── logs/         # Outputs produced by the evaluation pipelines
```

- [`benchmarks/`](benchmarks/benchmarks_guide.md) is the entry point for reproducing the benchmark experiments. It covers general-purpose Basque benchmarks, BAT and BL2MP judgments, reasoning-model evaluation, and result aggregation. Full instructions are in the [benchmark evaluation guide](benchmarks/benchmarks_guide.md).
- [`curriculum/`](curriculum/curriculum_guide.md) contains the corpora, Axolotl configurations, preprocessing workflow, and Slurm scripts used in the curriculum-training experiments. See the [curriculum training guide](curriculum/curriculum_guide.md).
- [`data/`](data/) provides the BAT and BL2MP minimal-pair datasets.
- [`envs/`](envs/envs_guide.md) documents the software and hardware setup needed to reproduce evaluation and training. Versions and installation instructions are collected in the [environment guide](envs/envs_guide.md).
- [`figures/`](figures/) contains the scripts used to generate the paper's figures and results visualizations. Before running them, update the `<TODO>` placeholder path in the scripts to the absolute path of `logs/data-all.json`.
- [`logprobs/`](logprobs/logprobs_guide.md) implements Direct Probability Measurements and Instruction Probability Measurements acceptability scoring from model log probabilities. Configuration, execution, and aggregation are described in the [log-probability guide](logprobs/logprobs_guide.md).
- [`logs/`](logs/) groups the generated outputs.