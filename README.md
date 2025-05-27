# Strong-Weak Code Collaboration

Code for the paper **[An Empirical Study on Strong-Weak Model Collaboration for Repo-level Code Generation](https://arxiv.org/abs/2505.20182)**

![Strong-Weak Collaboration](assets/taxonomy.png)

## Overview

We study cost-efficient collaboration between strong and weak language models for repository-level code generation, where the weak model handles simpler tasks at lower cost, and the most challenging tasks are delegated to the strong model.
While many works propose architectures for this task, few analyze performance relative to cost. 
We evaluate a broad spectrum of collaboration strategies: context-based, pipeline-based, and dynamic, on GitHub issue resolution.
Our most effective collaborative strategy achieves equivalent performance to the strong model while reducing the cost by 40%.
Based on our findings, we offer actionable guidelines for choosing collaboration strategies under varying budget and performance constraints.
Our results show that strong–weak collaboration substantially boosts the weak model’s performance at a fraction of the cost, pipeline and context-based methods being most efficient. 

## Paper Reference

**Citation:** 
```bibtex
@misc{gandhi2025empiricalstudystrongweakmodel,
      title={An Empirical Study on Strong-Weak Model Collaboration for Repo-level Code Generation}, 
      author={Shubham Gandhi and Atharva Naik and Yiqing Xie and Carolyn Rose},
      year={2025},
      eprint={2505.20182},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.20182}, 
}
```

## Installation

### Create conda environment and install requirements
```bash
conda create -n strong-weak-collab python=3.11
conda activate strong-weak-collab
pip install -r requirements.txt
```

### API Keys
Set the following environment variables:
- For retrieval: `VOYAGE_API_KEY` (or `OPENAI_API_KEY` if using OpenAI embeddings)
- For generation: `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (depending on models used)

### SWE-bench CLI for evaluation
For more details, see: https://github.com/swe-bench/sb-cli

## Quick Start

Run a basic experiment with the recommended model pair (O3-mini + GPT-4o-mini):

```bash
# 1. Run retrieval (only needed once)
bash run_retrieval.sh

# 2. Setup experiment directories
python scripts/prep_dirs.py

# 3. Run a basic experiment
bash run_pipeline_base.sh --model o3-mini-2025-01-31
```

## Repository Structure

```
├── agentless_lite/
│   ├── repair.py                    # Main generation script
│   ├── generate_plan.py             # Plan generation
│   ├── generate_instance_faq.py     # Instance-level FAQ generation
│   ├── repair_self_consistency.py   # Self-consistency methods
│   ├── extract_repo_insights.py     # Repo-level context generation
│   ├── retrieve_swe.py              # Code retrieval
│   └── util/                        # Utility modules
├── scripts/
│   ├── prep_dirs.py                 # Setup experiment directories
│   ├── get_avg_num_attempts.py      # Calculate API call attempts
│   ├── analyze_categories.py        # Performance analysis by categories
│   ├── compute_generation_cost.py   # Cost analysis
│   ├── localization_perf.py         # Localization performance
│   └── best_of_n_resolution_rate.py # Best-of-N analysis
├── run_retrieval.sh                 # Retrieval script
├── run_pipeline_*.sh                # Experiment scripts
└── results/                         # Output directory
```

## Results Reproduction

### 1. Run Retrieval
```bash
bash run_retrieval.sh
```

### 2. Setup Directories
```bash
python scripts/prep_dirs.py
```

### 3. Run Experiments

For repo-level context experiments, generate it using the `agentless_lite/extract_repo_insights.py` script. This is made available in the `data_{strong_model}` directories
Basic command format:
```bash
bash run_pipeline_{exp}.sh --model [MODEL_NAME]
```

Available experiments and their corresponding methods:

| Experiment | Method |
|------------|--------|
| `base_strong` | Base Strong |
| `base_weak` | Base Weak |
| `sc_direct` | Self-Consistency - Direct |
| `sc_clustering` | Self-Consistency - Clustering |
| `sc_universal` | Self-Consistency - Universal |
| `best_of_n` | Best of N |
| `first_strong` | Strong LM Single Attempt |
| `prompt_reduction` | Prompt Reduction |
| `fallback` | Weak LM First |
| `plan` | Plan |
| `instance_faq` | Instance Level QA Pairs |
| `router_weak` | Weak Router |
| `router_strong` | Strong Router |
| `fs_random_successful_1` | 1 Shot Successful - Random |
| `fs_random_successful_5` | 5 Shot Successful - Random |
| `fs_similarity_successful_1` | 1 Shot Successful - Similarity |
| `fs_similarity_successful_5` | 5 Shot Successful - Similarity |
| `repograph` | Repo Structure |
| `repo_faq` | Repo Level QA Pairs |
| `info` | Repo Summary |

Example commands:
```bash
# Run baseline experiments
bash run_pipeline_base.sh --model o3-mini-2025-01-31
bash run_pipeline_base.sh --model gpt-4o-mini-2024-07-18

# Run collaboration methods
bash run_pipeline_plan.sh --model gpt-4o-mini-2024-07-18
bash run_pipeline_first_strong.sh --model gpt-4o-mini-2024-07-18
bash run_pipeline_router_weak.sh --model gpt-4o-mini-2024-07-18
```

### 4. Evaluation

After running experiments, evaluate using SWE-bench CLI:
```bash
sb-cli submit --predictions_path results/{exp}_{model}/all_preds.jsonl --run_id agentless_lite_{exp}_{model} swe-bench_lite test
```

The evaluation command is automatically printed after each experiment completes.

## Analysis Scripts

### Cost and Performance Analysis
```bash
# Calculate average number of API attempts
python scripts/get_avg_num_attempts.py --jsonl_path results/{exp}_{model}/all_preds.jsonl

# Compute generation costs from logs
python scripts/compute_generation_cost.py --results_dir results/{exp}_{model}

# Analyze performance by issue categories
python scripts/analyze_categories.py --strong_model o3-mini-2025-01-31 --weak_model gpt-4o-mini-2024-07-18

# Calculate best-of-N resolution rates
python scripts/best_of_n_resolution_rate.py
```

### Plotting and factor analysis

All plotting-related scripts are available in the `plotting_scripts` directory
Scripts for compiling factor analysis data are available in `factor_analysis`

## Expected Results

After running experiments, you should see:
- `results/{exp}_{model}/all_preds.jsonl` - Generated patches
- `results/{exp}_{model}/logs/` - Detailed logs and cost information
- `sb-cli-reports/` - Evaluation results from SWE-bench CLI
- Resolution rates approximately matching the paper's reported performance (some non-determinism due to temperature)

## Models

Common model pairs used in the paper:
- `o3-mini-2025-01-31` + `gpt-4o-mini-2024-07-18`
- `o3-mini-2025-01-31` + `qwen2.5-coder-xb-instruct` where x ~ {7, 14, 32}
- `o4-mini-2025-04-16` + `gpt-4o-mini-2024-07-18`
- `gpt-4o-mini-2024-07-18` + `qwen2.5-coder-7b-instruct`
