# 🐱 Agentless Lite

<p align="center">
    <a href="https://arxiv.org/abs/2407.01489"><img src="https://img.shields.io/badge/📑-Arxiv-b31b1b?style=for-the-badge"></a>
    <a href="https://github.com/OpenAutoCoder/Agentless/blob/master/LICENSE"><img src="https://forthebadge.com/images/badges/license-mit.svg" style="height: 28px"></a>
</p>

<p align="center">
    <big><a href="#-news">📢News</a></big> |
    <big><a href="#-about">💡About</a></big> |
    <big><a href="#-setup">🐈Setup</a></big> |
    <big><a href="#-quickstart">⚡Quickstart</a></big> |
    <big><a href="#-localization">🐈Localization</a></big> |
    <big><a href="#-repair">🧶Repair</a></big> |
    <big><a href="#-comparison">🧶Comparison</a></big> |
    <big><a href="#-artifacts">🐈‍⬛Artifacts</a></big> |
    <big><a href="#-acknowledgement">😻Acknowledgement</a></big>
</p>

## 📢 News

- *Febuary 13th, 2025*: We just released **Agentless-Lite** 1.0! **Agentless-Lite** is the top-performing RAG-only scaffold for SWE-bench with a 

## 💡 About

<p align="left">
    <big>Check out the original Agentless implementation here: <a href="https://github.com/OpenAutoCoder/Agentless">🚀 Agentless Repository</a></big>
</p>

*Agentless Lite* is a generalized, lightweight adaptation of the [Agentless](https://github.com/OpenAutoCoder/Agentless) framework for solving software development issues. Specifically, *Agentless Lite* performs the following steps:

1. Use an embedding model to retrieve relevant files from the repository
2. Query the LLM to generate a repair based on the top 5 retrieved files, retrying the generation until the model outputs a valid patch.

Thats it! While simple this approach is competitive with SOTA agents and comes with several key advantages:

- 🔍 Exclusively RAG-based localization
- 💨 No required runtime environment
- 🐍 No python specific language dependencies
- ⚡ Simple, single-prompt inference
- 🤝 Support for over 300 models through *OpenRouter*

## 🐈 Setup

First create the environment

```shell
git clone https://github.com/sorendunn/Agentless-Lite.git
cd Agentless-Lite

conda create -n agentless_lite python=3.11
conda activate agentless_lite
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then set up your OpenAI API key, VOYAGE_API_KEY (if using Voyage embeddings), and WANDB_API_KEY (if using weave):

```shell
export OPENAI_API_KEY={openai_key_here}
export VOYAGE_API_KEY={vogage_key_here}
export WANDB_API_KEY={wandb_key_here}
```

## ⚡ Quickstart

### Prerequisites

1. Download and unzip the prepared retrieval contexts for SWE-Bench Lite [swe_bench_lite.zip](LINK_HERE) or [swe_bench_verified.zip](LINK_HERE)
    - Alternatively, see `Localization` section for how to generate your own retrieval contexts
2. Move the jsonl file to the main Agentless Lite directory (or specify the path with `--loc_file`)

### Run

```shell
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results \
        --loc_file results/retrieval.jsonl \
        --temp 0 \
        --model gpt-4o-mini \
        --max_completion_tokens 78000 \
        --max_input_tokens 118000 \
        --backend openai \
        --num_threads 8 \
        --max_retries 10 \
        --max_files 5
```

This command will iteratively prompt the model until a valid patch is produced or the `--max_retries` is reached. It will produce `all_preds.jsonl` that contains the generated patch for each instance_id which you can then directly evaluate with your favorite SWE-bench evaluation method!

## 🐈 Localization

> [!TIP]
>
> To quickly start evaluating on SWE-bench see the `Quickstart` section above

Create the embeddings and perform retrieval:

```shell
python agentless_lite/retrieve_swe.py \
        --dataset princeton-nlp/SWE-bench_Lite \
        --num_threads 16 \
        --output_folder results \
        --output_file retrieval.jsonl \
        --embedding_folder voyage_lite \
        --embedding_model voyage-code-3 \
        --filter_model text-embedding-3-small \
        --filter_python \
        --entire_file
```

This will split files in the repositories into small chunks for embedding. `--filter` specifies to only embed the non-test python files in the repository. `--entire-file` specifies to retrieve the entire file if any chunks within the file are retrieved. `--retrieve_num` indicates the number of chunks to retrieve.

> [!TIP]
>
> We currently support OpenAI and Voyage embeddings, you can use `--embedding-model` to select the desired embedding model (by default it will use Voyage embeddings)
>
> For example `--embedding-model=openai_small`

> [!TIP]
>
> We use multiple threads (controllable via `--num-threads`) to speed up the Agentless process

## 🧶 Repair

Using the retrieved contexts from the previous step, we now perform repair.

Unlike **Agentless**, **Agentless Lite** generates samples one at a time until a sample successfully passes syntax checks. Instead of performing subsequent selection, this patch is simply taken as the final patch.

Run the following command to generate the patches:

```shell
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results \
        --loc_file results/retrieval.jsonl \
        --temp 0 \
        --model o3-mini \
        --max_completion_tokens 78000 \
        --max_input_tokens 118000 \
        --backend openai \
        --num_threads 8 \
        --max_retries 10 \
        --max_files 5
```

> [!TIP]
>
> We currently support OpenRouter, OpenAI, and DeepSeek models. Additionally we support batch submission for compatible OpenAI models. You can change which of these backends to use via the `--backend` parameter (open_router, openai, openai_batch_offline or deepseek)
>
> For example `--backend deepseek`

This commands generates up to 10 samples as defined `--max_retries 10`. The patches are saved in `results/all_preds.jsonl`. The complete logs are also saved in `results/repair/logs`

## 🧶 Comparison

Below shows the comparison graph between **Agentless Lite**, **Agentless Lite Mini**, and the full **Agentless**.

<p align="center">
<img src="./resources/comparison_graph.png" style="width:75%; margin-left: auto; margin-right: auto;">
</p>

## 🐈‍⬛ Artifacts

You can download the complete artifacts of **Agentless Lite** in our [v0.1.0 release](https://github.com/sorendunn/Agentless-Lite/releases/tag/v0.1.0):

- 🐈‍⬛ agentless_swebench_lite: complete Agentless Lite run on SWE-bench Lite for r1
- 🐈‍⬛ swebench_repo_lite_voyage: top 100 retreived files for Voyage-Code-3 on SWE-bench Lite
- 🐈‍⬛ swebench_repo_lite_openai: top 100 retreived files for OpenAI-Embedding-3-Small on SWE-bench Lite

## 😻 Acknowledgement

* [Agentless](https://github.com/OpenAutoCoder/Agentless)
* [SWE-bench](https://www.swebench.com/)
* [Aider](https://github.com/paul-gauthier/aider)
* [SWE-bench-docker](https://github.com/aorwall/SWE-bench-docker)
