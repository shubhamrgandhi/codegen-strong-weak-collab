import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import weave

from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages
from agentless_lite.util.loading import *

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import traceback
import random
random.seed(42)

AGENTLESS_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

AGENTLESS_PROMPT_WITH_TRAJECTORY = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

An expert has already worked on this issue. Here are all of its analyses, attempts and solutions:
--- BEGIN EXPERT SOLUTIONS AND ATTEMPTS ---
{trajectory}
--- END EXPERT SOLUTIONS AND ATTEMPTS ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

AGENTLESS_PROMPT_WITH_PLAN = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here is a plan that you can follow to solve this issue:
--- BEGIN PLAN ---
{plan}
--- END PLAN ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


AGENTLESS_PROMPT_WITH_INFO = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here is some high-level information about the repository that you might find useful to solve this issue:
--- BEGIN REPOSITORY INFO ---
{info}
--- END REPOSITORY INFO ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


AGENTLESS_PROMPT_WITH_FEW_SHOT = """
Here are some {similar}example issues from the same repository along with the target file that were changed and final patch generated by an expert{successful}:
--- BEGIN EXAMPLES ---
{few_shot_examples}
--- END EXAMPLES ---

We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

def process_instance(instance, args, file_lock):

    if args.instance_id is not None:
        if args.instance_id != instance['instance_id']:
            return
            
    if args.repo_name is not None:
        if args.repo_name not in instance['instance_id']:
            return

    if args.use_raw_trajectories:
        # Load the trajectory for this instance
        trajectory = load_trajectory(instance['instance_id'], args.raw_trajectories_dir)
        if trajectory:
            repair_prompt = AGENTLESS_PROMPT_WITH_TRAJECTORY
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_plans:
        # Load the trajectory for this instance
        plan = load_plan(instance['instance_id'], args.plans_path, file_lock)
        if plan:
            repair_prompt = AGENTLESS_PROMPT_WITH_PLAN
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_info:
        # Load the trajectory for this instance
        info = load_info(instance['instance_id'], args.info_dir)
        if info:
            repair_prompt = AGENTLESS_PROMPT_WITH_INFO
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_fs:
        # Load few-shot examples for this instance
        few_shot_examples, similar, successful = load_fs(
            instance_id=instance['instance_id'], 
            fs_path=args.fs_path, 
            fs_mode=args.use_fs,  # Mode will be "random_all", "similarity_all", etc
            fs_k=args.fs_k,
            eval_path=args.eval_path,
            tokenizer=args.tokenizer,
            similarity_model=args.similarity_model,
            loc_file=args.loc_file,
        )
        if few_shot_examples:
            repair_prompt = AGENTLESS_PROMPT_WITH_FEW_SHOT
        else:
            repair_prompt = AGENTLESS_PROMPT
    else:
        repair_prompt = AGENTLESS_PROMPT

    formatted_files = ""
    for idx, file in enumerate(instance["found_files"]):
        if idx < args.max_files:
            formatted_file = f'### {file}\n{instance["file_contents"][idx]}\n'
            if args.use_raw_trajectories and trajectory:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    trajectory=trajectory
                )
            elif args.use_plans and plan:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    plan=plan
                )
            elif args.use_info and info:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    info=info
                )
            elif args.use_fs and few_shot_examples:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    few_shot_examples=few_shot_examples,
                    similar=similar,
                    successful=successful,
                )
            else:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                )
            if num_tokens_from_messages(expected_prompt) > args.max_input_tokens:
                print(
                    f"Maximum context length exceeded for instance: {instance['instance_id']} after {idx + 1} files"
                )
                break
            else:
                formatted_files += formatted_file
    if args.use_raw_trajectories and trajectory:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            trajectory=trajectory,
        )
    elif args.use_plans and plan:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            plan=plan,
        )
    elif args.use_info and info:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            info=info,
        )
    elif args.use_fs and few_shot_examples:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            few_shot_examples=few_shot_examples,
            similar=similar,
            successful=successful,
        )
    else:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
        )

    generator = get_generator(args.backend)
    if generator:
        generator.initialize_output_files(args)

    if not generator:
        raise ValueError(f"Unsupported backend: {args.backend}")

    strong_model_attempt = None
    
    # Use the strong model first if enabled
    if args.use_strong_first:
        print(f"Attempting first pass with strong model {args.strong_model} for {instance['instance_id']}")
        # Create a deep copy of args for the strong model
        strong_args = deepcopy(args)
        # Use the strong model
        strong_args.model = args.strong_model
        # Set max_retries to 1 for the strong model
        strong_args.max_retries = 1
        
        # Generate with the strong model
        strong_git_diff, strong_response = generator.generate_with_retries(
            instance,
            prompt,
            strong_args,  # Use the modified copy
            file_lock,
            args.output_file,
            instance.get("image_assets", None),
            return_response=True
        )
        
        # If we have a valid diff from the strong model, return it
        if strong_git_diff:
            print(f"Strong model successfully generated a valid patch for {instance['instance_id']}")
            return
        
        # Save the strong model's attempt for context
        strong_model_attempt = strong_response
        print(f"Falling back to weak model {args.model} for {instance['instance_id']}")
    
    # If we're using the weak model with the strong model's attempt
    if strong_model_attempt:
        # Create a new prompt with the strong model's attempt
        strong_prompt = AGENTLESS_PROMPT_WITH_TRAJECTORY.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            trajectory=f"STRONG MODEL INCORRECT ATTEMPT:\n{strong_model_attempt}"
        )
        # Use the weak model with the strong model's attempt
        git_diff = generator.generate_with_retries(
            instance,
            strong_prompt,
            args,
            file_lock,
            args.output_file,
            instance.get("image_assets", None)
        )
    else:
        # Use the regular prompt
        git_diff = generator.generate_with_retries(
            instance,
            prompt,
            args,
            file_lock,
            args.output_file,
            instance.get("image_assets", None)
        )
    
    if not git_diff:
        print(
            f"Failed to generate valid response for {instance['instance_id']} after {args.max_retries} attempts"
        )
    else:
        print(
            f"Processing completed for {instance['instance_id']}"
        )


def batch(args):
    file_lock = threading.Lock()

    with open(args.loc_file, "r", encoding="utf-8") as infile:
        retrieval_data = [json.loads(line.strip()) for line in infile]

    if args.num_threads == 1:
        for instance in retrieval_data:
            process_instance(instance, args, file_lock)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(process_instance, instance, args, file_lock)
                for instance in retrieval_data
            ]
            for future in futures:
                future.result()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate patches for the retrieved code"
    )
    parser.add_argument(
        "--loc_file",
        type=str,
        default="retrievals.jsonl",
        help="Path to the input JSONL file containing the retrieved files and contents",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="all_preds.jsonl",
        help="Path to save the generated predictions",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="agentless_lite",
        help="Base directory path for the project",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=111000,
        help="Maximum number of tokens allowed in the input prompt",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=4000,
        help="Maximum number of tokens allowed in the completion response (including reasoning tokens)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--instance_id",
        type=str,
    )
    parser.add_argument(
        "--repo_name",
        type=str,
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of concurrent threads for processing",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0,
        help="Initial temperature to start generations at (will increase by 0.1 if no valid response is generated)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=20,
        help="Maximum number of retries to generate a valid response",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=5,
        help="Maximum number of files to include in the context window",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "vllm", "open_router"],
        help="The backend service to use for generation (openai or vllm)",
    )
    parser.add_argument(
        "--tool_use",
        action="store_true",
        help="Have the language model call tools to generate the patch (only supported for some models)",
    )
    parser.add_argument(
        "--use_raw_trajectories",
        action="store_true",
        help="Have a weak language model use the raw trajectories from a previous run by a strong model",
    )
    parser.add_argument(
        "--raw_trajectories_dir",
        type=str,
        default="data/strong_model_trajectories",
        help="Directory for raw trajectories",
    )
    parser.add_argument(
        "--use_plans",
        action="store_true",
        help="Have a weak language model use a plan generated by a strong model",
    )
    parser.add_argument(
        "--plans_path",
        type=str,
        default="results/generate_plan_o3-mini-2025-01-31/all_plans.jsonl",
        help="JSONL file containing plans",
    )
    parser.add_argument(
        "--use_info",
        action="store_true",
        help="Have a weak language model use a high-level repository-specific generated by a strong model",
    )
    parser.add_argument(
        "--info_dir",
        type=str,
        default="data/repo_insights",
        help="Directory for high-level repository-specific information, should contain <repo_name>_insights.json files.",
    )
    parser.add_argument(
        "--use_fs",
        type=str,
        choices=["random_all", "similarity_all", "random_successful", "similarity_successful"],
        help="Mode for selecting few-shot examples (random or similarity-based)"
    )
    parser.add_argument(
        "--fs_path",
        type=str,
        default="results/base_o3-mini-2025-01-31/all_preds.jsonl",
        help="JSONL file containing strong model predictions"
    )
    parser.add_argument(
        "--fs_k",
        type=int,
        default=3,
        help="Number of similar examples to include (per repository)"
    )
    parser.add_argument(
        "--strong_model",
        type=str,
        default="o3-mini-2025-01-31",
        help="Strong model to use for the first generation attempt",
    )
    parser.add_argument(
        "--use_strong_first",
        action="store_true",
        help="Use the strong model for the first attempt, then fall back to the weak model",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="sb-cli-reports/swe-bench_lite__test__agentless_lite_base_o3-mini-2025-01-31.json",
        help="Path to the evaluation results JSON containing resolved_ids"
    )
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--warming", action="store_true")
    parser.add_argument("--enable_weave", action="store_true")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs",
        help="Folder to save the output files",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Use multimodal prompt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.testbed_dir = os.path.join(args.base_path, "testbed")

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "logs"), exist_ok=True)
    args.output_file = os.path.join(
        args.output_folder, os.path.basename(args.output_file)
    )

    log_file = os.path.join(args.output_folder, "logs", "repair_parameters.json")
    with open(log_file, "w", encoding="utf-8") as f:
        args_dict = {
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }
        json.dump(args_dict, f, indent=4)

    if args.enable_weave:
        weave.init(f"agentless_{args.output_folder}")
    args.tokenizer = None
    args.similarity_model = None
    if args.use_fs and 'similarity' in args.use_fs:
        try:
            # Load CodeBERT model and tokenizer
            args.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            args.similarity_model = AutoModel.from_pretrained("microsoft/codebert-base")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            args.similarity_model.to(device)  # Move model to GPU if available
        except:
            print("Couldn't load model microsoft/codebert-base for calculating similarity.")

    batch(args)
    print("Finished Generation for all.")
