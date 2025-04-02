import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import weave

from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages

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


AGENTLESS_PROMPT_MULTIMODAL = """
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
Wrap the *SEARCH/REPLACE* edit in blocks (e.g.) ```python...```.
"""

AGENTLESS_PROMPT_TOOL_USE = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate edits to fix the issue.
"""

def load_trajectory(instance_id, trajectory_dir):
    """
    Load all trajectory entries from JSON file if it exists.
    Concatenates all content fields from all entries.
    """
    trajectory_path = os.path.join(trajectory_dir, f"{instance_id}.json")
    
    if not os.path.exists(trajectory_path):
        print(f"No trajectory file found at {trajectory_path}")
        return None
    
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        # Ensure we have a list of responses
        if not isinstance(trajectory_data, list) or not trajectory_data:
            print(f"Unexpected trajectory format in {trajectory_path}")
            return None
        
        # Extract and format all content fields
        all_content = []
        for i, entry in enumerate(trajectory_data):
            content = entry.get("content", "")
            if content:
                header = f"ATTEMPT {i+1} OF {len(trajectory_data)}"
                all_content.append(f"{'='*20} {header} {'='*20}\n\n{content}")
        
        # Join all content with separators
        full_trajectory = "\n\n" + "\n\n".join(all_content)
        
        return full_trajectory
    
    except Exception as e:
        print(f"Error loading trajectory for {instance_id}: {e}")
        return None


def load_plan(instance_id, plans_path, file_lock=None):
    """
    Load a plan from a JSONL file if it exists.
    Each line in the JSONL file should contain 'instance_id' and 'plan' fields.
    Thread-safe if a file_lock is provided.
    
    Args:
        instance_id (str): The ID of the instance to find a plan for
        plans_path (str): Path to the JSONL file containing plans
        file_lock (threading.Lock, optional): Lock for thread-safe file access
    
    Returns:
        str or None: The plan for the specified instance_id if found, None otherwise
    """
    if not os.path.exists(plans_path):
        print(f"No plans file found at {plans_path}")
        return None
    
    try:
        # Use a context manager for the lock if provided
        with file_lock if file_lock else nullcontext():
            with open(plans_path, 'r', encoding='utf-8') as f:
                for line in f:
                    plan_data = json.loads(line.strip())
                    if plan_data.get('instance_id') == instance_id and 'plan' in plan_data:
                        return plan_data['plan']
            
            print(f"No plan found for instance ID: {instance_id}")
            return None
    
    except Exception as e:
        print(f"Error loading plan for {instance_id}: {e}")
        return None


def load_info(instance_id, info_dir):
    """
    Load repository insights for the given instance ID.
    
    Args:
        instance_id (str): The instance ID in the format {something}__{repo_name}-{issue_id}
        info_dir (str): Directory containing repository insight JSON files
    
    Returns:
        str or None: The repository insights if found, None otherwise
    """
    try:
        # Extract repo name from instance_id
        # Format is typically {something}__{repo_name}-{issue_id}
        parts = instance_id.split('__')
        if len(parts) < 2:
            print(f"Invalid instance ID format: {instance_id}")
            return None
            
        # Extract repo_name from the second part (before the hyphen)
        repo_identifier = parts[1]
        repo_name = repo_identifier.split('-')[0]
        
        # Construct path to the insights file
        insights_path = os.path.join(info_dir, f"{repo_name}_insights.json")
        
        if not os.path.exists(insights_path):
            print(f"No insights file found at {insights_path}")
            return None
        
        # Load the insights file
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights_data = json.load(f)
        
        # Extract the 'insights' field which contains high-level repository information
        if 'insights' in insights_data:
            return insights_data['insights']
        else:
            print(f"No 'insights' field found in {insights_path}")
            return None
            
    except Exception as e:
        print(f"Error loading insights for {instance_id}: {e}")
        return None

        

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
    elif args.tool_use:
        repair_prompt = AGENTLESS_PROMPT_TOOL_USE
    elif args.multimodal:
        repair_prompt = AGENTLESS_PROMPT_MULTIMODAL
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

    git_diff = generator.generate_with_retries(
        instance,
        prompt,
        args,
        file_lock,
        args.output_file,
        instance.get("image_assets", None),
    )
    if not git_diff:
        print(
            f"Failed to generate valid response for {instance['instance_id']} after {args.max_retries} attempts"
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
        choices=["openai", "anthropic", "deepseek", "open_router"],
        help="The backend service to use for generation (openai, anthropic, deepseek, or open_router)",
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

    batch(args)
    print("Finished Generation for all.")
