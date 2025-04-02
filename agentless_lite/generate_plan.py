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

Please analyze the issue and provide a detailed plan to fix it. Do NOT generate any code patches or specific edits.

Your plan should include:

1. Bug localization: Identify which file(s) contain the bug based on the issue statement
2. Root cause analysis: Explain why the bug is occurring
3. Solution approach: Describe conceptually how to fix the issue
4. Implementation strategy: Outline the logical steps needed to implement the solution

Keep your analysis focused on the problem-solving approach rather than specific code changes.
"""

def process_instance(instance, args, file_lock):

    if args.instance_id is not None:
        if args.instance_id != instance['instance_id']:
            return
            
    if args.repo_name is not None:
        if args.repo_name not in instance['instance_id']:
            return

    repair_prompt = AGENTLESS_PROMPT

    formatted_files = ""
    for idx, file in enumerate(instance["found_files"]):
        if idx < args.max_files:
            formatted_file = f'### {file}\n{instance["file_contents"][idx]}\n'
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
    prompt = repair_prompt.format(
        problem_statement=instance["problem_description"],
        retrieval=formatted_files,
    )

    generator = get_generator(args.backend)
    if generator:
        generator.initialize_output_files(args)

    if not generator:
        raise ValueError(f"Unsupported backend: {args.backend}")

    generator.generate_plan(
        instance,
        prompt,
        args,
        file_lock,
        args.output_file,
        instance.get("image_assets", None),
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
        default="all_plans.jsonl",
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
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--warming", action="store_true")
    parser.add_argument("--enable_weave", action="store_true")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs",
        help="Folder to save the output files",
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
