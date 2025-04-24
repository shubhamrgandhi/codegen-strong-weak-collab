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

Please create a detailed FAQ (Frequently Asked Questions) document that would help a junior developer understand and fix this specific issue. Do NOT generate any code patches or specific edits.

Your FAQ should include 7-10 questions and answers about:

1. Issue Understanding:
   - What is the exact problem described in the issue?
   - What are the expected vs. actual behaviors?
   - What conditions trigger this issue?

2. Codebase Navigation:
   - Which specific files and functions are most relevant to this issue?
   - What are the key components involved in this functionality?
   - How do these components interact?

3. Technical Analysis:
   - What are the potential root causes of this issue?
   - What code patterns or anti-patterns might be contributing to the bug?
   - What specific edge cases might not be handled correctly?

4. Implementation Guidance:
   - What approaches could be used to fix this issue?
   - What implementation pitfalls should be avoided?
   - How should the solution be tested?

5. Codebase Specifics:
   - What patterns or conventions does this codebase use that are relevant to the fix?
   - What existing helper functions or utilities could be leveraged?
   - What dependencies or side effects need to be considered?

Make your questions and answers detailed, specific to this issue, and include concrete references to the code when possible. Avoid generic programming advice - focus on information that directly helps solve this specific issue.
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
        field_name="instance_faq",
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
        description="Generate instance level FAQs for solving issue for the retrieved code"
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
        default="all_instance_faqs.jsonl",
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
