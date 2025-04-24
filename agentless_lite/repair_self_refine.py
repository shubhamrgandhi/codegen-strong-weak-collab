import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import weave

from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages
from agentless_lite.util.methods import *
from agentless_lite.util.prompts import *

import traceback
import random
random.seed(42)


def format_previous_attempts(attempts):
    """Format previous attempts for inclusion in the prompt"""
    if not attempts:
        return ""
    
    formatted = "Here are my previous attempts to solve this issue:\n"
    for i, attempt in enumerate(attempts):
        formatted += f"--- ATTEMPT {i+1} ---\n{attempt}\n\n"
    
    return formatted

def extract_git_diff(response):
    """Extract git diff from a response"""
    git_diff = None
    # Look for ```python blocks that contain SEARCH/REPLACE patterns
    python_blocks = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
    for block in python_blocks:
        if '<<<<<<< SEARCH' in block and '>>>>>>> REPLACE' in block:
            if git_diff is None:
                git_diff = block
            else:
                git_diff += "\n\n" + block
    return git_diff

def process_instance(instance, args, file_lock):
    """Process a single SWE-bench instance using self-refinement"""
    if args.instance_id is not None:
        if args.instance_id != instance['instance_id']:
            return
            
    if args.repo_name is not None:
        if args.repo_name not in instance['instance_id']:
            return

    # Format the retrieved files
    formatted_files = ""
    for idx, file in enumerate(instance["found_files"]):
        if idx < args.max_files:
            formatted_file = f'### {file}\n{instance["file_contents"][idx]}\n'
            expected_prompt = SELF_REFINE_PROMPT.format(
                problem_statement=instance["problem_description"],
                retrieval=formatted_files + formatted_file,
                previous_attempts=""
            )
            if num_tokens_from_messages(expected_prompt) > args.max_input_tokens:
                print(
                    f"Maximum context length exceeded for instance: {instance['instance_id']} after {idx + 1} files"
                )
                break
            else:
                formatted_files += formatted_file
    
    # Get the appropriate generator
    generator = get_generator(args.backend)
    if generator:
        generator.initialize_output_files(args)

    if not generator:
        raise ValueError(f"Unsupported backend: {args.backend}")
    
    # Initialize tracking variables
    previous_attempts = []
    final_git_diff = None
    
    # Log the start of processing
    print(f"Starting self-refinement for {instance['instance_id']}")
    
    # Store all attempts for logging
    all_responses = []
    
    # Create log directory for this instance
    instance_log_dir = os.path.join(args.output_folder, "logs", instance['instance_id'])
    os.makedirs(instance_log_dir, exist_ok=True)
    
    # Outer loop for self-refinement
    for refine_iter in range(args.refine_iterations):
        print(f"Starting refinement iteration {refine_iter+1}/{args.refine_iterations} for {instance['instance_id']}")
        
        # Format previous attempts for the prompt
        previous_attempts_text = format_previous_attempts(previous_attempts)
        
        # Create the prompt with previous attempts
        prompt = SELF_REFINE_PROMPT.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            previous_attempts=previous_attempts_text
        )
        
        # Save the prompt to log
        with open(os.path.join(instance_log_dir, f"prompt_refine_{refine_iter+1}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # Track attempts for this refinement iteration
        refinement_responses = []
        valid_diff_found = False
        
        # Inner loop for generation attempts - we'll handle this ourselves instead of relying on generate_with_retries
        temperature = args.temp
        for attempt in range(args.generation_attempts_per_refinement):
            try:
                print(f"  Attempt {attempt+1}/{args.generation_attempts_per_refinement} for refinement {refine_iter+1}")

                if not os.path.exists(f"{args.output_folder}/temp_{refine_iter}_{attempt}.jsonl"):
                    with open(f"{args.output_folder}/temp_{refine_iter}_{attempt}.jsonl", "w", encoding="utf-8") as outfile:
                        pass

                response = generator.generate(
                    instance, 
                    prompt,
                    args,
                    file_lock,
                    f"{args.output_folder}/temp_{refine_iter}_{attempt}.jsonl"
                )
                
                # Save the response to log
                with open(os.path.join(instance_log_dir, f"response_refine_{refine_iter+1}_attempt_{attempt+1}.txt"), 
                         "w", encoding="utf-8") as f:
                    f.write(f"Temperature: {temperature}\n\n")
                    f.write(response)
                
                # Add to refinement responses
                refinement_responses.append({
                    "attempt": attempt + 1,
                    "temperature": temperature,
                    "response": response
                })
                
                # Extract git diff from response
                git_diff = extract_git_diff(response)
                
                # If valid git_diff found, save it and break inner loop
                if git_diff:
                    print(f"  Valid patch found in attempt {attempt+1} of refinement {refine_iter+1}")
                    valid_diff_found = True
                    previous_attempts.append(response)
                    final_git_diff = git_diff
                    break
                
                # Increase temperature for next attempt
                temperature += 0.1
                
                # Add to previous attempts regardless of validity
                previous_attempts.append(response)
                
            except Exception as e:
                print(f"  Error in attempt {attempt+1} of refinement {refine_iter+1}: {str(e)}")
                with open(os.path.join(instance_log_dir, f"error_refine_{refine_iter+1}_attempt_{attempt+1}.txt"), 
                         "w", encoding="utf-8") as f:
                    f.write(f"Temperature: {temperature}\n\n")
                    f.write(traceback.format_exc())
                temperature += 0.1
        
        # Add all attempts from this refinement to our log
        all_responses.append({
            "refinement_iteration": refine_iter + 1,
            "attempts": refinement_responses,
            "valid_patch_found": valid_diff_found
        })
        
        # If we found a valid patch, break the outer loop
        if valid_diff_found:
            break
    
    # After all refinement iterations, save the final result
    if final_git_diff:
        print(f"Processing completed for {instance['instance_id']} with successful refinement")
        
        # Save to the output file
        with file_lock:
            with open(args.output_file, "a", encoding="utf-8") as outfile:
                result = {
                    "instance_id": instance["instance_id"],
                    "model_name_or_path": "agentless_lite",
                    "model_patch": final_git_diff,
                    "successful_refinement_iteration": next(
                        (i+1 for i, r in enumerate(all_responses) if r["valid_patch_found"]), None
                    ),
                    "total_attempts": sum(len(r["attempts"]) for r in all_responses),
                    "success": True
                }
                outfile.write(json.dumps(result) + "\n")
        
        # Save detailed log for analysis
        log_file = os.path.join(args.output_folder, "logs", f"{instance['instance_id']}_summary.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "instance_id": instance["instance_id"],
                "success": True,
                "final_git_diff": final_git_diff,
                "successful_refinement_iteration": next(
                    (i+1 for i, r in enumerate(all_responses) if r["valid_patch_found"]), None
                ),
                "total_attempts": sum(len(r["attempts"]) for r in all_responses),
                "all_responses": all_responses
            }, f, indent=2)
    else:
        print(f"Failed to generate valid response for {instance['instance_id']} after all refinement iterations")
        
        # # Save failure information
        # with file_lock:
        #     with open(args.output_file, "a", encoding="utf-8") as outfile:
        #         result = {
        #             "instance_id": instance["instance_id"],
        #             "git_diff": None,
        #             "successful_refinement_iteration": None,
        #             "total_attempts": sum(len(r["attempts"]) for r in all_responses),
        #             "success": False
        #         }
        #         outfile.write(json.dumps(result) + "\n")
        
        # Save detailed log for analysis
        log_file = os.path.join(args.output_folder, "logs", f"{instance['instance_id']}_summary.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "instance_id": instance["instance_id"],
                "success": False,
                "final_git_diff": None,
                "total_attempts": sum(len(r["attempts"]) for r in all_responses),
                "all_responses": all_responses
            }, f, indent=2)


def batch(args):
    """Process all instances in the retrieval data"""
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate patches using self-refinement with weak models"
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
        help="The weak model to use for generation",
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        help="Specific instance ID to process",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        help="Specific repository name to filter for",
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
        "--refine_iterations",
        type=int,
        default=2,
        help="Number of self-refinement iterations",
    )
    parser.add_argument(
        "--generation_attempts_per_refinement",
        type=int,
        default=5,
        help="Maximum number of generation attempts per refinement iteration",
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
        choices=["openai", "vllm", "open_router", "deepseek", "openai_batch_offline"],
        help="The backend service to use for generation",
    )
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--warming", action="store_true")
    parser.add_argument("--enable_weave", action="store_true")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs_self_refine",
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

    # Setup output directories
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "logs"), exist_ok=True)
    args.output_file = os.path.join(
        args.output_folder, os.path.basename(args.output_file)
    )

    # Save configuration
    log_file = os.path.join(args.output_folder, "logs", "self_refine_parameters.json")
    with open(log_file, "w", encoding="utf-8") as f:
        args_dict = {
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }
        json.dump(args_dict, f, indent=4)

    # Initialize weave if enabled
    if args.enable_weave:
        weave.init(f"agentless_self_refine_{args.output_folder}")

    # Process all instances
    batch(args)
    print("Finished Self-Refinement Generation for all.")