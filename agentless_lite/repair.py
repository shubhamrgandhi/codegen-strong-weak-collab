import argparse
import json
import os
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import weave

from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages
from agentless_lite.util.methods import *
from agentless_lite.util.prompts import *

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

import traceback
import random
random.seed(42)


def process_instance(instance, args, file_lock):

    if args.instance_id is not None:
        if args.instance_id != instance['instance_id']:
            return
            
    if args.repo_name is not None:
        if args.repo_name not in instance['instance_id']:
            return

    # RepoGraph handling
    if args.repo_graph:
        issue_name = instance['instance_id']
        if issue_name in args.repo_graph_prompts:
            prompt = args.repo_graph_prompts[issue_name]
            if prompt == "skipped since no files were localized":
                print(f"Skipping {issue_name}: no files were localized")
                return
            
            # Get the generator and initialize output files
            generator = get_generator(args.backend)
            if generator:
                generator.initialize_output_files(args)
            
            if not generator:
                raise ValueError(f"Unsupported backend: {args.backend}")
                
            # Use the prompt directly from the pickle file
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
            
            return
        else:
            print(f"Warning: {issue_name} not found in repo_graph prompts")

    if args.use_raw_trajectories:
        # Load the trajectory for this instance
        trajectory = load_trajectory(instance['instance_id'], args.raw_trajectories_dir)
        if trajectory:
            repair_prompt = AGENTLESS_PROMPT_WITH_TRAJECTORY
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_plans:
        # Load the trajectory for this instance
        plan = load_plan_or_instance_faq(instance['instance_id'], args.plans_path, args, file_lock)
        if plan:
            repair_prompt = AGENTLESS_PROMPT_WITH_PLAN
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_instance_faq:
        # Load the trajectory for this instance
        instance_faq = load_plan_or_instance_faq(instance['instance_id'], args.instance_faq_path, args, file_lock)
        if instance_faq:
            repair_prompt = AGENTLESS_PROMPT_WITH_INSTANCE_FAQ
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_info:
        # Load the trajectory for this instance
        info = load_info_or_repo_faq(instance['instance_id'], args.info_dir, args)
        if info:
            repair_prompt = AGENTLESS_PROMPT_WITH_INFO
        else:
            repair_prompt = AGENTLESS_PROMPT
    elif args.use_repo_faq:
        # Load the trajectory for this instance
        repo_faq = load_info_or_repo_faq(instance['instance_id'], args.repo_faq_dir, args)
        if repo_faq:
            repair_prompt = AGENTLESS_PROMPT_WITH_REPO_FAQ
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
            elif args.use_instance_faq and instance_faq:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    instance_faq=instance_faq
                )
            elif args.use_info and info:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    info=info
                )
            elif args.use_repo_faq and repo_faq:
                expected_prompt = repair_prompt.format(
                    problem_statement=instance["problem_description"],
                    retrieval=formatted_files + formatted_file,
                    repo_faq=repo_faq
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
    
    generator = get_generator(args.backend)
    if generator:
        generator.initialize_output_files(args)

    if not generator:
        raise ValueError(f"Unsupported backend: {args.backend}")
    
    # If prompt reduction is enabled, use weak model to identify relevant code sections
    if args.use_prompt_reduction:
        print(f"Applying prompt reduction for {instance['instance_id']}")
        formatted_files = reduce_prompt_context(generator, instance, formatted_files, args, file_lock)
    
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
    elif args.use_instance_faq and instance_faq:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            instance_faq=instance_faq,
        )
    elif args.use_info and info:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            info=info,
        )
    elif args.use_repo_faq and repo_faq:
        prompt = repair_prompt.format(
            problem_statement=instance["problem_description"],
            retrieval=formatted_files,
            repo_faq=repo_faq,
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

    if args.use_router:
        use_strong_model = route_instance(generator, instance, formatted_files, args, file_lock)
        
        if use_strong_model:
            print(f"Using strong model for {instance['instance_id']} based on router decision")
            # Create a deep copy of args for the strong model
            strong_args = deepcopy(args)
            # Use the strong model
            strong_args.model = args.strong_model
            
            # Generate with the strong model (keep the original max_retries)
            git_diff = generator.generate_with_retries(
                instance,
                prompt,
                strong_args,
                file_lock,
                args.output_file,
                instance.get("image_assets", None)
            )
            
            if not git_diff:
                print(f"Strong model failed to generate valid response for {instance['instance_id']} after {strong_args.max_retries} attempts")
            else:
                print(f"Strong model successfully generated a valid patch for {instance['instance_id']}")
            
            # Return immediately after using the strong model, regardless of result
            return

    # Fallback setting implementation
    if args.use_fallback:
        print(f"Using fallback setting for {instance['instance_id']}: first weak model with 5 iterations, then strong model if needed")
        
        # Create a deep copy of args for the weak model
        weak_args = deepcopy(args)
        # Keep the default model (weak model)
        # Set max_retries to 5 for the weak model
        weak_args.max_retries = 5
        
        # Try with weak model first
        print(f"Attempting with weak model {args.model} for {instance['instance_id']} (max 5 iterations)")
        git_diff = generator.generate_with_retries(
            instance,
            prompt,
            weak_args,
            file_lock,
            args.output_file,
            instance.get("image_assets", None)
        )
        
        # If weak model succeeded, return
        if git_diff:
            print(f"Weak model successfully generated a valid patch for {instance['instance_id']}")
            return
        
        # Otherwise, fall back to strong model
        print(f"Weak model failed after 5 iterations. Falling back to strong model {args.strong_model} for {instance['instance_id']}")
        
        # Create a deep copy of args for the strong model
        strong_args = deepcopy(args)
        # Use the strong model
        strong_args.model = args.strong_model
        # Set max_retries to 1 for the strong model
        strong_args.max_retries = 1
        
        # Generate with the strong model
        git_diff = generator.generate_with_retries(
            instance,
            prompt,
            strong_args,
            file_lock,
            args.output_file,
            instance.get("image_assets", None)
        )
        
        if not git_diff:
            print(f"Both weak and strong models failed for {instance['instance_id']}")
        else:
            print(f"Strong model successfully generated a valid patch for {instance['instance_id']}")
        
        return

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
        "--use_instance_faq",
        action="store_true",
        help="Have a weak language model use instance FAQs generated by a strong model",
    )
    parser.add_argument(
        "--instance_faq_path",
        type=str,
        default="results/generate_instance_faq_o3-mini-2025-01-31/all_instance_faqs.jsonl",
        help="JSONL file containing instance FAQs",
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
        "--use_repo_faq",
        action="store_true",
        help="Have a weak language model use repository-specific FAQ generated by a strong model",
    )
    parser.add_argument(
        "--repo_faq_dir",
        type=str,
        default="data/repo_faqs",
        help="Directory for repository-specific FAQ, should contain <repo_name>_repo_faq.json files.",
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
        "--use_fallback",
        action="store_true",
        help="Use the weak model first (5 iterations), then fall back to the strong model if needed (1 iteration)",
    )
    parser.add_argument(
        "--use_prompt_reduction",
        action="store_true",
        help="Use weak model to identify relevant code sections to reduce context for strong model",
    )
    parser.add_argument(
        "--weak_model",
        type=str,
        help="Weak model to use for prompt reduction (if different from main model)",
    )
    parser.add_argument(
        "--use_router",
        action="store_true",
        help="Use a router to determine whether to use the strong or weak model for each instance",
    )
    parser.add_argument(
        "--router_model",
        type=str,
        default=None,
        help="Model to use as router (defaults to the main model if not specified)",
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
    parser.add_argument(
        "--repo_graph",
        action="store_true",
        help="Use repograph prompts from a pickle file",
    )
    parser.add_argument(
        "--repo_graph_path",
        type=str,
        default="data/issue2repograph_prompt.pkl",
        help="Path to the pickle file containing repograph prompts",
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

    # Load repograph prompts if enabled
    if args.repo_graph:
        try:
            with open(args.repo_graph_path, 'rb') as f:
                args.repo_graph_prompts = pickle.load(f)
            print(f"Loaded {len(args.repo_graph_prompts)} repograph prompts from {args.repo_graph_path}")
        except Exception as e:
            print(f"Error loading repograph prompts: {e}")
            args.repo_graph_prompts = {}

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