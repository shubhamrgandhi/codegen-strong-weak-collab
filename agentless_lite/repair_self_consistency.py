import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import difflib
from collections import Counter

from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages
from agentless_lite.util.methods import *
from agentless_lite.util.prompts import *


def load_existing_patches(instance, args):
    """Load previously generated patches from specified directory."""
    if not args.reuse_patches_dir:
        return None, None
    
    # Construct path to the debug directory for this instance
    debug_dir = os.path.join(args.reuse_patches_dir, "debug", instance["instance_id"])
    
    if not os.path.exists(debug_dir):
        print(f"No existing patches found for {instance['instance_id']} in {args.reuse_patches_dir}")
        return None, None
    
    patches = []
    responses = []
    
    # Load patches and responses
    patch_files = sorted([f for f in os.listdir(debug_dir) if f.startswith("patch_") and f.endswith(".diff")])
    response_files = sorted([f for f in os.listdir(debug_dir) if f.startswith("response_") and f.endswith(".txt")])
    
    if not patch_files:
        print(f"No patch files found for {instance['instance_id']} in {debug_dir}")
        return [], []
    
    # Load patches
    for patch_file in patch_files[:args.num_samples]:
        with open(os.path.join(debug_dir, patch_file), "r", encoding="utf-8") as f:
            patch = f.read()
            if patch:  # Only add non-empty patches
                patches.append(patch)
    
    # Load responses if available
    for response_file in response_files[:args.num_samples]:
        with open(os.path.join(debug_dir, response_file), "r", encoding="utf-8") as f:
            response = f.read()
            responses.append(response)
    
    print(f"Loaded {len(patches)} existing patches for {instance['instance_id']}")
    return patches, responses

def generate_diverse_patches(instance, args, generator, prompt, file_lock):
    """Generate multiple diverse patches from a weak model by varying temperature."""
    # First try to load existing patches if reuse is enabled
    if args.reuse_patches:
        patches, responses = load_existing_patches(instance, args)
        if patches is not None and responses is not None:
            return patches[:args.num_samples], responses[:args.num_samples]
    
    patches = []
    responses = []
    
    # Create a deep copy of args for modification
    diverse_args = deepcopy(args)
    
    for idx in range(args.num_samples):
        print(f"Generating patch {idx+1}/{args.num_samples} for {instance['instance_id']}")
        
        if not os.path.exists(f"{args.output_folder}/temp_{idx}.jsonl"):
            with open(f"{args.output_folder}/temp_{idx}.jsonl", "w", encoding="utf-8") as outfile:
                pass

        # Generate the patch
        git_diff, response = generator.generate_with_retries(
            instance,
            prompt,
            diverse_args,
            file_lock,
            f"{args.output_folder}/temp_{idx}.jsonl",  # Temporary output file
            instance.get("image_assets", None),
            return_response=True
        )
        
        if git_diff:
            patches.append(git_diff)
            responses.append(response)
    
    return patches, responses

def direct_consistency(patches):
    """Find the most consistent patch (most frequently occurring)."""
    patch_counter = Counter(patches)
    if not patch_counter:
        return None
    
    most_common_patch, count = patch_counter.most_common(1)[0]
    return most_common_patch

def similarity_clustering(patches):
    """Cluster similar patches and select the largest cluster."""
    if not patches:
        return None
    
    # Simple implementation: use difflib to compare patches
    similarity_matrix = []
    for i, patch1 in enumerate(patches):
        row = []
        for j, patch2 in enumerate(patches):
            if i == j:
                similarity = 1.0
            else:
                # Calculate similarity ratio between patches
                similarity = difflib.SequenceMatcher(None, patch1, patch2).ratio()
            row.append(similarity)
        similarity_matrix.append(row)
    
    # Identify clusters (very basic implementation)
    threshold = 0.8  # Similarity threshold for clustering
    clusters = []
    
    for i in range(len(patches)):
        # Check if this patch belongs to any existing cluster
        assigned = False
        for cluster in clusters:
            for j in cluster:
                if similarity_matrix[i][j] > threshold:
                    cluster.append(i)
                    assigned = True
                    break
            if assigned:
                break
        
        # If not assigned to any cluster, create a new one
        if not assigned:
            clusters.append([i])
    
    # Find the largest cluster
    if not clusters:
        return None
    
    largest_cluster = max(clusters, key=len)
    
    # Return the patch from the center of the largest cluster
    # (you could also implement a consensus mechanism here)
    cluster_center = largest_cluster[0]  # Simple approach: just take the first patch in cluster
    return patches[cluster_center]

def universal_self_consistency(instance, patches, responses, generator, args, file_lock, formatted_files):
    """Implement Universal Self-Consistency by using the model to select the best patch."""
    if not patches:
        return None
    
    # Format the patches for the selection prompt
    formatted_patches = ""
    for i, (patch, response) in enumerate(zip(patches, responses)):
        formatted_patches += f"\n--- SOLUTION {i+1} ---\n"
        formatted_patches += f"{patch}\n\n"
    
    # Create the selection prompt - try to use a condensed version of the formatted files
    # to avoid exceeding token limits
    condensed_files = ""
    max_files_tokens = args.usc_max_input_tokens - num_tokens_from_messages(
        USC_SELECTION_PROMPT.format(
            n_samples=len(patches),
            problem_statement=instance["problem_description"],
            formatted_files="",
            patches=formatted_patches
        )
    )
    
    # Include as many files as possible without exceeding token limit
    for idx, file in enumerate(instance["found_files"]):
        file_content = f'### {file}\n{instance["file_contents"][idx]}\n'
        if num_tokens_from_messages(condensed_files + file_content) > max_files_tokens:
            print(f"USC: Maximum context length for files exceeded after {idx} files")
            break
        condensed_files += file_content
    
    # Create the selection prompt
    selection_prompt = USC_SELECTION_PROMPT.format(
        n_samples=len(patches),
        problem_statement=instance["problem_description"],
        formatted_files=condensed_files if condensed_files else "Files omitted due to context length constraints",
        patches=formatted_patches
    )
    
    # Check if the selection prompt exceeds the token limit
    prompt_tokens = num_tokens_from_messages(selection_prompt)
    if prompt_tokens > args.usc_max_input_tokens:
        print(f"Warning: USC selection prompt exceeds token limit ({prompt_tokens} > {args.usc_max_input_tokens})")
        print(f"Falling back to direct consistency selection for {instance['instance_id']}")
        return direct_consistency(patches)
    
    # Create a deep copy of args for the selection step
    selection_args = deepcopy(args)
    selection_args.max_completion_tokens = min(args.usc_max_completion_tokens, 15000)  # Cap at 15,000 
    selection_args.temp = 0.0  # Use deterministic generation for selection
    selection_args.max_retries = 1  # Allow a few retries if needed
    
    # Get the model to select the best patch
    print(f"Using Universal Self-Consistency to select best patch for {instance['instance_id']}")
    
    try:

        if not os.path.exists(f"{args.output_folder}/temp_preds.jsonl"):
            with open(f"{args.output_folder}/temp_preds.jsonl", "w", encoding="utf-8") as outfile:
                pass

        # Generate using the selection prompt
        selection_response = generator.generate(
            instance,
            selection_prompt,
            selection_args,
            file_lock,
            f"{args.output_folder}/temp_preds.jsonl"
        )
        
        # Parse the selection response to find which patch was selected
        selection_match = re.search(r"SELECTED_PATCH:\s*(\d+)", selection_response)
        if not selection_match:
            print(f"Failed to parse selection response for {instance['instance_id']}")
            return direct_consistency(patches)  # Fall back to direct consistency
        
        selected_index = int(selection_match.group(1)) - 1  # Convert to 0-indexed
        if selected_index < 0 or selected_index >= len(patches):
            print(f"Invalid selection index {selected_index} for {instance['instance_id']}")
            return direct_consistency(patches)  # Fall back to direct consistency
        
        print(f"USC selected patch {selected_index + 1} for {instance['instance_id']}")
        
        # Save the selection reasoning for debugging
        debug_dir = os.path.join(args.output_folder, "debug", instance["instance_id"])
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "usc_selection.txt"), "w", encoding="utf-8") as f:
            f.write(f"Selected patch: {selected_index + 1}\n\n")
            f.write(f"Selection reasoning:\n{selection_response}\n")
        
        return patches[selected_index]
    
    except Exception as e:
        print(f"Error in USC selection for {instance['instance_id']}: {e}")
        return direct_consistency(patches)  # Fall back to direct consistency

def process_instance(instance, args, file_lock):
    if args.instance_id is not None and args.instance_id != instance['instance_id']:
        return
    
    if args.repo_name is not None and args.repo_name not in instance['instance_id']:
        return
    
    # Format files for context
    formatted_files = format_files_for_context(instance, args)
    
    # Create the prompt
    prompt = AGENTLESS_PROMPT.format(
        problem_statement=instance["problem_description"],
        retrieval=formatted_files,
    )
    
    # Get the generator
    generator = get_generator(args.backend)
    if not generator:
        raise ValueError(f"Unsupported backend: {args.backend}")
    generator.initialize_output_files(args)
    
    # Use different token limits for generation step if USC is used
    if args.consistency_strategy == "universal":
        # For USC, we need to save some context space for the selection step
        # So reduce the max tokens for the generation step
        generation_args = deepcopy(args)
        generation_args.max_input_tokens = min(args.max_input_tokens, 110000)
        generation_args.max_completion_tokens = min(args.max_completion_tokens, 4000)
    else:
        generation_args = args
    
    # Generate multiple diverse patches
    patches, responses = generate_diverse_patches(
        instance, generation_args, generator, prompt, file_lock
    )
    
    # Apply consistency mechanism based on the specified strategy
    if args.consistency_strategy == "direct":
        final_patch = direct_consistency(patches)
    elif args.consistency_strategy == "clustering":
        final_patch = similarity_clustering(patches)
    elif args.consistency_strategy == "universal":
        final_patch = universal_self_consistency(
            instance, patches, responses, generator, args, file_lock, formatted_files
        )
    else:
        raise ValueError(f"Unsupported consistency strategy: {args.consistency_strategy}")
    
    # Save the final result
    if final_patch:
        save_result(instance, final_patch, args.output_file, file_lock)
        print(f"Successfully generated self-consistent patch for {instance['instance_id']}")
    else:
        print(f"Failed to generate self-consistent patch for {instance['instance_id']}")
    
    # Save debugging information
    save_debug_info(instance, patches, responses, args)

def format_files_for_context(instance, args):
    """Format files for the prompt context."""
    formatted_files = ""
    for idx, file in enumerate(instance["found_files"]):
        if idx < args.max_files:
            formatted_file = f'### {file}\n{instance["file_contents"][idx]}\n'
            
            # Check if adding this file would exceed the context limit
            test_prompt = AGENTLESS_PROMPT.format(
                problem_statement=instance["problem_description"],
                retrieval=formatted_files + formatted_file,
            )
            
            if num_tokens_from_messages(test_prompt) > args.max_input_tokens:
                print(f"Maximum context length exceeded for instance: {instance['instance_id']} after {idx + 1} files")
                break
            else:
                formatted_files += formatted_file
    
    return formatted_files

def save_result(instance, patch, output_file, file_lock):
    """Save the final patch to the output file."""
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            result = {
                "model_name_or_path": "agentless_lite",
                "instance_id": instance["instance_id"],
                "model_patch": patch,
            }
            f.write(json.dumps(result) + "\n")

def save_debug_info(instance, patches, responses, args):
    """Save detailed debug information about the self-consistency process."""
    debug_dir = os.path.join(args.output_folder, "debug", instance["instance_id"])
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save all generated patches
    for i, (patch, response) in enumerate(zip(patches, responses)):
        with open(os.path.join(debug_dir, f"patch_{i}.diff"), "w", encoding="utf-8") as f:
            f.write(patch)
        
        with open(os.path.join(debug_dir, f"response_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(response)
    
    # Save consistency information
    consistency_info = {
        "instance_id": instance["instance_id"],
        "num_patches_generated": len(patches),
        "consistency_strategy": args.consistency_strategy,
        "num_valid_patches": len(patches),
    }
    
    with open(os.path.join(debug_dir, "consistency_info.json"), "w", encoding="utf-8") as f:
        json.dump(consistency_info, f, indent=4)

def batch(args):
    file_lock = threading.Lock()
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "debug"), exist_ok=True)
    
    # Clear output file if it exists
    with open(args.output_file, "w") as f:
        pass
    
    # Load retrieval data
    with open(args.loc_file, "r", encoding="utf-8") as infile:
        retrieval_data = [json.loads(line.strip()) for line in infile]
    
    # Process instances
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
        description="Generate patches using self-consistency from multiple weak model runs"
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
        help="Maximum number of tokens allowed in the completion response",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Weak model to use for generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of patches to generate for consistency checking (default: 10)",
    )
    parser.add_argument(
        "--consistency_strategy",
        type=str,
        default="direct",
        choices=["direct", "clustering", "universal"],
        help="Strategy to use for determining the most consistent patch",
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        help="Process only a specific instance ID",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        help="Process only instances from a specific repository",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of concurrent threads for processing",
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
        help="The backend service to use for generation",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="self_consistency_results",
        help="Folder to save the output files",
    )
    parser.add_argument(
        "--usc_max_input_tokens",
        type=int,
        default=111000,
        help="Maximum number of tokens allowed in the Universal Self-Consistency input prompt",
    )
    parser.add_argument(
        "--usc_max_completion_tokens",
        type=int,
        default=15000,
        help="Maximum number of tokens allowed in the Universal Self-Consistency completion",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0,
        help="Initial temperature for generations",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Maximum number of retries per temperature",
    )
    parser.add_argument(
        "--reuse_patches",
        action="store_true",
        help="Reuse previously generated patches instead of generating new ones",
    )
    parser.add_argument(
        "--reuse_patches_dir",
        type=str,
        help="Directory containing previously generated patches to reuse",
    )
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--warming", action="store_true")
    parser.add_argument("--enable_weave", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    args.testbed_dir = os.path.join(args.base_path, "testbed")
    args.output_file = os.path.join(args.output_folder, os.path.basename(args.output_file))
    
    # Log parameters
    os.makedirs(os.path.join(args.output_folder, "logs"), exist_ok=True)
    log_file = os.path.join(args.output_folder, "logs", "self_consistency_parameters.json")
    with open(log_file, "w", encoding="utf-8") as f:
        args_dict = {
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }
        json.dump(args_dict, f, indent=4)
    
    batch(args)
    print("Finished self-consistency generation for all instances.")