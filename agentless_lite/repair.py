import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import weave

from agentless_lite.util.backends import get_generator
from agentless_lite.util.repair import num_tokens_from_messages

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import traceback
import random
random.seed(42)

try:
    # Load CodeBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move model to GPU if available
except:
    print("Couldn't load model")

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

def get_target_files(instance, patch):
    """
    Extract file names from a git diff patch and retrieve their contents from instance.
    Format the results as a string with file paths followed by their contents.
    
    Parameters:
    -----------
    instance : dict
        Dictionary containing 'found_files' (list of file paths) and 'file_contents'
    patch : str
        Git diff patch string
    
    Returns:
    --------
    str
        Formatted string with file paths and contents
    """
    # Initialize result string
    result = ""
    
    # Split the patch into lines
    lines = patch.split('\n')
    
    # Set to keep track of processed files (to avoid duplicates)
    processed_files = set()
    
    # Extract file paths from diff lines
    for line in lines:
        if line.startswith('diff --git'):
            # Extract file path (takes the b/ path)
            parts = line.split()
            file_path = parts[3][2:]  # Remove the b/ prefix
            
            # Skip if we've already processed this file
            if file_path in processed_files:
                continue
            
            # Find the index of the file path in instance['found_files']
            if file_path in instance['found_files']:
                processed_files.add(file_path)
                index = instance['found_files'].index(file_path)
                # Get the corresponding file content
                content = instance['file_contents'][index]
                
                # Add to result string
                result += f"File: {file_path}\n\n"
                result += f"{content}\n\n"
    
    return result


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

        
def load_fs(instance_id, fs_path, fs_mode="random_all", fs_k=3, eval_path='sb-cli-reports/swe-bench_lite__test__agentless_lite_base_o3-mini-2025-01-31.json'):
    """
    Load few-shot examples of similar problem statements and their solutions
    from the same repository, using either embedding-based similarity or random selection.
    
    Args:
        instance_id (str): The ID of the current instance
        fs_path (str): Path to the JSONL file containing patches from the strong model
        fs_mode (str): Mode for selecting examples - "similarity_all", "random_all", "similarity_successful" or "random_successful"
        fs_k (int): Number of examples to retrieve
    
    Returns:
        str or None: Formatted few-shot examples if found, None otherwise
    """
    if not os.path.exists(fs_path):
        print(f"No model predictions file found at {fs_path}")
        return None, None, None
    
    try:
        # Extract repo name from the instance_id
        parts = instance_id.split('__')
        if len(parts) < 2:
            print(f"Invalid instance ID format: {instance_id}")
            return None, None, None
            
        repo_identifier = parts[1]
        repo_name = repo_identifier.split('-')[0]
        
        # Current instance data
        current_instance = None
        repo_instances = []
        
        # First, load the localization file to get problem descriptions and file contexts
        with open(args.loc_file, "r", encoding="utf-8") as loc_file:
            for line in loc_file:
                instance_data = json.loads(line.strip())
                instance_repo = instance_data['instance_id'].split('__')[1].split('-')[0]
                
                # Check if this instance is from the same repository
                if instance_repo == repo_name:
                    # Extract the relevant information
                    instance_info = {
                        'instance_id': instance_data['instance_id'],
                        'problem_description': instance_data['problem_description'],
                        'found_files': instance_data.get('found_files', []),
                        'file_contents': instance_data.get('file_contents', [])
                    }
                    
                    if instance_data['instance_id'] == instance_id:
                        current_instance = instance_info
                    else:
                        repo_instances.append(instance_info)
        
        # Then, load patches from the strong model
        valid_instances = {}
        with open(fs_path, "r", encoding="utf-8") as preds_file:
            for line in preds_file:
                pred_data = json.loads(line.strip())
                pred_instance_id = pred_data.get('instance_id')
                model_patch = pred_data.get('model_patch')
                
                # Only consider instances with valid patches
                if pred_instance_id and model_patch:
                    valid_instances[pred_instance_id] = model_patch
    
        if not current_instance or not repo_instances:
            print(f"No problem description or repository instances found for {instance_id}")
            return None, None, None
        
        # Filter out instances that don't have valid patches
        valid_repo_instances = [
            instance for instance in repo_instances 
            if instance['instance_id'] in valid_instances
        ]
        
        # If using a 'successful' mode, further filter to only include instances that 
        # were successfully RESOLVED (not just with valid patches)
        if 'successful' in fs_mode:
            # Load the evaluation results to get the list of resolved IDs
            try:
                with open(eval_path, 'r', encoding='utf-8') as eval_file:
                    eval_data = json.load(eval_file)
                    resolved_ids = set(eval_data.get('resolved_ids', []))
                    
                # Filter instances to only include those that were successfully resolved
                valid_repo_instances = [
                    instance for instance in valid_repo_instances
                    if instance['instance_id'] in resolved_ids
                ]
                
                if not valid_repo_instances:
                    print(f"No successfully resolved issues found for repository {repo_name}")
                    return None, None, None
                    
            except Exception as e:
                print(f"Error loading evaluation results from {eval_path}: {e}")
                print("Falling back to using all valid instances")
                # Continue with all valid instances if evaluation file can't be loaded

        if not valid_repo_instances:
            print(f"No valid patches found for repository {repo_name}")
            return None, None, None
        
        # Get current problem statement and files
        current_problem = current_instance['problem_description']
        current_files = set(current_instance['found_files'])
        
        # If we have too few instances, just use all of them
        if len(valid_repo_instances) <= fs_k:
            selected_instances = valid_repo_instances
        else:
            # Select examples based on specified mode
            if "random" in fs_mode:
                # Random selection
                selected_instances = random.sample(valid_repo_instances, fs_k)
            
            elif "similarity" in fs_mode:
                # Use CodeBERT for embedding-based similarity
                try:
                    
                    # Function to get embeddings
                    def get_embeddings(texts):
                        # Tokenize
                        encoded_input = tokenizer(texts, padding=True, truncation=True, 
                                                 max_length=512, return_tensors='pt')
                        
                        # Move tensors to the same device as the model
                        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                        
                        # Get model output
                        with torch.no_grad():
                            model_output = model(**encoded_input)
                        
                        # Mean pooling
                        attention_mask = encoded_input['attention_mask']
                        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        
                        return embeddings.cpu().numpy()
                    
                    # Get embeddings for all problem descriptions
                    problem_texts = [current_problem] + [instance['problem_description'] for instance in valid_repo_instances]
                    embeddings = get_embeddings(problem_texts)
                    
                    # Calculate cosine similarity
                    query_embedding = embeddings[0].reshape(1, -1)
                    problem_embeddings = embeddings[1:]
                    
                    # Compute semantic similarities
                    
                    semantic_similarities = cosine_similarity(query_embedding, problem_embeddings).flatten()
                    
                    # Calculate file overlap similarity (as additional signal)
                    file_similarities = []
                    for instance in valid_repo_instances:
                        instance_files = set(instance['found_files'])
                        # Jaccard similarity for file sets
                        if len(current_files) == 0 and len(instance_files) == 0:
                            file_sim = 1.0  # Both have no files
                        else:
                            file_sim = len(current_files.intersection(instance_files)) / max(1, len(current_files.union(instance_files)))
                        file_similarities.append(file_sim)
                    
                    # Combine semantic and file similarities
                    combined_similarities = [0.8 * sem_sim + 0.2 * file_sim 
                                            for sem_sim, file_sim in zip(semantic_similarities, file_similarities)]
                    
                    # Sort instances by combined similarity
                    similarity_pairs = list(zip(valid_repo_instances, combined_similarities))
                    similarity_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Select top k
                    selected_instances = [pair[0] for pair in similarity_pairs[:fs_k]]
                    
                except ImportError as e:
                    print(f"Warning: Transformers library not available, falling back to TF-IDF: {e}")
                    # Fallback to TF-IDF if transformers not available
                    
                    
                    # Create TF-IDF vectors for problem descriptions
                    all_problems = [current_problem] + [
                        instance['problem_description'] for instance in valid_repo_instances
                    ]
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(all_problems)
                    
                    # Calculate text similarity
                    query_vector = tfidf_matrix[0:1]
                    similarities = cosine_similarity(query_vector, tfidf_matrix[1:]).flatten()
                    
                    # Calculate combined similarities with file overlap
                    file_similarities = []
                    for instance in valid_repo_instances:
                        instance_files = set(instance['found_files'])
                        file_sim = len(current_files.intersection(instance_files)) / max(1, len(current_files.union(instance_files)))
                        file_similarities.append(file_sim)
                    
                    combined_similarities = [0.8 * text_sim + 0.2 * file_sim 
                                            for text_sim, file_sim in zip(similarities, file_similarities)]
                    
                    # Sort and select top k
                    similarity_pairs = list(zip(valid_repo_instances, combined_similarities))
                    similarity_pairs.sort(key=lambda x: x[1], reverse=True)
                    selected_instances = [pair[0] for pair in similarity_pairs[:fs_k]]
            else:
                print(f"Unknown few-shot mode: {fs_mode}, falling back to random selection")
                selected_instances = random.sample(valid_repo_instances, fs_k)
        
        # Format few-shot examples
        few_shot_examples = []
        for instance in selected_instances:
            similar_id = instance['instance_id']
            patch = valid_instances[similar_id]
            
            # Count shared files
            instance_files = set(instance['found_files'])
            shared_files = current_files.intersection(instance_files)
            
            target_files = get_target_files(instance, patch)

            example = f"""
--- BEGIN EXAMPLE PROBLEM ---
{instance['problem_description']}
--- END EXAMPLE PROBLEM ---

--- BEGIN SHARED FILES ({len(shared_files)} shared) ---
{', '.join(shared_files) if shared_files else 'None'}
--- END SHARED FILES ---

--- BEGIN TARGET FILES ---
{target_files}
--- END TARGET FILES ---

--- BEGIN SOLUTION ---
{patch}
--- END SOLUTION ---
"""
            few_shot_examples.append(example)
        
        similar = 'similar ' if 'similarity' in fs_mode else ''
        successful = ' that was able to successfully solve these example issues' if 'successful' in fs_mode else ''
        return "\n\n".join(few_shot_examples), similar, successful
    
    except Exception as e:
        print(f"Error loading few-shot examples for {instance_id}: {e}")
        traceback.print_exc()
        return None, None, None


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
            instance['instance_id'], 
            args.fs_path, 
            args.use_fs,  # Mode will be "random_all", "similarity_all", etc
            args.fs_k,
            args.eval_path
        )
        if few_shot_examples:
            repair_prompt = AGENTLESS_PROMPT_WITH_FEW_SHOT
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
    parser.add_argument(
        "--use_fs",
        type=str,
        default="random_all",
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

    batch(args)
    print("Finished Generation for all.")
