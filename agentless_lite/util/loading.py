import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import weave

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import traceback
import random
random.seed(42)


def get_target_files(instance, patch):
    """
    Extract file names from a git diff patch and retrieve code sections with
    15 lines of context before and after the changes.
    
    Parameters:
    -----------
    instance : dict
        Dictionary containing 'found_files' (list of file paths) and 'file_contents'
    patch : str
        Git diff patch string
    
    Returns:
    --------
    str
        Formatted string with file paths and relevant code segments
    """
    # Initialize result string
    result = ""
    
    # Split the patch into sections based on diff headers
    patch_sections = patch.split('diff --git')
    
    # Skip the first empty section if it exists
    patch_sections = [s for s in patch_sections if s.strip()]
    
    # Process each patch section
    for section in patch_sections:
        # Find the file path
        file_line = section.split('\n', 1)[0].strip()
        parts = file_line.split()
        if len(parts) >= 2:
            file_path = parts[1][2:]  # Remove the b/ prefix
        else:
            continue
        
        # Find indices in the patch that show which lines are modified
        hunk_headers = [line for line in section.split('\n') if line.startswith('@@')]
        
        # Find the index of the file path in instance['found_files']
        if file_path in instance['found_files']:
            index = instance['found_files'].index(file_path)
            # Get the corresponding file content
            full_content = instance['file_contents'][index]
            content_lines = full_content.split('\n')
            
            # Add file header to result
            result += f"File: {file_path}\n\n"
            
            # For each hunk in the patch, extract the relevant lines
            for hunk in hunk_headers:
                # Parse the @@ line to find line numbers
                # Format is typically @@ -start,count +start,count @@
                line_info = hunk.split('@@')[1].strip()
                line_parts = line_info.split()
                if line_parts and line_parts[0].startswith('-'):
                    # Extract line numbers from the original file
                    line_range = line_parts[0][1:].split(',')
                    if len(line_range) >= 1:
                        try:
                            start_line = int(line_range[0])
                            # Use 15 lines before and after as context
                            context_lines = 15
                            
                            # Calculate the range to show (with context)
                            start_idx = max(0, start_line - 1 - context_lines)
                            
                            # If we have the count of changed lines
                            if len(line_range) > 1:
                                count = int(line_range[1])
                                end_idx = min(len(content_lines), start_line - 1 + count + context_lines)
                            else:
                                # Default to showing 15 lines after the start if count is not available
                                end_idx = min(len(content_lines), start_line - 1 + 1 + context_lines)
                            
                            # Extract the relevant lines
                            relevant_section = '\n'.join(content_lines[start_idx:end_idx])
                            
                            # Add to result
                            result += f"Lines {start_idx+1}-{end_idx} (with 15 lines context):\n"
                            result += relevant_section + "\n\n"
                        except ValueError:
                            # If we can't parse the line numbers, include a note
                            result += "Couldn't parse line numbers in patch.\n\n"
            
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

        
def load_fs(instance_id, fs_path, loc_file, fs_mode="random_all", fs_k=3, eval_path='sb-cli-reports/swe-bench_lite__test__agentless_lite_base_o3-mini-2025-01-31.json', tokenizer = None, similarity_model = None):
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
        with open(loc_file, "r", encoding="utf-8") as l_file:
            for line in l_file:
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
                            model_output = similarity_model(**encoded_input)
                        
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