import os
import argparse
import json
import re
import pandas as pd
from collections import defaultdict
from pathlib import Path
import difflib

# Mapping of method codes to proper names
METHOD_NAMES = {
    "base_strong": "Base Strong",
    "base_weak": "Base Weak",
    "sc_direct": "Self-Consistency - Direct",
    "sc_clustering": "Self-Consistency - Clustering",
    "sc_universal": "Self-Consistency - Universal",
    "best_of_n": "Best of N",
    "first_strong": "Strong LM Single Attempt",
    "prompt_reduction": "Prompt Reduction",
    "fallback": "Weak LM First",
    "plan": "Plan",
    "instance_faq": "Instance Level QA Pairs",
    "router_weak": "Weak Router",
    "router_strong": "Strong Router",
    "fs_random_successful_1": "1 Shot Succesfull - Random",
    "fs_random_successful_5": "5 Shot Succesfull - Random",
    "fs_similarity_successful_1": "1 Shot Succesfull - Similarity",
    "fs_similarity_successful_5": "5 Shot Succesfull - Similarity",
    "repograph": "Repo Structure",
    "repo_faq": "Repo Level QA Pairs",
    "info": "Repo Summary"
}

def parse_diff(diff_text):
    """
    Parse a diff text to extract file paths, modules, and line numbers.
    Returns:
        - files: list of modified files
        - modules: list of modified modules (e.g., django.utils.autoreload)
        - line_ranges: list of (file, start_line, end_line) tuples
    """
    if not diff_text:
        return [], [], []
    
    files = []
    modules = []
    line_ranges = []
    
    # Split diff into file sections
    file_sections = re.split(r'diff --git ', diff_text)
    if file_sections[0] == '':
        file_sections = file_sections[1:]
    
    for section in file_sections:
        # Extract file path
        file_match = re.search(r'a/(.*?) b/', section)
        if not file_match:
            continue
        
        file_path = file_match.group(1)
        files.append(file_path)
        
        # Convert file path to module path for Python files
        if file_path.endswith('.py'):
            module_path = file_path.replace('/', '.').replace('.py', '')
            modules.append(module_path)
        
        # Extract line numbers
        chunk_headers = re.findall(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', section)
        for start, length, _, _ in chunk_headers:
            start_line = int(start)
            length = int(length) if length else 1
            line_ranges.append((file_path, start_line, start_line + length - 1))
    
    return files, modules, line_ranges

def calculate_overlap(range1, range2):
    """Calculate overlap between two line ranges"""
    start1, end1 = range1
    start2, end2 = range2
    
    # Check if there's no overlap
    if end1 < start2 or end2 < start1:
        return 0
    
    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return overlap_end - overlap_start + 1

def calculate_accuracy(gold_items, pred_items):
    """
    Calculate precision, recall, and F1 score
    """
    if not gold_items or not pred_items:
        return 0.0, 0.0, 0.0
    
    true_positives = len(set(gold_items) & set(pred_items))
    precision = true_positives / len(pred_items) if pred_items else 0.0
    recall = true_positives / len(gold_items) if gold_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_line_accuracy(gold_ranges, pred_ranges):
    """
    Calculate line-level accuracy by comparing line ranges
    """
    # Group ranges by file
    gold_by_file = defaultdict(list)
    pred_by_file = defaultdict(list)
    
    for file, start, end in gold_ranges:
        gold_by_file[file].append((start, end))
    
    for file, start, end in pred_ranges:
        pred_by_file[file].append((start, end))
    
    # Count total lines in gold and predicted patches
    total_gold_lines = sum(end - start + 1 for _, start, end in gold_ranges)
    total_pred_lines = sum(end - start + 1 for _, start, end in pred_ranges)
    
    # Count overlapping lines
    overlapping_lines = 0
    for file in set(gold_by_file.keys()) & set(pred_by_file.keys()):
        for gold_range in gold_by_file[file]:
            for pred_range in pred_by_file[file]:
                overlapping_lines += calculate_overlap(gold_range, pred_range)
    
    # Calculate precision, recall, F1
    precision = overlapping_lines / total_pred_lines if total_pred_lines > 0 else 0.0
    recall = overlapping_lines / total_gold_lines if total_gold_lines > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def process_best_of_n(file_prefix, n, args):
    """
    Process Best of N results by collecting patches from multiple temperature files
    """
    predictions = {}
    base_dir = os.path.join(args.results_dir, f"sc_direct_{args.weak_model}")
    
    # Collect all patches for each instance across temperature files
    instance_patches = defaultdict(list)
    
    # Check for temperature-based files
    for i in range(n):
        temp_file = os.path.join(base_dir, f"temp_{i}.jsonl")
        if not os.path.exists(temp_file):
            print(f"Warning: Best of N file {temp_file} not found.")
            continue
            
        with open(temp_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                instance_id = data['instance_id']
                patch = data.get('model_patch', '')
                if patch:  # Only add non-empty patches
                    instance_patches[instance_id].append(patch)
    
    # Choose the "best" patch for each instance (taking the first non-empty one)
    for instance_id, patches in instance_patches.items():
        if patches:  # Use the first available patch if any exist
            predictions[instance_id] = patches[0]
    
    return predictions

def main(args):
    # Load gold patches
    gold_patches = {}
    with open(args.gold_patches, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold_patches[data['instance_id']] = data['patch']
    
    results = []
    
    # Process each experiment
    for exp in args.methods:
        # Special case for best_of_n
        if exp == "best_of_n":
            predictions = process_best_of_n(exp, args.best_of_n, args)
        else:
            # Handle base_strong and base_weak special cases
            if exp == "base_strong":
                folder_name = f"base_{args.strong_model}"
            elif exp == "base_weak":
                folder_name = f"base_{args.weak_model}"
            else:
                # For all other experiments including first_strong and router_strong, use weak model
                folder_name = f"{exp}_{args.weak_model}"
            
            exp_file = os.path.join(args.results_dir, folder_name, "all_preds.jsonl")
            
            if not os.path.exists(exp_file):
                print(f"Warning: File {exp_file} not found. Skipping.")
                continue
            
            # Load predicted patches
            predictions = {}
            with open(exp_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    predictions[data['instance_id']] = data.get('model_patch', '')
        
        # Calculate accuracy metrics for each instance
        file_precisions, file_recalls, file_f1s = [], [], []
        module_precisions, module_recalls, module_f1s = [], [], []
        line_precisions, line_recalls, line_f1s = [], [], []
        
        for instance_id in set(gold_patches.keys()) & set(predictions.keys()):
            gold_diff = gold_patches[instance_id]
            pred_diff = predictions[instance_id]
            
            # Parse diffs
            gold_files, gold_modules, gold_line_ranges = parse_diff(gold_diff)
            pred_files, pred_modules, pred_line_ranges = parse_diff(pred_diff)
            
            # File-level accuracy
            file_precision, file_recall, file_f1 = calculate_accuracy(gold_files, pred_files)
            file_precisions.append(file_precision)
            file_recalls.append(file_recall)
            file_f1s.append(file_f1)
            
            # Module-level accuracy
            module_precision, module_recall, module_f1 = calculate_accuracy(gold_modules, pred_modules)
            module_precisions.append(module_precision)
            module_recalls.append(module_recall)
            module_f1s.append(module_f1)
            
            # Line-level accuracy
            line_precision, line_recall, line_f1 = calculate_line_accuracy(gold_line_ranges, pred_line_ranges)
            line_precisions.append(line_precision)
            line_recalls.append(line_recall)
            line_f1s.append(line_f1)
        
        # Calculate averages
        file_precision = sum(file_precisions) / len(file_precisions) if file_precisions else 0
        file_recall = sum(file_recalls) / len(file_recalls) if file_recalls else 0
        file_f1 = sum(file_f1s) / len(file_f1s) if file_f1s else 0
        
        module_precision = sum(module_precisions) / len(module_precisions) if module_precisions else 0
        module_recall = sum(module_recalls) / len(module_recalls) if module_recalls else 0
        module_f1 = sum(module_f1s) / len(module_f1s) if module_f1s else 0
        
        line_precision = sum(line_precisions) / len(line_precisions) if line_precisions else 0
        line_recall = sum(line_recalls) / len(line_recalls) if line_recalls else 0
        line_f1 = sum(line_f1s) / len(line_f1s) if line_f1s else 0
        
        # Store results
        method_name = METHOD_NAMES.get(exp, exp)
        results.append({
            'Method': method_name,
            'File Precision': file_precision,
            'File Recall': file_recall,
            'File F1': file_f1,
            'Module Precision': module_precision,
            'Module Recall': module_recall,
            'Module F1': module_f1,
            'Line Precision': line_precision,
            'Line Recall': line_recall,
            'Line F1': line_f1
        })
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create base filename pattern with model names
    base_filename = f"_{args.strong_model}_{args.weak_model}"
    
    # Create and save dataframes
    df = pd.DataFrame(results)
    
    # Format all float columns to 2 decimal places
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = df[col].apply(lambda x: round(x * 100, 2))  # Convert to percentage with 2 decimal places
    
    # Save all metrics
    df.to_csv(os.path.join(args.output_dir, f'all_metrics{base_filename}.csv'), index=False)
    
    # Save separate CSVs for each metric type
    file_df = df[['Method', 'File Precision', 'File Recall', 'File F1']]
    file_df.to_csv(os.path.join(args.output_dir, f'file_metrics{base_filename}.csv'), index=False)
    
    module_df = df[['Method', 'Module Precision', 'Module Recall', 'Module F1']]
    module_df.to_csv(os.path.join(args.output_dir, f'module_metrics{base_filename}.csv'), index=False)
    
    line_df = df[['Method', 'Line Precision', 'Line Recall', 'Line F1']]
    line_df.to_csv(os.path.join(args.output_dir, f'line_metrics{base_filename}.csv'), index=False)
    
    print(f"Results saved to {args.output_dir} with model identifier {base_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate localization accuracy for SWE-bench patches')
    parser.add_argument('--gold-patches', type=str, default='swebench_lite_gold_patches.jsonl',
                        help='Path to gold patches JSONL file')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing experiment results')
    parser.add_argument('--strong-model', type=str, default='o4-mini-2025-04-16',
                        help='Strong model name for file paths')
    parser.add_argument('--weak-model', type=str, default='gpt-4o-mini-2024-07-18',
                        help='Weak model name for file paths')
    parser.add_argument('--best-of-n', type=int, default=8,
                        help='Number of temperature files for best_of_n method')
    parser.add_argument('--output-dir', type=str, default='localization_metrics',
                        help='Directory to save output CSV files')
    parser.add_argument('--methods', nargs='+', default=[
        "base_strong", "base_weak", "sc_direct", "sc_clustering", "sc_universal",
        "best_of_n", "first_strong", "prompt_reduction", "fallback", "plan",
        "instance_faq", "router_weak", "router_strong", "fs_random_successful_1",
        "fs_random_successful_5", "fs_similarity_successful_1", "fs_similarity_successful_5",
        "repograph", "repo_faq", "info"
    ], help='List of experiment methods to process')
    
    args = parser.parse_args()
    main(args)