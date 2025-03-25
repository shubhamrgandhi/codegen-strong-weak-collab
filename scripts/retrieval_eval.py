import json
import argparse
import os
import re
import numpy as np
from typing import List, Dict, Set, Optional
from pathlib import Path
from datasets import load_dataset


def extract_files_from_patch(patch: str) -> Set[str]:
    """
    Extract file paths from a git diff patch.
    
    Args:
        patch (str): Git diff patch content
        
    Returns:
        Set[str]: Set of file paths found in the patch
    """
    files = set()
    # Match the common git diff header pattern
    pattern = r"diff --git a/(.*?) b/(.*?)$"
    
    for line in patch.split('\n'):
        match = re.match(pattern, line)
        if match:
            # Both paths should be the same, but we'll extract both to be safe
            files.add(match.group(1))
            files.add(match.group(2))
    
    return files


def normalize_path(path: str) -> str:
    """
    Normalize file path for consistent comparison.
    
    Args:
        path (str): File path to normalize
        
    Returns:
        str: Normalized path
    """
    # Convert to Path object and back to string to normalize separators
    return str(Path(path))


def evaluate_retrieval(retrieved_files: List[str], gold_file: str, 
                      max_files: int = 5) -> Dict[str, float]:
    """
    Evaluate retrieval performance by comparing retrieved files to the gold file,
    focusing on the ranking of the gold file in the results.
    
    Args:
        retrieved_files (List[str]): List of retrieved file paths
        gold_file (str): The gold standard file path
        max_files (int): Maximum number of files to consider from retrieved list
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Normalize paths for consistent comparison
    retrieved = [normalize_path(f) for f in retrieved_files[:max_files]]
    normalized_gold = normalize_path(gold_file)
    
    # Check if gold file is in retrieved files
    is_found = normalized_gold in retrieved
    
    # Get rank of the gold file (0-indexed)
    try:
        rank = retrieved.index(normalized_gold)
        # MRR uses 1-indexed rank
        reciprocal_rank = 1.0 / (rank + 1)
    except ValueError:
        rank = -1
        reciprocal_rank = 0.0
    
    # Calculate metrics
    precision = 1.0 / len(retrieved) if is_found else 0
    recall = 1.0 if is_found else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": reciprocal_rank,  # Primary ranking metric
        "is_found": is_found,
        "rank": rank,
        "retrieved_count": len(retrieved)
    }


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict]: List of JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_gold_patches() -> Dict[str, str]:
    """
    Load gold patches from SWEBench-lite test dataset and extract the file path.
    Since all patches touch only one file, we extract just that file.
    
    Returns:
        Dict[str, str]: Dictionary mapping example IDs to the gold file path
    """
    print("Loading SWEBench-lite dataset from HuggingFace...")
    
    # Load the dataset using the test split
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    gold_files = {}
    
    # Process each example
    for example in dataset:
        instance_id = example['instance_id']
        patch = example['patch']
        
        # Extract files from the patch
        files = extract_files_from_patch(patch)
        
        if files:  # Only add if we found files
            # Since all patches touch only one file, take the first one
            gold_file = next(iter(files))
            gold_files[instance_id] = gold_file
    
    return gold_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate file retrieval method against gold patches")
    parser.add_argument("--results_file", help="Path to the JSONL results file")
    parser.add_argument("--max_files", type=int, default=5, 
                        help="Maximum number of files to consider from retrieved lists")
    parser.add_argument("--output", help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Load gold patches and extract file paths
    gold_files_by_id = load_gold_patches()
    
    if not gold_files_by_id:
        print("Error: No gold patches found. Unable to proceed with evaluation.")
        return
    
    print(f"Loaded {len(gold_files_by_id)} gold patches.")
    
    # Load results
    results_data = load_jsonl(args.results_file)
    
    print(f"Loaded {len(results_data)} results.")
    
    # Create dictionary for easier lookup, using instance_id as the key
    results_by_id = {item.get("instance_id", item.get("id", i)): item 
                     for i, item in enumerate(results_data)}
    
    # Print some example keys to help debug ID matching
    print("Example gold patch IDs:", list(gold_files_by_id.keys())[:3])
    print("Example result IDs:", list(results_by_id.keys())[:3])
    
    # Collect common IDs
    common_ids = set(gold_files_by_id.keys()) & set(results_by_id.keys())
    
    print(f"Found {len(common_ids)} common examples for evaluation.")
    
    if not common_ids:
        print("Error: No common examples found between gold patches and results.")
        print("Check that the instance_id fields match between your datasets.")
        return
    
    # Initialize aggregated metrics
    total_metrics = {
        "mrr": 0.0,
        "correct": 0,
        "total": 0
    }
    
    # Store per-example results
    per_example_results = []
    
    # Process each example
    for example_id in common_ids:
        gold_file = gold_files_by_id[example_id]
        
        # Get retrieved files
        retrieved_files = results_by_id[example_id].get("found_files", [])
        
        # Evaluate
        eval_result = evaluate_retrieval(retrieved_files, gold_file, args.max_files)
        
        # Add to per-example results
        per_example_results.append({
            "id": example_id,
            "metrics": eval_result,
            "gold_file": gold_file,
            "retrieved_files": retrieved_files[:args.max_files]
        })
        
        # Accumulate metrics
        total_metrics["mrr"] += eval_result["mrr"]
        if eval_result["is_found"]:
            total_metrics["correct"] += 1
        total_metrics["total"] += 1
    
    # Calculate final metrics
    overall_metrics = {
        "mrr": total_metrics["mrr"] / total_metrics["total"] if total_metrics["total"] > 0 else 0,
        "accuracy": total_metrics["correct"] / total_metrics["total"] if total_metrics["total"] > 0 else 0,
        "total_examples": total_metrics["total"]
    }
    
    # Calculate rank distribution
    rank_distribution = {i: 0 for i in range(-1, args.max_files)}
    for result in per_example_results:
        rank = result["metrics"]["rank"]
        rank_distribution[rank] += 1
    
    # Convert to percentages
    rank_percentage = {}
    for rank, count in rank_distribution.items():
        percentage = (count / total_metrics["total"]) * 100 if total_metrics["total"] > 0 else 0
        if rank == -1:
            rank_percentage["not_found"] = percentage
        else:
            rank_percentage[f"rank_{rank}"] = percentage
    
    # Prepare final results
    evaluation_results = {
        "per_example": per_example_results,
        "overall": overall_metrics,
        "rank_distribution": rank_distribution,
        "rank_percentage": rank_percentage,
        "method": os.path.basename(args.results_file)
    }
    
    # Print summary results
    print("\nEvaluation Summary:")
    print("=" * 80)
    print(f"Number of examples evaluated: {total_metrics['total']}")
    print(f"Method: {os.path.basename(args.results_file)}")
    
    print("\nOverall Metrics:")
    print("-" * 80)
    print(f"Mean Reciprocal Rank (MRR): {overall_metrics['mrr']:.4f}")
    print(f"Accuracy (file found in top {args.max_files}): {overall_metrics['accuracy']:.4f}")
    
    print("\nRank Distribution:")
    print("-" * 80)
    for rank in range(args.max_files):
        count = rank_distribution[rank]
        percentage = (count / total_metrics["total"]) * 100 if total_metrics["total"] > 0 else 0
        print(f"Rank {rank}: {count} examples ({percentage:.2f}%)")
    
    count_not_found = rank_distribution[-1]
    percentage_not_found = (count_not_found / total_metrics["total"]) * 100 if total_metrics["total"] > 0 else 0
    print(f"Not found: {count_not_found} examples ({percentage_not_found:.2f}%)")
    
    # Save results if output path is provided
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()