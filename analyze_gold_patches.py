import argparse
from datasets import load_dataset
import re
from collections import Counter


def extract_files_from_patch(patch: str) -> set:
    """
    Extract file paths from a git diff patch.
    
    Args:
        patch (str): Git diff patch content
        
    Returns:
        set: Set of file paths found in the patch
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


def analyze_patches():
    """
    Analyze the gold patches in the test split of SWEBench-lite
    to check how many files each patch touches.
    """
    print("Loading SWEBench-lite test dataset from HuggingFace...")
    
    # Load the dataset using the test split
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    print(f"Loaded {len(dataset)} patches from the test split.")
    
    # Count how many files each patch affects
    files_per_patch = []
    patches_with_multiple_files = []
    
    for example in dataset:
        instance_id = example['instance_id']
        patch = example['patch']
        
        # Extract files from the patch
        files = extract_files_from_patch(patch)
        
        # Add to our list
        file_count = len(files)
        files_per_patch.append(file_count)
        
        # Save examples with multiple files
        if file_count > 1:
            patches_with_multiple_files.append({
                'id': instance_id,
                'file_count': file_count,
                'files': list(files)
            })
    
    # Calculate statistics
    file_count_distribution = Counter(files_per_patch)
    multi_file_percentage = (sum(1 for count in files_per_patch if count > 1) / len(files_per_patch)) * 100
    
    # Print results
    print("\nResults:")
    print("=" * 80)
    print(f"Total number of patches: {len(files_per_patch)}")
    print("\nDistribution of file counts per patch:")
    for count, frequency in sorted(file_count_distribution.items()):
        percentage = (frequency / len(files_per_patch)) * 100
        print(f"{count} file(s): {frequency} patches ({percentage:.2f}%)")
    
    print("\nSummary:")
    print(f"Patches with multiple files: {sum(1 for count in files_per_patch if count > 1)} ({multi_file_percentage:.2f}%)")
    print(f"Maximum number of files in a single patch: {max(files_per_patch)}")
    
    # Print examples with multiple files
    if patches_with_multiple_files:
        print("\nExample patches with multiple files:")
        for i, example in enumerate(patches_with_multiple_files[:5], 1):  # Show at most 5 examples
            print(f"\nExample {i}:")
            print(f"  ID: {example['id']}")
            print(f"  Number of files: {example['file_count']}")
            print(f"  Files:")
            for file in example['files']:
                print(f"    - {file}")
    else:
        print("\nNo patches with multiple files found.")


if __name__ == "__main__":
    analyze_patches()