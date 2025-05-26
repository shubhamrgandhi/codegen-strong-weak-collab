#!/usr/bin/env python
import argparse
import subprocess
import re
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Compute generation costs for all experiments')
    parser.add_argument('--results_dir', type=str, default="results",
                        help='Base directory for results, e.g., results_o3-mini-2025-01-31')
    parser.add_argument('--model', type=str, default="qwen-2.5-coder-7b-instruct",
                        help='Model name, e.g., qwen-2.5-coder-32b-instruct')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # List of experiments to process
    experiments = [
        "first_strong",
        "prompt_reduction",
        "fallback",
        "plan",
        "instance_faq",
        "router_weak",
        "router_strong",
        "fs_random_successful_1",
        "fs_random_successful_5",
        "fs_similarity_successful_1",
        "fs_similarity_successful_5",
        "repograph",
        "repo_faq",
        "info"
    ]
    
    results = {}
    
    # Print a header
    print(f"Computing costs for experiments in {args.results_dir}")
    print(f"Model: {args.model}")
    print("-" * 50)
    
    # Process each experiment
    for i, exp in enumerate(experiments):
        # Create the progress bar
        progress = (i / len(experiments)) * 100
        bar_length = 30
        filled_length = int(bar_length * i // len(experiments))
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print progress
        sys.stdout.write(f"\r[{bar}] {progress:.1f}% - Processing: {exp}")
        sys.stdout.flush()
        
        # Build the full experiment directory path
        exp_dir = f"{args.results_dir}/{exp}_{args.model}"
        
        # Skip if the directory doesn't exist
        if not os.path.exists(exp_dir):
            continue
        
        # Run the compute_generation_cost.py script
        try:
            cmd = ["python", "scripts/compute_generation_cost.py", "--results_dir", exp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extract the total cost using regex
            if result.returncode == 0:
                output = result.stdout
                match = re.search(r'Total cost: \$([0-9.]+)', output)
                if match:
                    total_cost = match.group(1)
                    results[exp] = total_cost
            else:
                results[exp] = "Error"
        except Exception as e:
            results[exp] = f"Error: {str(e)}"
    
    # Complete the progress bar
    sys.stdout.write("\r[" + "█" * bar_length + "] 100.0% - Processing complete\n\n")
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    for exp in experiments:
        if exp in results:
            print(f"{results[exp]}")
        else:
            print(f"N/A (directory not found)")

if __name__ == "__main__":
    main()