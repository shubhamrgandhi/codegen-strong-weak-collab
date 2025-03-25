import os
import re
import csv
import argparse
from collections import defaultdict

def extract_token_counts(log_content):
    """Extract prompt, completion, and reasoning tokens from log content."""
    token_data = []
    
    # Updated pattern to correctly capture reasoning_tokens
    pattern = r"usage=CompletionUsage\(completion_tokens=(\d+), prompt_tokens=(\d+).*?completion_tokens_details=CompletionTokensDetails\(.*?reasoning_tokens=(\d+)"
    
    matches = re.findall(pattern, log_content)
    for match in matches:
        completion_tokens = int(match[0])
        prompt_tokens = int(match[1])
        reasoning_tokens = int(match[2]) if match[2] else 0
        
        token_data.append((completion_tokens, prompt_tokens, reasoning_tokens))
    
    return token_data

def calculate_cost(tokens_data, results_dir):
    """Calculate cost based on model type and token counts."""
    if "o3-mini" in results_dir:
        prompt_price = 1.1 / 1_000_000  # $1.1 per million
        completion_price = 4.4 / 1_000_000  # $4.4 per million
    elif "gpt-4o-mini" in results_dir:
        prompt_price = 0.15 / 1_000_000  # $0.15 per million
        completion_price = 0.6 / 1_000_000  # $0.6 per million
    else:
        # Default to gpt-4o-mini pricing if not specified
        prompt_price = 0.15 / 1_000_000
        completion_price = 0.6 / 1_000_000
    
    prompt_tokens = sum(pt for _, pt, _ in tokens_data)
    completion_tokens = sum(ct for ct, _, _ in tokens_data)
    reasoning_tokens = sum(rt for _, _, rt in tokens_data)
    
    prompt_cost = prompt_tokens * prompt_price
    completion_cost = completion_tokens * completion_price
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate costs from log files.")
    parser.add_argument("--results_dir", default="results_base_gpt-4o-mini-2024-07-18",
                        help="Directory containing logs subdirectory")
    args = parser.parse_args()
    
    logs_dir = os.path.join(args.results_dir, "logs")
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory not found at {logs_dir}")
        return
    
    instance_costs = {}
    all_tokens_data = []
    
    # Process each log file
    for filename in os.listdir(logs_dir):
        if not filename.endswith(".log"):
            continue
        
        instance_id = filename.split(".")[0]
        file_path = os.path.join(logs_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        tokens_data = extract_token_counts(content)
        all_tokens_data.extend(tokens_data)
        
        if tokens_data:
            instance_cost = calculate_cost(tokens_data, args.results_dir)
            instance_costs[instance_id] = instance_cost
    
    # Calculate total cost
    total_cost_data = calculate_cost(all_tokens_data, args.results_dir)
    
    # Calculate average cost per sample
    num_samples = len(instance_costs)
    avg_cost = total_cost_data["total_cost"] / num_samples if num_samples > 0 else 0
    avg_prompt_tokens = total_cost_data["prompt_tokens"] / num_samples if num_samples > 0 else 0
    avg_completion_tokens = total_cost_data["completion_tokens"] / num_samples if num_samples > 0 else 0
    avg_reasoning_tokens = total_cost_data["reasoning_tokens"] / num_samples if num_samples > 0 else 0
    
    # Write instance-wise costs to CSV
    output_file = os.path.join(args.results_dir, "instance_wise_costs.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["instance_id", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                     "prompt_cost", "completion_cost", "total_cost"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for instance_id, cost_data in instance_costs.items():
            row = {"instance_id": instance_id}
            row.update(cost_data)
            writer.writerow(row)
    
    # Prepare summary statistics
    summary_stats = [
        f"Total samples: {num_samples}",
        f"Total prompt tokens: {total_cost_data['prompt_tokens']}",
        f"Total completion tokens: {total_cost_data['completion_tokens']}",
        f"Total reasoning tokens: {total_cost_data['reasoning_tokens']}",
        f"Total prompt cost: ${total_cost_data['prompt_cost']:.4f}",
        f"Total completion cost: ${total_cost_data['completion_cost']:.4f}",
        f"Total cost: ${total_cost_data['total_cost']:.4f}",
        f"Average prompt tokens per sample: {avg_prompt_tokens:.2f}",
        f"Average completion tokens per sample: {avg_completion_tokens:.2f}",
        f"Average reasoning tokens per sample: {avg_reasoning_tokens:.2f}",
        f"Average cost per sample: ${avg_cost:.6f}"
    ]
    
    # Write summary statistics to txt file
    summary_file = os.path.join(args.results_dir, "cost_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for stat in summary_stats:
            f.write(stat + "\n")
    
    # Print summary statistics
    for stat in summary_stats:
        print(stat)
    print(f"Results saved to {output_file}")
    print(f"Summary statistics saved to {summary_file}")

if __name__ == "__main__":
    main()