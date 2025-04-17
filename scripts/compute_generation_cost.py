import os
import re
import csv
import argparse
from collections import defaultdict
import sys

def extract_token_counts_and_model(log_content):
    """Extract prompt, completion, reasoning tokens and model name from log content."""
    token_data = []
    
    # Find all API response entries
    # First, try to match a sample entry and print it for debugging
    sample_match = re.search(r"API response ChatCompletion.*?(?=\n\d{4}-\d{2}-\d{2}|\Z)", log_content, re.DOTALL)
    if sample_match:
        sample_text = sample_match.group(0)
        # print("Sample API response entry found:")
        # print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
    else:
        print("No API response entry found for debugging. Check the log format.")
    
    # Process each line independently to increase chances of finding token usage
    for line in log_content.splitlines():
        if "model=" in line and "usage=CompletionUsage" in line:
            # Extract model name
            model_match = re.search(r"model='([^']+)'", line)
            if not model_match:
                continue
            
            model_name = model_match.group(1)
            
            # Extract token usage using a simpler pattern
            usage_match = re.search(r"completion_tokens=(\d+), prompt_tokens=(\d+)", line)
            if not usage_match:
                continue
                
            completion_tokens = int(usage_match.group(1))
            prompt_tokens = int(usage_match.group(2))
            
            # Extract reasoning tokens if present
            reasoning_tokens = 0
            reasoning_match = re.search(r"reasoning_tokens=(\d+)", line)
            if reasoning_match:
                reasoning_tokens = int(reasoning_match.group(1))
            
            token_data.append((model_name, completion_tokens, prompt_tokens, reasoning_tokens))
        
    if not token_data:
        print("Warning: No token data could be extracted from logs. Trying alternative pattern.")
        
        # Try a more aggressive approach with the exact pattern from the example
        for line in log_content.splitlines():
            if "model='" in line and "usage=CompletionUsage" in line:
                # print("Found line with model and usage information:")
                # print(line[:200] + "..." if len(line) > 200 else line)
                
                # Try to extract with the exact pattern from the example
                exact_match = re.search(r"model='([^']+)'.*?usage=CompletionUsage\(completion_tokens=(\d+), prompt_tokens=(\d+).*?reasoning_tokens=(\d+)", line)
                if exact_match:
                    model_name = exact_match.group(1)
                    completion_tokens = int(exact_match.group(2))
                    prompt_tokens = int(exact_match.group(3))
                    reasoning_tokens = int(exact_match.group(4))
                    token_data.append((model_name, completion_tokens, prompt_tokens, reasoning_tokens))
        
    # if token_data:
    #     print(f"Successfully extracted {len(token_data)} token entries")
    # else:
    #     print("Could not extract any token data with any pattern. Check log format.")
        
    return token_data

def get_model_pricing(model_name):
    """Get pricing for a specific model."""
    # Define pricing for each model
    pricing = {
        # OpenAI models
        "o3-mini": {"prompt": 1.1 / 1_000_000, "completion": 4.4 / 1_000_000},
        "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.6 / 1_000_000},
        # Other models
        "qwen2.5-coder-7b-instruct": {"prompt": 0.05 / 1_000_000, "completion": 0.1 / 1_000_000},
        "deepseek-r1-distill-llama-70b": {"prompt": 0.17 / 1_000_000, "completion": 0.4 / 1_000_000},
        # Default pricing (fallback)
        "default": {"prompt": 1.1 / 1_000_000, "completion": 4.4 / 1_000_000}
    }
    
    # Check if the model name contains any of our known model identifiers
    for model_id, price in pricing.items():
        if model_id in model_name.lower():
            return price
    
    # Return default pricing if no match found
    print(f"Warning: Unknown model '{model_name}'. Using default pricing.")
    return pricing["default"]

def calculate_cost(tokens_data):
    """Calculate cost based on model type and token counts for each API call."""
    # Initialize counters for total tokens and costs
    model_stats = defaultdict(lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "prompt_cost": 0,
        "completion_cost": 0,
        "total_cost": 0,
        "calls": 0
    })
    
    total_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "prompt_cost": 0,
        "completion_cost": 0,
        "total_cost": 0
    }
    
    # Process each API call
    for model_name, completion_tokens, prompt_tokens, reasoning_tokens in tokens_data:
        # Get pricing for this specific model
        pricing = get_model_pricing(model_name)
        
        # Calculate costs
        prompt_cost = prompt_tokens * pricing["prompt"]
        completion_cost = completion_tokens * pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        # Update model-specific statistics
        model_stats[model_name]["prompt_tokens"] += prompt_tokens
        model_stats[model_name]["completion_tokens"] += completion_tokens
        model_stats[model_name]["reasoning_tokens"] += reasoning_tokens
        model_stats[model_name]["prompt_cost"] += prompt_cost
        model_stats[model_name]["completion_cost"] += completion_cost
        model_stats[model_name]["total_cost"] += total_cost
        model_stats[model_name]["calls"] += 1
        
        # Update total statistics
        total_stats["prompt_tokens"] += prompt_tokens
        total_stats["completion_tokens"] += completion_tokens
        total_stats["reasoning_tokens"] += reasoning_tokens
        total_stats["prompt_cost"] += prompt_cost
        total_stats["completion_cost"] += completion_cost
        total_stats["total_cost"] += total_cost
    
    return model_stats, total_stats

def main():
    parser = argparse.ArgumentParser(description="Calculate costs from log files.")
    parser.add_argument("--results_dir", default="results/prompt_reduction_o3-mini-2025-01-31",
                        help="Directory containing logs subdirectory")
    args = parser.parse_args()
    
    logs_dir = os.path.join(args.results_dir, "logs")
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory not found at {logs_dir}")
        return
        
    print(f"Processing log files in: {logs_dir}")
    
    instance_costs = {}
    all_tokens_data = []
    
    # Process each log file
    for filename in os.listdir(logs_dir):
        if not filename.endswith(".log"):
            continue
        
        instance_id = filename.split(".")[0]
        file_path = os.path.join(logs_dir, filename)
        
        # print(f"Processing file: {filename}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            tokens_data = extract_token_counts_and_model(content)
            
            if tokens_data:
                # print(f"  Found {len(tokens_data)} API calls in {filename}")
                # for i, (model, completion, prompt, reasoning) in enumerate(tokens_data[:5]):  # Print first 5 for debugging
                #     print(f"  {i+1}. Model: {model}, Completion: {completion}, Prompt: {prompt}, Reasoning: {reasoning}")
                
                # if len(tokens_data) > 5:
                #     print(f"  ... and {len(tokens_data) - 5} more calls")
                
                all_tokens_data.extend(tokens_data)
                instance_model_stats, instance_total_stats = calculate_cost(tokens_data)
                instance_costs[instance_id] = {
                    "model_stats": instance_model_stats,
                    "total_stats": instance_total_stats
                }
            else:
                print(f"  Warning: No API calls found in {filename}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    # Calculate total cost across all instances
    all_model_stats, all_total_stats = calculate_cost(all_tokens_data)
    
    # Calculate average cost per sample
    num_samples = len(instance_costs)
    avg_cost = all_total_stats["total_cost"] / num_samples if num_samples > 0 else 0
    avg_prompt_tokens = all_total_stats["prompt_tokens"] / num_samples if num_samples > 0 else 0
    avg_completion_tokens = all_total_stats["completion_tokens"] / num_samples if num_samples > 0 else 0
    avg_reasoning_tokens = all_total_stats["reasoning_tokens"] / num_samples if num_samples > 0 else 0
    
    # Write instance-wise costs to CSV
    output_file = os.path.join(args.results_dir, "instance_wise_costs.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["instance_id", "model", "calls", "prompt_tokens", "completion_tokens", 
                     "reasoning_tokens", "prompt_cost", "completion_cost", "total_cost"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for instance_id, cost_data in instance_costs.items():
            # Write a row for each model used in this instance
            for model_name, model_data in cost_data["model_stats"].items():
                row = {
                    "instance_id": instance_id,
                    "model": model_name,
                    "calls": model_data["calls"],
                    "prompt_tokens": model_data["prompt_tokens"],
                    "completion_tokens": model_data["completion_tokens"],
                    "reasoning_tokens": model_data["reasoning_tokens"],
                    "prompt_cost": model_data["prompt_cost"],
                    "completion_cost": model_data["completion_cost"],
                    "total_cost": model_data["total_cost"]
                }
                writer.writerow(row)
            
            # Write a total row for this instance
            total_row = {
                "instance_id": f"{instance_id}_TOTAL",
                "model": "ALL",
                "calls": sum(data["calls"] for data in cost_data["model_stats"].values()),
                "prompt_tokens": cost_data["total_stats"]["prompt_tokens"],
                "completion_tokens": cost_data["total_stats"]["completion_tokens"],
                "reasoning_tokens": cost_data["total_stats"]["reasoning_tokens"],
                "prompt_cost": cost_data["total_stats"]["prompt_cost"],
                "completion_cost": cost_data["total_stats"]["completion_cost"],
                "total_cost": cost_data["total_stats"]["total_cost"]
            }
            writer.writerow(total_row)
    
    # Write model-wise summary to CSV
    model_summary_file = os.path.join(args.results_dir, "model_summary.csv")
    with open(model_summary_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["model", "calls", "prompt_tokens", "completion_tokens", 
                     "reasoning_tokens", "prompt_cost", "completion_cost", "total_cost"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        # Write a row for each model
        for model_name, model_data in all_model_stats.items():
            row = {
                "model": model_name,
                "calls": model_data["calls"],
                "prompt_tokens": model_data["prompt_tokens"],
                "completion_tokens": model_data["completion_tokens"],
                "reasoning_tokens": model_data["reasoning_tokens"],
                "prompt_cost": model_data["prompt_cost"],
                "completion_cost": model_data["completion_cost"],
                "total_cost": model_data["total_cost"]
            }
            writer.writerow(row)
        
        # Write a total row
        total_row = {
            "model": "ALL",
            "calls": sum(data["calls"] for data in all_model_stats.values()),
            "prompt_tokens": all_total_stats["prompt_tokens"],
            "completion_tokens": all_total_stats["completion_tokens"],
            "reasoning_tokens": all_total_stats["reasoning_tokens"],
            "prompt_cost": all_total_stats["prompt_cost"],
            "completion_cost": all_total_stats["completion_cost"],
            "total_cost": all_total_stats["total_cost"]
        }
        writer.writerow(total_row)
    
    # Prepare summary statistics
    summary_stats = [
        f"Total samples: {num_samples}",
        f"Total API calls: {sum(data['calls'] for data in all_model_stats.values())}",
        f"Total prompt tokens: {all_total_stats['prompt_tokens']}",
        f"Total completion tokens: {all_total_stats['completion_tokens']}",
        f"Total reasoning tokens: {all_total_stats['reasoning_tokens']}",
        f"Total prompt cost: ${all_total_stats['prompt_cost']:.4f}",
        f"Total completion cost: ${all_total_stats['completion_cost']:.4f}",
        f"Total cost: ${all_total_stats['total_cost']:.4f}",
        f"Average prompt tokens per sample: {avg_prompt_tokens:.2f}",
        f"Average completion tokens per sample: {avg_completion_tokens:.2f}",
        f"Average reasoning tokens per sample: {avg_reasoning_tokens:.2f}",
        f"Average cost per sample: ${avg_cost:.6f}",
        "",
        "Model-wise statistics:"
    ]
    
    # Add model-wise statistics
    for model_name, model_data in all_model_stats.items():
        model_stats = [
            f"  Model: {model_name}",
            f"    Calls: {model_data['calls']}",
            f"    Prompt tokens: {model_data['prompt_tokens']}",
            f"    Completion tokens: {model_data['completion_tokens']}",
            f"    Reasoning tokens: {model_data['reasoning_tokens']}",
            f"    Prompt cost: ${model_data['prompt_cost']:.4f}",
            f"    Completion cost: ${model_data['completion_cost']:.4f}",
            f"    Total cost: ${model_data['total_cost']:.4f}"
        ]
        summary_stats.extend(model_stats)
    
    # Write summary statistics to txt file
    summary_file = os.path.join(args.results_dir, "cost_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for stat in summary_stats:
            f.write(stat + "\n")
    
    # Print summary statistics
    for stat in summary_stats:
        print(stat)
    print(f"Results saved to {output_file}")
    print(f"Model summary saved to {model_summary_file}")
    print(f"Summary statistics saved to {summary_file}")

if __name__ == "__main__":
    main()