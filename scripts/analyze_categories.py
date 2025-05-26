import os
import json
import pandas as pd
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_categories(strong_model="o4-mini-2025-04-16", weak_model="gpt-4o-mini-2024-07-18", best_of_n_runs=10):
    """
    Analyzes SWE-bench method performance across different categories and creates heatmaps.
    
    Args:
        strong_model (str): Strong model identifier
        weak_model (str): Weak model identifier
        best_of_n_runs (int): Number of runs to consider for best_of_n method
    """
    # Updated list of methods to analyze
    methods = [
        "base_strong",  # Moving base_strong above base_weak
        "base_weak",
        "sc_direct",
        "sc_clustering", 
        "sc_universal",
        "best_of_n",
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
    
    # Load categories from the CSV file
    categories_df = pd.read_csv("swebench_lite_classifications.csv")
    
    # Get unique values for each categorical field
    category_fields = [
        "description quality", 
        "solution in description", 
        "line location", 
        "function location", 
        "file location"
    ]
    
    # Dictionary to store category values and their counts
    category_values = {}
    for field in category_fields:
        # Get all values including None/NaN
        field_values = categories_df[field].copy()
        
        # Count occurrences of each value including None
        value_counts = {}
        
        # Handle non-null values
        for value in field_values.dropna().unique():
            value_counts[value] = len(categories_df[categories_df[field] == value])
        
        # Handle None/NaN values separately
        none_count = categories_df[field].isna().sum()
        if none_count > 0:
            value_counts["None"] = none_count
            
        category_values[field] = value_counts
    
    # Create results dataframe
    results = []
    
    # Process each method
    for method in methods:
        # Special handling for base_strong method
        if method == "base_strong":
            file_path = f"sb-cli-reports_{strong_model}_{weak_model}/swe-bench_lite__test__agentless_lite_base_{strong_model}.json"
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            # Load the JSON results file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get resolved IDs
            resolved_ids = set(data.get("resolved_ids", []))
        
        # Special handling for base_weak method (renamed from "base")
        elif method == "base_weak":
            file_path = f"sb-cli-reports_{strong_model}_{weak_model}/swe-bench_lite__test__agentless_lite_base_{weak_model}.json"
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            # Load the JSON results file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get resolved IDs
            resolved_ids = set(data.get("resolved_ids", []))
            
        # Special handling for best_of_n method
        elif method == "best_of_n":
            # Collect resolved IDs from all runs
            all_resolved_ids = set()
            
            # Use sc_direct as the base method for best_of_n
            base_method = "sc_direct"
            
            for i in range(best_of_n_runs):
                file_path = f"sb-cli-reports_{strong_model}_{weak_model}/swe-bench_lite__test__agentless_lite_{base_method}_{weak_model}_{i}.json"
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found for best_of_n run {i}: {file_path}")
                    continue
                
                with open(file_path, 'r') as f:
                    run_data = json.load(f)
                
                run_resolved_ids = set(run_data.get("resolved_ids", []))
                all_resolved_ids.update(run_resolved_ids)
            
            resolved_ids = all_resolved_ids
            
        # Handle prompt_reduction specially
        elif method == "prompt_reduction":
            file_path = f"sb-cli-reports_{strong_model}_{weak_model}/swe-bench_lite__test__agentless_lite_{method}_{strong_model}.json"
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            # Load the JSON results file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get resolved IDs
            resolved_ids = set(data.get("resolved_ids", []))
            
        # Normal method handling
        else:
            file_path = f"sb-cli-reports_{strong_model}_{weak_model}/swe-bench_lite__test__agentless_lite_{method}_{weak_model}.json"
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            # Load the JSON results file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get resolved IDs
            resolved_ids = set(data.get("resolved_ids", []))
        
        # Create a row for this method
        row = {"Method": method}
        
        # Calculate success rates for each category and its values
        for field in category_fields:
            for value, count in category_values[field].items():
                if value == "None":
                    # Handle None values
                    resolved_in_category = len([
                        instance_id for instance_id in resolved_ids
                        if instance_id in categories_df["instance_id"].values and 
                        pd.isna(categories_df.loc[categories_df["instance_id"] == instance_id, field].iloc[0])
                    ])
                else:
                    # Handle normal values
                    resolved_in_category = len([
                        instance_id for instance_id in resolved_ids
                        if instance_id in categories_df["instance_id"].values and 
                        categories_df.loc[categories_df["instance_id"] == instance_id, field].iloc[0] == value
                    ])
                
                # Calculate percentage
                percentage = (resolved_in_category / count) * 100 if count > 0 else 0
                
                # Add to row with category count in column name
                # Using full column name for the master CSV
                full_column_name = f"{field} - {value} ({count})"
                row[full_column_name] = round(percentage, 2)
        
        results.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add average row
    average_row = {"Method": "Average"}
    for column in results_df.columns:
        if column != "Method":
            average_row[column] = round(results_df[column].mean(), 2)
    
    results_df = pd.concat([results_df, pd.DataFrame([average_row])], ignore_index=True)
    
    # Sort columns based on average values in descending order
    if len(results_df) > 0 and len(results_df.columns) > 1:
        # Get the last row (average row)
        average_values = results_df.iloc[-1]
        
        # Determine sort order for columns (keep Method first, sort others by average value)
        cols_to_sort = [col for col in results_df.columns if col != "Method"]
        sorted_cols = sorted(cols_to_sort, key=lambda x: average_values[x], reverse=True)
        column_order = ["Method"] + sorted_cols
        
        # Reorder columns
        results_df = results_df[column_order]
    
    # Create output directory if it doesn't exist
    output_dir = f"results_category_wise_csvs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create directory for heatmaps
    heatmap_dir = f"heatmaps_{strong_model}_{weak_model}"
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Save the complete results to CSV
    output_file = f"{output_dir}/strong_{strong_model}_weak_{weak_model}_category_wise_numbers.csv"
    results_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Results saved to {output_file}")
    
    # Create separate CSVs and heatmaps for each category field
    for field in category_fields:
        # Filter columns for this category
        field_columns = [col for col in results_df.columns if col != "Method" and col.startswith(f"{field} - ")]
        
        if field_columns:  # Only create CSV if we have data for this category
            # Create a new DataFrame with simplified column names
            category_df = results_df[["Method"]].copy()
            
            # Add columns with simplified names
            for col in field_columns:
                # Extract just the value and count from the column name
                value_part = col.replace(f"{field} - ", "")
                category_df[value_part] = results_df[col]
            
            # Save to CSV
            category_output_file = f"{output_dir}/strong_{strong_model}_weak_{weak_model}_{field.replace(' ', '_')}_numbers.csv"
            category_df.to_csv(category_output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"Category results saved to {category_output_file}")
            
            # Create heatmap
            create_heatmap(
                category_df, 
                field,
                f"{heatmap_dir}/{field.replace(' ', '_')}_heatmap.png",
                strong_model,
                weak_model
            )

def create_heatmap(df, category_name, output_path, strong_model, weak_model):
    """Create a heatmap for the given category results."""
    # Make a copy without the average row for plotting
    plot_df = df.copy()
    average_row = plot_df[plot_df["Method"] == "Average"]
    plot_df = plot_df[plot_df["Method"] != "Average"]
    
    # Set method as index
    plot_df = plot_df.set_index("Method")
    
    # Create the heatmap
    plt.figure(figsize=(max(12, len(plot_df.columns) * 1.5), max(10, len(plot_df) * 0.4)))
    
    # Create a custom colormap (red for min, yellow for middle, green for max)
    cmap = plt.cm.RdYlGn
    
    # Create heatmap with better color mapping
    ax = sns.heatmap(
        plot_df, 
        annot=True, 
        fmt=".1f", 
        cmap=cmap,
        linewidths=0.5,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Success Rate (%)'}
    )
    
    # Improve x-axis labels by making them multi-line
    # First get current labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    # Create multi-line labels
    # For labels like "Contains reproducible example (16)", split at space before the "("
    new_labels = []
    for label in labels:
        if "reproducible" in label:
            parts = label.split("reproducible")
            new_label = parts[0] + "reproducible\n" + parts[1]
            new_labels.append(new_label)
        elif "(" in label:
            # Split at the last space before the opening parenthesis
            parts = label.rsplit(" (", 1)
            new_label = parts[0] + "\n(" + parts[1]
            new_labels.append(new_label)
        else:
            new_labels.append(label)
    
    # Set the new labels
    ax.set_xticklabels(new_labels, rotation=0, ha='center')
    
    # Increase bottom margin to accommodate multi-line x labels
    plt.subplots_adjust(bottom=0.2)
    
    # Fine-tune the plot
    plt.title(f"{category_name.title()} Performance Comparison\nStrong: {strong_model}, Weak: {weak_model}", fontsize=14)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SWE-bench methods by category")
    parser.add_argument("--strong_model", type=str, default="o4-mini-2025-04-16", 
                        help="Strong model identifier")
    parser.add_argument("--weak_model", type=str, default="gpt-4o-mini-2024-07-18", 
                        help="Weak model identifier")
    parser.add_argument("--best_of_n_runs", type=int, default=8,
                        help="Number of runs to consider for best_of_n method")
    
    args = parser.parse_args()
    analyze_categories(args.strong_model, args.weak_model, args.best_of_n_runs)