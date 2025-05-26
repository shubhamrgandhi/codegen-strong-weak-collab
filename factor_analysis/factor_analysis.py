import pandas as pd
import os
import re

# Function to parse the filename and extract model information
def parse_filename(filename):
    parts = filename.replace('.csv', '').split('_weak_')
    
    if len(parts) == 2:
        strong_part = parts[0]
        weak_part = parts[1]
        
        # Extract strong model name
        strong_model = strong_part.replace('strong_', '')
        
        # Extract weak model name
        weak_model = weak_part
        
        return strong_model, weak_model
    return None, None

# List of CSV files
csv_files = [
    "strong_gpt-4o-mini_weak_qwen2.5-coder-7b.csv",
    "strong_o3-mini_weak_gpt-4o-mini.csv",
    "strong_o4-mini_weak_gpt-4o-mini.csv",
    "strong_o3-mini_weak_qwen2.5-coder-7b.csv",
    "strong_o3-mini_weak_qwen2.5-coder-14b.csv",
    "strong_o3-mini_weak_qwen2.5-coder-32b.csv"
]

# Create a list to store all results
all_results = []

# Process each CSV file
for csv_file in csv_files:
    # Parse filename to get strong and weak model names
    strong_model, weak_model = parse_filename(csv_file)
    
    if strong_model and weak_model:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Calculate efficiency score for all rows
        df['Efficiency Score'] = df['Performance'] / df['Cost']
        
        # Extract Base Strong and Base Weak rows to get their metrics
        base_strong_row = df[df['Experiment'] == 'Base Strong']
        base_weak_row = df[df['Experiment'] == 'Base Weak']
        
        strong_model_performance = None
        strong_model_cost = None
        strong_model_efficiency = None
        weak_model_performance = None
        weak_model_cost = None
        weak_model_efficiency = None
        
        if not base_strong_row.empty:
            strong_model_performance = base_strong_row['Performance'].values[0]
            strong_model_cost = base_strong_row['Cost'].values[0]
            strong_model_efficiency = base_strong_row['Efficiency Score'].values[0]
        
        if not base_weak_row.empty:
            weak_model_performance = base_weak_row['Performance'].values[0]
            weak_model_cost = base_weak_row['Cost'].values[0]
            weak_model_efficiency = base_weak_row['Efficiency Score'].values[0]
        
        # Filter out Base Strong and Base Weak rows
        df = df[~df['Experiment'].isin(['Base Strong', 'Base Weak'])]
        
        # Add model information and base metrics
        df['Strong Model'] = strong_model
        df['Strong Model Base Performance'] = strong_model_performance
        df['Strong Model Base Cost'] = strong_model_cost
        df['Strong Model Base Efficiency Score'] = strong_model_efficiency
        df['Weak Model'] = weak_model
        df['Weak Model Base Performance'] = weak_model_performance
        df['Weak Model Base Cost'] = weak_model_cost
        df['Weak Model Base Efficiency Score'] = weak_model_efficiency
        
        # Select and rename columns for output
        result_df = df[[
            'Strong Model', 
            'Strong Model Base Performance',
            'Strong Model Base Cost',
            'Strong Model Base Efficiency Score', 
            'Weak Model',
            'Weak Model Base Performance',
            'Weak Model Base Cost',
            'Weak Model Base Efficiency Score', 
            'Experiment', 
            'Performance', 
            'Cost', 
            'Efficiency Score'
        ]]
        result_df = result_df.rename(columns={'Experiment': 'Method'})
        
        # Append to all results
        all_results.append(result_df)

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)

# Rank by efficiency score within each strong-weak model pair
combined_df['Rank'] = combined_df.groupby(['Strong Model', 'Weak Model'])['Efficiency Score'].rank(ascending=False)

# Sort by Strong Model, Weak Model, and Rank
combined_df = combined_df.sort_values(['Strong Model', 'Weak Model', 'Rank'])

# Save to factor_analysis.csv
combined_df.to_csv('factor_analysis.csv', index=False)

print("Analysis complete. Results saved to factor_analysis.csv")