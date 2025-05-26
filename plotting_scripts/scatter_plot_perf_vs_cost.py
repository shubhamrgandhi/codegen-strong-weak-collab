import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate performance vs cost plot for a specific model.')
parser.add_argument('--weak_model', type=str, required=True, help='Model name (e.g., gpt-4o-mini, o3-mini)')
parser.add_argument('--strong_model', type=str, required=True, help='Model name (e.g., gpt-4o-mini, o3-mini)')
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Read the CSV file from results_csvs directory
csv_filename = f'results_csvs/Code onboarding results - Plotting - {args.weak_model}_{args.strong_model}_strong.csv'
df = pd.read_csv(csv_filename)

# Define experiment categories
base_models = [f'Base - {args.strong_model}', f'Base - {args.weak_model}']
strong_single = ['Strong Single Attempt']
context_experiments = ['Plan', 'QA_instance', '1 Shot Random', '5 Shot Random', 
                      '1 Shot Similarity', '5 Shot Similarity', 'Repo Structure', 
                      'QA_repo', 'Repo Summary']
pipeline_experiments = ['Strong LM First', 'Prompt Reduction', 'Weak LM First']
dynamic_experiments = ['Weak Router', 'Strong Router']
cost_equated_experiments = ['Self-Consistency - Direct', 'Self-Consistency - Clustering', 
                           'Self-Consistency - Universal', 'Best of n ']

# Create figure with enough space for the full plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot each category separately with different colors and markers
categories = [
    (base_models, '#008000', 'o', f'Base Models ({args.strong_model}, {args.weak_model})'),  # Green circles
    (strong_single, '#FF0000', 's', 'Strong Single Attempt'),             # Red squares
    (context_experiments, '#0000FF', 'o', 'Context'),                     # Blue circles
    (pipeline_experiments, '#800080', '^', 'Pipeline'),                   # Purple triangles
    (dynamic_experiments, '#FFA500', 'D', 'Dynamic'),                     # Orange diamonds
    (cost_equated_experiments, '#8B4513', 'v', 'Cost-equated Weak Model') # Brown inverted triangles
]

for experiments, color, marker, label in categories:
    mask = df['Experiment'].isin(experiments)
    if mask.any():  # Only plot if there are matching experiments
        points = ax.scatter(df[mask]['Cost'], df[mask]['Performance'], 
                  c=color, marker=marker, s=100, alpha=0.8, 
                  edgecolors='black', linewidth=0.5, label=label)

# Add all experiments to the dataframe
all_experiments = []
for category in categories:
    all_experiments.extend(category[0])
df_filtered = df[df['Experiment'].isin(all_experiments)]

# Sort dataframe by Cost
df_sorted = df_filtered.sort_values('Cost')

# Build monotonic line - only include points that increase performance
line_points = []
best_experiments = []
current_best_perf = -float('inf')
for _, row in df_sorted.iterrows():  # Fixed the asterisk syntax
    if row['Performance'] > current_best_perf:
        line_points.append((row['Cost'], row['Performance']))
        best_experiments.append(row['Experiment'])
        current_best_perf = row['Performance']

# Extract x and y coordinates
line_x = [point[0] for point in line_points]
line_y = [point[1] for point in line_points]

# Plot the line
ax.plot(line_x, line_y, 'k--', alpha=0.7, linewidth=2, label='Best Performance by Cost')

# Define a better function to calculate label positions
def get_optimized_label_position(index, point, line_points, ax):
    """Calculate an optimized position for a label to avoid overlaps."""
    x, y = point
    
    # Get current plot limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Calculate x-offset based on position in plot
    x_range = x_max - x_min
    if x < x_min + 0.3 * x_range:  # Left third of plot
        x_offset = 15  # Offset to the right
    elif x > x_min + 0.7 * x_range:  # Right third of plot
        x_offset = -15  # Offset to the left
    else:  # Middle of plot
        x_offset = 15 if index % 2 == 0 else -15  # Alternate
    
    # Calculate y-offset to avoid line
    # Find closest points on the line to determine line direction
    line_indices = np.argsort([np.sqrt((lx - x)**2 + (ly - y)**2) for lx, ly in zip(line_x, line_y)])
    
    if len(line_indices) >= 2:
        closest_idx = line_indices[0]
        second_idx = line_indices[1]
        
        # Determine if line goes up or down at this point
        dx = line_x[second_idx] - line_x[closest_idx]
        dy = line_y[second_idx] - line_y[closest_idx]
        
        if abs(dx) > 0:
            slope = dy / dx
            # If line has positive slope, place label below; if negative, place above
            y_offset = -25 if slope > 0 else 25
        else:
            y_offset = 25  # Default if can't determine slope
    else:
        # Default offsets if we can't determine line direction
        y_offset = 25 if index % 2 == 0 else -25
    
    # Add variations based on index to distribute labels
    variation = 10 * (index % 3)  # Add variety based on index
    
    return x_offset + variation, y_offset + (variation if index % 2 == 0 else -variation)

# Add labels for points on the monotonic line
for i, (point, exp_name) in enumerate(zip(line_points, best_experiments)):
    # Format experiment name nicely
    short_name = exp_name
    if exp_name == f'Base - {args.weak_model}':
        short_name = f'Base - {args.weak_model}'
    elif exp_name == f'Base - {args.strong_model}':
        short_name = f'Base - {args.strong_model}'
    elif ' - ' in short_name:
        short_name = short_name.split(' - ')[0]
    
    # Get optimized position for this label
    x_offset, y_offset = get_optimized_label_position(i, point, line_points, ax)
    
    # Add annotation with a thin line connector
    ax.annotate(short_name, 
              xy=(point[0], point[1]),
              xytext=(x_offset, y_offset),
              textcoords='offset points',
              ha='center',
              va='center',
              fontsize=12,  # Smaller font size
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
              arrowprops=dict(
                  arrowstyle='-', 
                  connectionstyle=f'arc3,rad={0.1 if i % 2 == 0 else -0.1}',
                  color='gray',
                  lw=0.8,
                  alpha=0.7
              ))

# Add labels and title
ax.set_xlabel('Cost', fontsize=16)
ax.set_ylabel('Performance', fontsize=16)
ax.set_title(f'Performance vs Cost for {args.strong_model} + {args.weak_model}', fontsize=18)

# Add grid for better readability
ax.grid(True, alpha=0.3)

# Add legend
ax.legend(loc='upper left', framealpha=0.9, fontsize=14)

# Ensure the x-axis extends far enough
min_cost = df_filtered['Cost'].min()
max_cost = df_filtered['Cost'].max()
ax.set_xlim(min_cost * 0.9, max_cost * 1.1)  # Add 10% padding on both sides

# Ensure the y-axis extends far enough
min_perf = df_filtered['Performance'].min()
max_perf = df_filtered['Performance'].max()
ax.set_ylim(min_perf * 0.9, max_perf * 1.1)  # Add 10% padding on both sides

# Save the plot with model name in the filename
output_filename = f'plots/performance_vs_cost_scatter_{args.weak_model}_{args.strong_model}_monotonic.png'

# First try with a looser bbox configuration
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()

print(f"Plot saved to {output_filename}")

# For debugging, let's also print the experiments on the monotonic line
print("\nExperiments on the monotonic performance line:")
for i, (point, exp_name) in enumerate(zip(line_points, best_experiments)):
    print(f"{i+1}. {exp_name}: performance {point[1]:.4f} at cost {point[0]:.3f}")