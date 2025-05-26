import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define model pairs to analyze - order by importance
model_pairs = [
    ('o3-mini', 'gpt-4o-mini', 'O3+4o'),
    ('o4-mini', 'gpt-4o-mini', 'O4+4o'),
    ('o3-mini', 'qwen2.5-coder-32b', 'O3+Q32'),
    ('o3-mini', 'qwen2.5-coder-14b', 'O3+Q14'),
    ('o3-mini', 'qwen2.5-coder-7b', 'O3+Q7'),
    ('gpt-4o-mini', 'qwen2.5-coder-7b', '4o+Q7')
]

# Define distinct colors for each model pair - using a more distinguishable palette
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
line_styles = ['-', '-', '--', '--', '-.', '-.']  # Adding line styles for extra distinction
line_widths = [3, 3, 2.5, 2.5, 2, 2]  # Different line widths for emphasis

# Create figure with better dimensions for this visualization
fig, ax = plt.subplots(figsize=(12, 9))

# Define marker styles for experiment categories
marker_map = {
    'Base Models': 'o',       # circle
    'Strong Single': 's',     # square
    'Context': '^',           # triangle
    'Pipeline': 'D',          # diamond
    'Dynamic': 'v',           # inverted triangle
    'Cost-equated': 'p'       # plus
}

# Store handles for legend
line_handles = []
category_markers = []

# Process each model pair
for i, (strong_model, weak_model, label) in enumerate(model_pairs):
    # Read the CSV file
    csv_filename = f'results_csvs/Code onboarding results - Plotting - {weak_model}_{strong_model}_strong.csv'
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Warning: Could not find file {csv_filename}")
        continue
    
    # Define experiment categories as in the original script
    base_models = [f'Base - {strong_model}', f'Base - {weak_model}']
    strong_single = ['Strong Single Attempt']
    context_experiments = ['Plan', 'QA_instance', '1 Shot Random', '5 Shot Random', 
                          '1 Shot Similarity', '5 Shot Similarity', 'RepoGraph', 
                          'QA_repo', 'Repo Summary']
    pipeline_experiments = ['Strong LM First', 'Prompt Reduction', 'Weak LM First']
    dynamic_experiments = ['Weak Router', 'Strong Router']
    cost_equated_experiments = ['Self-Consistency - Direct', 'Self-Consistency - Clustering', 
                               'Self-Consistency - Universal', 'Best of n ']
    
    # Create category lookup
    category_lookup = {}
    for exp in base_models:
        category_lookup[exp] = 'Base Models'
    for exp in strong_single:
        category_lookup[exp] = 'Strong Single'
    for exp in context_experiments:
        category_lookup[exp] = 'Context'
    for exp in pipeline_experiments:
        category_lookup[exp] = 'Pipeline'
    for exp in dynamic_experiments:
        category_lookup[exp] = 'Dynamic'
    for exp in cost_equated_experiments:
        category_lookup[exp] = 'Cost-equated'
    
    # Add all experiments to the dataframe
    all_experiments = []
    for category in category_lookup.keys():
        all_experiments.append(category)
    
    df_filtered = df[df['Experiment'].isin(all_experiments)]
    
    # Sort dataframe by Cost
    df_sorted = df_filtered.sort_values('Cost')
    
    # Build monotonic line - only include points that increase performance
    line_points = []
    best_experiments = []
    experiment_categories = []
    current_best_perf = -float('inf')
    
    for _, row in df_sorted.iterrows():
        if row['Performance'] > current_best_perf:
            line_points.append((row['Cost'], row['Performance']))
            best_experiments.append(row['Experiment'])
            category = category_lookup.get(row['Experiment'], 'Other')
            experiment_categories.append(category)
            current_best_perf = row['Performance']
    
    # Extract x and y coordinates
    if line_points:
        line_x = [point[0] for point in line_points]
        line_y = [point[1] for point in line_points]
        
        # Plot the line with model pair label
        line = ax.plot(line_x, line_y, linestyle=line_styles[i], color=colors[i], 
                     linewidth=line_widths[i], alpha=0.8, label=label, zorder=5)
        line_handles.append(line[0])
        
        # Plot the points on the line with their respective categories
        for j, ((x, y), exp_name, category) in enumerate(zip(line_points, best_experiments, experiment_categories)):
            marker_style = marker_map[category]
            
            # Plot the point
            ax.scatter(x, y, color=colors[i], marker=marker_style, s=100, 
                      edgecolors='black', linewidth=0.8, alpha=0.8, zorder=10)
            
            # Removed the endpoint label code that was here previously

# Create category marker legend items
for category, marker in marker_map.items():
    category_markers.append(
        Line2D([0], [0], marker=marker, color='gray', markerfacecolor='gray', 
               markersize=10, label=category, linestyle='None')
    )

# Add the model pair legend
first_legend = ax.legend(handles=line_handles, title="Model Pairs", 
                        loc='upper left', fontsize=18, framealpha=0.9, title_fontsize=20)

# Add the marker category legend
ax.add_artist(first_legend)  # Ensure first legend stays visible
ax.legend(handles=category_markers, title="Experiment Categories", 
         loc='lower right', fontsize=18, framealpha=0.9, title_fontsize=20)

# Add labels and title
ax.set_xlabel('Cost ($)', fontsize=20)
ax.set_ylabel('Performance', fontsize=20)
ax.set_title('Optimal Performance-Cost Trade-offs for Different Model Pairs', fontsize=22)

# Add grid for better readability
ax.grid(True, alpha=0.3)

# Set better axis limits with some padding - using hardcoded values
ax.set_xlim(0, 50)  # Adjust these values based on your actual data range
ax.set_ylim(0, 0.5)  # Adjust these values based on your actual data range

# Add a vertical line at the intersection of blue (O4+4o) and green (O3+Q32) lines
# This is approximately at cost = 18.5 (you may need to adjust this value based on actual data)
intersection_x = 18.3  # Approximate intersection point
ax.axvline(x=intersection_x, color='black', linestyle=':', alpha=0.7)

# Add label for the intersection point on the x-axis
ax.annotate(f'${intersection_x:.1f}',
            xy=(intersection_x, 0),
            xytext=(0, -20),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=18,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc='white', ec="black", alpha=0.7),
            arrowprops=dict(
                arrowstyle='->', 
                connectionstyle='arc3,rad=0.0',
                color='black',
                lw=1,
                alpha=0.7
            ))

# Save the plot
output_filename = 'plots/combined_performance_cost_curves.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()

print(f"Combined plot saved to {output_filename}")