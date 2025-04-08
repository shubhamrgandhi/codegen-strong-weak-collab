import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Load the data
try:
    df = pd.read_csv('results.csv')
    print(f"Successfully loaded data with {len(df)} rows")
    print("Column names:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise

# Clean up the data - convert columns to numeric, removing any non-numeric characters
for col in ['Resolution Rate', 'Avg. #iterations', 'Valid Patch Rate', 'Generation Cost', 'Additional Method Cost', 'Total Cost']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert #Shots column to numeric, treating '-' as NaN and then fill with 0
df['#Shots'] = pd.to_numeric(df['#Shots'].replace('-', np.nan), errors='coerce').fillna(0)

# Create a category column for better grouping
df['Category'] = df['Experiment'].apply(lambda x: x.split(' - ')[0] if ' - ' in x else x)

# Calculate improvement over base gpt-4o-mini for Resolution Rate
# Try to get the base gpt-4o-mini resolution rate
try:
    base_row = df.loc[(df['Experiment'] == 'Base') & (df['Generator'] == 'gpt-4o-mini')]
    if not base_row.empty:
        base_gpt4o_mini_resolution = base_row['Resolution Rate'].values[0]
    else:
        # Fallback to just the Base row if the specific generator isn't found
        base_row = df.loc[df['Experiment'] == 'Base']
        if not base_row.empty:
            base_gpt4o_mini_resolution = base_row.iloc[0]['Resolution Rate']
        else:
            # If no Base row exists, use the minimum resolution rate as a fallback
            base_gpt4o_mini_resolution = df['Resolution Rate'].min()
            print("Warning: Could not find Base model. Using minimum resolution rate as baseline.")
except Exception as e:
    print(f"Error finding base resolution rate: {e}")
    base_gpt4o_mini_resolution = df['Resolution Rate'].min()
    print("Using minimum resolution rate as baseline.")
df['Resolution Rate Improvement'] = df['Resolution Rate'] - base_gpt4o_mini_resolution

# Define experiment categories for consistent coloring
categories = df['Category'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
category_colors = dict(zip(categories, colors))

# Create a figure for all metrics combined
plt.figure(figsize=(16, 12))

# 1. Resolution Rate plot
plt.subplot(2, 2, 1)
ax = sns.barplot(x='Experiment', y='Resolution Rate', hue='Category', data=df, palette=category_colors)
plt.title('Resolution Rate by Experiment', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, df['Resolution Rate'].max() * 1.1)
plt.tight_layout()

# Add values on top of bars
for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
            f'{p.get_height():.3f}', 
            fontsize=9, ha='center', va='bottom')

# 2. Average Iterations plot
plt.subplot(2, 2, 2)
ax = sns.barplot(x='Experiment', y='Avg. #iterations', hue='Category', data=df, palette=category_colors)
plt.title('Average Number of Iterations by Experiment', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, df['Avg. #iterations'].max() * 1.1)
plt.tight_layout()

# Add values on top of bars
for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
            f'{p.get_height():.2f}', 
            fontsize=9, ha='center', va='bottom')

# 3. Valid Patch Rate plot
plt.subplot(2, 2, 3)
ax = sns.barplot(x='Experiment', y='Valid Patch Rate', hue='Category', data=df, palette=category_colors)
plt.title('Valid Patch Rate by Experiment', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0.9, 1.0)  # Most values are close to 1, so adjust y-axis for better visibility
plt.tight_layout()

# Add values on top of bars
for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
            f'{p.get_height():.3f}', 
            fontsize=9, ha='center', va='bottom')

# 4. Total Cost plot
plt.subplot(2, 2, 4)
ax = sns.barplot(x='Experiment', y='Total Cost', hue='Category', data=df, palette=category_colors)
plt.title('Total Cost by Experiment', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, df['Total Cost'].max() * 1.1)
plt.tight_layout()

# Add values on top of bars
for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), 
            f'${p.get_height():.2f}', 
            fontsize=9, ha='center', va='bottom')
            
plt.tight_layout(pad=2.0)
plt.savefig('plots/metrics_combined.png', dpi=300, bbox_inches='tight')

# Create a scatter plot to show Cost vs. Resolution Rate improvement
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Total Cost'], 
                     df['Resolution Rate'], 
                     c=[category_colors[cat] for cat in df['Category']], 
                     s=100, alpha=0.7)

# Add experiment labels
for i, row in df.iterrows():
    plt.annotate(row['Experiment'], 
                 (row['Total Cost'], row['Resolution Rate']),
                 fontsize=8, 
                 xytext=(5, 5), 
                 textcoords='offset points')

# Add reference line for base gpt-4o-mini
plt.axhline(y=base_gpt4o_mini_resolution, color='r', linestyle='--', 
           label=f'Base gpt-4o-mini ({base_gpt4o_mini_resolution:.3f})')

plt.xlabel('Total Cost ($)', fontsize=12)
plt.ylabel('Resolution Rate', fontsize=12)
plt.title('Cost-Effectiveness Analysis: Resolution Rate vs. Total Cost', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('plots/cost_vs_resolution.png', dpi=300)

# Create a plot specifically for Resolution Rate by number of shots and experiment type
plt.figure(figsize=(12, 7))
shot_experiments = df[df['#Shots'].isin([1, 3, 5])]

if not shot_experiments.empty:
    # Group by Experiment type and #Shots
    grouped = shot_experiments.groupby(['Category', '#Shots'])['Resolution Rate'].mean().reset_index()
    
    # Plot
    ax = sns.barplot(x='#Shots', y='Resolution Rate', hue='Category', data=grouped)
    plt.title('Resolution Rate by Number of Shots', fontsize=14)
    plt.xlabel('Number of Shots', fontsize=12)
    plt.ylabel('Resolution Rate', fontsize=12)
    
    # Add base model reference
    plt.axhline(y=base_gpt4o_mini_resolution, color='r', linestyle='--', 
               label=f'Base gpt-4o-mini ({base_gpt4o_mini_resolution:.3f})')
    
    plt.legend(title='Experiment Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/shots_analysis.png', dpi=300)

# Create a radar chart for the different approaches
plt.figure(figsize=(10, 10))

# Select key metrics
metrics = ['Resolution Rate', 'Valid Patch Rate', 'Avg. #iterations']

# Normalize the metrics for radar chart
df_radar = df[['Experiment'] + metrics].copy()
for metric in metrics:
    if metric == 'Avg. #iterations':
        # For iterations, lower is better, so invert the normalization
        max_val = df_radar[metric].max()
        df_radar[metric] = 1 - (df_radar[metric] / max_val)
    else:
        min_val = df_radar[metric].min()
        max_val = df_radar[metric].max()
        df_radar[metric] = (df_radar[metric] - min_val) / (max_val - min_val)

# Select a subset of experiments for clarity
# Try to match key experiments with what's in the data
possible_key_experiments = ['Base', 'Raw Trajectory', 'Plan', 'High level Repo Info']
key_experiments = []
for exp in possible_key_experiments:
    # Check for exact matches first
    if exp in df_radar['Experiment'].values:
        key_experiments.append(exp)
    else:
        # Then check for partial matches
        matches = [e for e in df_radar['Experiment'].values if exp in e]
        key_experiments.extend(matches)

# If we still don't have enough experiments, include some from the data
if len(key_experiments) < 3 and len(df_radar) > 3:
    # Get some representative experiments from different categories
    categories = df_radar['Experiment'].unique()
    sample_size = min(4, len(categories))
    key_experiments = list(categories[:sample_size])

# Make sure key_experiments is not empty
if not key_experiments and not df_radar.empty:
    key_experiments = df_radar['Experiment'].unique()[:min(4, len(df_radar['Experiment'].unique()))]
df_radar = df_radar[df_radar['Experiment'].isin(key_experiments)]

# Number of variables
N = len(metrics)

# Create angles for each metric
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create the plot
ax = plt.subplot(111, polar=True)

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], metrics, size=12)

# Draw the performance for each experiment
for i, exp in enumerate(df_radar['Experiment']):
    values = df_radar.loc[df_radar['Experiment'] == exp, metrics].values.flatten().tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, label=exp)
    ax.fill(angles, values, alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Performance Comparison Across Key Metrics', size=15)
plt.tight_layout()
plt.savefig('plots/radar_chart.png', dpi=300)

print("Visualizations have been saved as PNG files.")