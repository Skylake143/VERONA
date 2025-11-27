import pandas as pd
from pathlib import Path

# Define the path to the results directory
results_dir = Path('examples/CNNYangBig/TransferLearned/results/pgd_27-11-2025+21_34/results')

# Read the CSV file
df = pd.read_csv(results_dir / 'result_df.csv')

# Sort by network column
df_sorted = df.sort_values('network')

# Save the sorted dataframe
df_sorted.to_csv(results_dir / 'result_df_sorted.csv', index=False)

# Display first few rows to verify
print(df_sorted.head())
print(f"\nSorted by network. Shape: {df_sorted.shape}")
print(f"Unique networks: {df_sorted['network'].unique()}")