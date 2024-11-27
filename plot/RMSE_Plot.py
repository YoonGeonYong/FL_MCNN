import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
Origin_log = pd.read_csv('plot/origin_MCNN.csv')
FL_MCNN_split = pd.read_csv('plot/FL_MCNN_split.csv')

# Calculate the average RMSE over every 10 epochs for both dataframes
Origin_log['Epoch_Group'] = (Origin_log['Epoch'] - 1) // 10
FL_MCNN_split['Epoch_Group'] = (FL_MCNN_split['Epoch'] - 1) // 10

Origin_log_avg = Origin_log.groupby('Epoch_Group').mean().reset_index()
FL_MCNN_split_avg = FL_MCNN_split.groupby('Epoch_Group').mean().reset_index()

# Calculate overall RMSE averages
Origin_log_mean_rmse = Origin_log['rmse'].mean()
FL_MCNN_split_mean_rmse = FL_MCNN_split['rmse'].mean()

# Calculate percentage difference
percent_diff = ((Origin_log_mean_rmse - FL_MCNN_split_mean_rmse) / Origin_log_mean_rmse) * 100

# Plot RMSE over epochs
plt.figure(figsize=(12, 6))

# Line plot for RMSE values
plt.plot(
    Origin_log_avg['Epoch_Group'] * 10 + 1,
    Origin_log_avg['rmse'],
    label='Origin_log RMSE (Averaged)',
    marker='o',
    linestyle='--'
)
plt.plot(
    FL_MCNN_split_avg['Epoch_Group'] * 10 + 1,
    FL_MCNN_split_avg['rmse'],
    label='FL_MCNN_split RMSE (Averaged)',
    marker='o'
)

# Add the percentage difference annotation
plt.text(
    700,  # X-coordinate (adjust based on your graph)
    max(Origin_log_avg['rmse'].max(), FL_MCNN_split_avg['rmse'].max()) * 0.9,  # Y-coordinate
    f"Difference: {percent_diff:.2f}%",
    fontsize=16,
    # color='red',
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

# Adding labels and title
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

# Increase tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show the plot
plt.title('Averaged RMSE over Epochs (10 Epochs Grouped)', fontsize=18)
plt.show()
