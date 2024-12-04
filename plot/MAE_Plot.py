import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
Origin_log = pd.read_csv('plot/origin_MCNN.csv')
FL_MCNN_split = pd.read_csv('plot/FL_MCNN_split.csv')

# Calculate the average MAE over every 10 epochs for both dataframes
Origin_log['Epoch_Group'] = (Origin_log['Epoch'] - 1) // 10
FL_MCNN_split['Epoch_Group'] = (FL_MCNN_split['Epoch'] - 1) // 10

Origin_log_avg = Origin_log.groupby('Epoch_Group').mean().reset_index()
FL_MCNN_split_avg = FL_MCNN_split.groupby('Epoch_Group').mean().reset_index()

# Calculate overall MAE averages
Origin_log_mean_mae = Origin_log['mae'].mean()
FL_MCNN_split_mean_mae = FL_MCNN_split['mae'].mean()

# Calculate percentage difference
percent_diff = ((Origin_log_mean_mae - FL_MCNN_split_mean_mae) / Origin_log_mean_mae) * 100

# Plot MAE over epochs
plt.figure(figsize=(16, 12))

# Line plot for MAE values
plt.plot(
    Origin_log_avg['Epoch_Group'] * 10 + 1,
    Origin_log_avg['mae'],
    label='Origin_log MAE (Averaged)',    
    marker='o',
    linestyle='--',
    color='blue'
)
plt.plot(
    FL_MCNN_split_avg['Epoch_Group'] * 10 + 1,
    FL_MCNN_split_avg['mae'],
    label='FL_MCNN_split MAE (Averaged)',
    marker='o',
    color='orange'
)


# Adding labels and title
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('MAE', fontsize=20)
plt.legend(fontsize=14)
plt.grid(True)


# Modify legend box size and position
plt.legend(
    fontsize=23,          # Increase font size
    loc='upper right',    # Legend position
    frameon=True,         # Add a frame around the legend
    )
# Increase tick label size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Show the plot
plt.show()
