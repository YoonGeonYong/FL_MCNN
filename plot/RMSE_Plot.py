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

plt.figure(figsize=(12, 6))

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

# Adding labels and title with increased font size
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
# plt.title('Averaged RMSE over Epochs (10 Epochs Grouped)', fontsize=23)
plt.legend(fontsize=23)
plt.grid(True)

# Increase tick label size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Display the plot
plt.show()
