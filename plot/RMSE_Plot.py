import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
Origin_log = pd.read_csv('plot/origin_MCNN.csv')
FL_MCNN_split = pd.read_csv('plot/FL_MCNN_split.csv')


# Calculate the average RMSE over every 5 epochs for both dataframes
Origin_log['Epoch_Group'] = (Origin_log['Epoch'] - 1) // 10
FL_MCNN_split['Epoch_Group'] = (FL_MCNN_split['Epoch'] - 1) // 10


Origin_log_avg = Origin_log.groupby('Epoch_Group').mean().reset_index()
FL_MCNN_split_avg = FL_MCNN_split.groupby('Epoch_Group').mean().reset_index()



plt.figure(figsize=(12, 6))


plt.plot(Origin_log_avg['Epoch_Group'] * 5 + 1, Origin_log_avg['rmse'], label='Origin_log RMSE (Averaged)', marker='o', linestyle='--')

plt.plot(FL_MCNN_split_avg['Epoch_Group'] * 5 + 1, FL_MCNN_split_avg['rmse'], label='FL_MCNN_split RMSE (Averaged)', marker='o')


# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Averaged RMSE over Epochs (10 Epochs Grouped)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
