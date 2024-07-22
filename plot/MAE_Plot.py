import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
Origin_log = pd.read_csv('plot/origin_MCNN.csv')
FL_MCNN = pd.read_csv('plot/FL_MCNN.csv')
FL_MCNN_split = pd.read_csv('plot/FL_MCNN_split.csv')

# Calculate the average MAE over every 5 epochs for both dataframes
Origin_log['Epoch_Group'] = (Origin_log['Epoch'] - 1) // 5
FL_MCNN['Epoch_Group'] = (FL_MCNN['Epoch'] - 1) // 5
FL_MCNN_split['Epoch_Group'] = (FL_MCNN_split['Epoch'] - 1) // 5


Origin_log_avg = Origin_log.groupby('Epoch_Group').mean().reset_index()
FL_MCNN_avg = FL_MCNN.groupby('Epoch_Group').mean().reset_index()
FL_MCNN_split_avg = FL_MCNN_split.groupby('Epoch_Group').mean().reset_index()



plt.figure(figsize=(12, 6))


plt.plot(Origin_log_avg['Epoch_Group'] * 5 + 1, Origin_log_avg['mae'], label='Origin_log MAE (Averaged)', marker='o')

plt.plot(FL_MCNN_avg['Epoch_Group'] * 5 + 1, FL_MCNN_avg['mae'], label='FL_MCNN MAE (Averaged)', marker='x')

plt.plot(FL_MCNN_split_avg['Epoch_Group'] * 5 + 1, FL_MCNN_split_avg['mae'], label='FL_MCNN_split MAE (Averaged)', marker='o')


# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Averaged MAE over Epochs (5 Epochs Grouped)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
