import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
mae_mse_log = pd.read_csv('plot/mae_mse_log.csv')
evaluate_metrics = pd.read_csv('plot/evaluate_metrics.csv')

# Calculate the average MAE over every 5 epochs for both dataframes
mae_mse_log['Epoch_Group'] = (mae_mse_log['Epoch'] - 1) // 5
evaluate_metrics['Epoch_Group'] = (evaluate_metrics['Epoch'] - 1) // 5

mae_mse_log_avg = mae_mse_log.groupby('Epoch_Group').mean().reset_index()
evaluate_metrics_avg = evaluate_metrics.groupby('Epoch_Group').mean().reset_index()

# Plot the averaged MAE values from both dataframes
plt.figure(figsize=(12, 6))

# Plot MAE from mae_mse_log.csv
plt.plot(mae_mse_log_avg['Epoch_Group'] * 5 + 1, mae_mse_log_avg['mae'], label='mae_mse_log MAE (Averaged)', marker='o')
# Plot MAE from evaluate_metrics.csv
plt.plot(evaluate_metrics_avg['Epoch_Group'] * 5 + 1, evaluate_metrics_avg['mae'], label='evaluate_metrics MAE (Averaged)', marker='x')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Averaged MAE over Epochs (5 Epochs Grouped)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
