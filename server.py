import flwr as fl
from flwr.server.strategy import FedAvg
import pandas as pd
import os

def fit_config(rnd: int):
    return {}

def evaluate_config(rnd: int):
    return {}

def aggregate_fit(results):
    aggregated_metrics = {
        "mae":      [],
        "rmse":     [],
        "huber":    [],
    }

    for num_examples, metric_dict in results:
        for key, value in metric_dict.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)

    for key in aggregated_metrics:
        if len(aggregated_metrics[key]) > 0:
            aggregated_metrics[key] = sum(aggregated_metrics[key]) / len(aggregated_metrics[key])
        else:
            aggregated_metrics[key] = None

    return aggregated_metrics

def aggregate_evaluate(metrics: list):
    aggregated_metrics = {
        "mae":      [],
        "rmse":     [],
        "huber":    [],
    }

    for _, metric_dict in metrics:
        for key, value in metric_dict.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)

    for key in aggregated_metrics:
        if len(aggregated_metrics[key]) > 0:
            aggregated_metrics[key] = sum(aggregated_metrics[key]) / len(aggregated_metrics[key])
        else:
            aggregated_metrics[key] = None

    return aggregated_metrics

class CustomFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
  
        return aggregated_loss, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Save aggregated metrics to a CSV file
        metrics_df = pd.DataFrame([aggregated_metrics])
        metrics_df.to_csv("evaluate_metrics.csv", mode="a", header=not os.path.exists("evaluate_metrics.csv"), index=False)

        # Print aggregated evaluation metrics
        print(f"Round {rnd} evaluation metrics: {aggregated_metrics}")

        return aggregated_loss, aggregated_metrics

strategy = CustomFedAvg(
    fraction_fit                    =1.0,
    fraction_evaluate               =1.0,
    min_fit_clients                 =2,
    min_evaluate_clients            =2,
    min_available_clients           =2,
    on_fit_config_fn                =fit_config,
    on_evaluate_config_fn           =evaluate_config,
    fit_metrics_aggregation_fn      =aggregate_fit,  # Add this line to provide fit metrics aggregation function
    evaluate_metrics_aggregation_fn =aggregate_evaluate,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address  ="0.0.0.0:8080",
        config          =fl.server.ServerConfig(num_rounds=500),
        strategy        =strategy,
    )
