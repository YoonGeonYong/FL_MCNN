import torch
import numpy as np
import flwr as fl
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.metricszoo import Mae, Rmse, Huber

# Configuration
train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

lr = 0.00001

# Load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.train()

# Optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

# DataLoader
train_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
val_loader = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

def train(net, train_loader, optimizer, epochs=1):
    net.train()
    for epoch in range(epochs):
        for blob in train_loader:
            im_data = blob['data']
            gt_data = blob['gt_density']
            density_map = net(im_data, gt_data)
            loss = net.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, weights):
    state_dict = {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), weights)}
    net.load_state_dict(state_dict, strict=True)

class CrowdCountingClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return get_weights(net)

    def set_parameters(self, parameters):
        set_weights(net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, optimizer, epochs=1)
        return self.get_parameters(), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Additional metrics
        mae_metric = Mae()
        rmse_metric = Rmse()
        huber_metric = Huber(delta=1.0)

        for blob in val_loader:
            im_data = torch.tensor(blob['data'], dtype=torch.float32).to(device)
            gt_data = torch.tensor(blob['gt_density'], dtype=torch.float32).to(device)
            density_map = net(im_data, gt_data)
            mae_metric.collect(density_map, gt_data)
            rmse_metric.collect(density_map, gt_data)
            huber_metric.collect(density_map, gt_data)

        metrics = {
            "mae": float(mae_metric.summarize()),
            "rmse": float(rmse_metric.summarize()),
            "huber": float(huber_metric.summarize())
        }

        return metrics["mae"], len(val_loader), metrics

if __name__ == "__main__":
    client = CrowdCountingClient().to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
