# evaluate_model.py
from src.crowd_count import CrowdCounter
import src.network as network
import numpy as np
import torch

def evaluate_model(net, data_loader):
    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:
        im_data = torch.tensor(blob['data'], dtype=torch.float32).to(net.device)
        gt_data = torch.tensor(blob['gt_density'], dtype=torch.float32).to(net.device)
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data.cpu().numpy())
        et_count = np.sum(density_map)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
    mae = mae / data_loader.get_num_samples()
    mse = np.sqrt(mse / data_loader.get_num_samples())
    return mae, mse
