import torch
import torch.nn as nn
import src.network as network
from src.models import MCNN

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()        
        self.DME = MCNN()        
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, im_data, gt_data=None):
        im_data = torch.tensor(im_data, dtype=torch.float32).to(self.device)
        density_map = self.DME(im_data)
        
        if self.training and gt_data is not None:
            gt_data = torch.tensor(gt_data, dtype=torch.float32).to(self.device)
            self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss
