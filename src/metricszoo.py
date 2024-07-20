import torch
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .basemetric import BaseMetric

warnings.filterwarnings('ignore')

class BaseMetric:
    def collect(self, pred, true):
        raise NotImplementedError

    def summarize(self):
        raise NotImplementedError

class Mae(BaseMetric):
    def __init__(self):
        self.absolute_errors = []

    def collect(self, pred, true):
        pred = pred.sum().item()
        true = true.sum().item()
        self.absolute_errors.append(abs(true - pred))

    def summarize(self):
        return np.mean(self.absolute_errors)

class Mse(BaseMetric):
    def __init__(self):
        self.squared_errors = []

    def collect(self, pred, true):
        pred = pred.sum().item()
        true = true.sum().item()
        self.squared_errors.append((true - pred) ** 2)

    def summarize(self):
        return np.mean(self.squared_errors)

class Rmse(BaseMetric):
    def __init__(self):
        self.squared_errors = []

    def collect(self, pred, true):
        pred = pred.sum().item()
        true = true.sum().item()
        self.squared_errors.append((true - pred) ** 2)

    def summarize(self):
        return np.sqrt(np.mean(self.squared_errors))

class Huber(BaseMetric):
    def __init__(self, delta=1.0):
        self.errors = []
        self.delta = delta

    def collect(self, pred, true):
        pred = pred.sum().item()
        true = true.sum().item()
        error = true - pred
        if abs(error) <= self.delta:
            self.errors.append(0.5 * (error ** 2))
        else:
            self.errors.append(self.delta * (abs(error) - 0.5 * self.delta))

    def summarize(self):
        return np.mean(self.errors)
