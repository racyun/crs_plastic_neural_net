import matplotlib.pyplot as plt
from google.colab import drive
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class ThreeLayerNet(nn.Module):
    def __init__(self):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(257, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 257)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x