import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # 정규화. 
    ])