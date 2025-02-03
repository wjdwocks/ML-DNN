'''
가장 간단한 CNN구조를 이용한 MNIST 데이터 encoder - decoder로 구현해보기.
'''
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt


class model_AE(nn.Module):
    def __init__(self):
        super(model_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        # print(x.shape)
        x_encoded = self.encoder(x)
        x = self.decoder(x_encoded)
        return x, x_encoded


if __name__ == '__main__':
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_datasets = torchvision.datasets.MNIST(
        root='ML/pytorch/Datasets/MNIST',
        train=True,
        transform=pipeline,
        download=True
    )
    test_datasets = torchvision.datasets.MNIST(
        root='ML/pytorch/Datasets/MNIST',
        train=False,
        transform=pipeline,
        download=True
    )
    
    train_size = int(len(train_datasets)*0.8)
    val_size = len(train_datasets) - train_size
    test_size = len(test_datasets)
    
    train_datasets, val_datasets = torch.utils.data.random_split(train_datasets, (train_size, val_size))
    
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size = 64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=True)
    
    model = model_AE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_epoch = 10
    for epoch in range(n_epoch):
        for data in train_loader:
            img, _ = data # label은 안씀. AE니까
            img = img.view(img.size(0), -1) # Flatten함.
            output, _ = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{n_epoch}], Loss : {loss.item():.4f}')
    print("Training Complete")
    
    # Test Part
    for images, _ in test_loader:
        images = images.view(images.size(0), -1) # Flatten함.
        fake_images, _ = model(images)
        
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(images[0].reshpae(28, 28).detach().numpy())
        axes[1].imshow(fake_images[0].reshape(28, 28).detach().numpy())
        plt.show()