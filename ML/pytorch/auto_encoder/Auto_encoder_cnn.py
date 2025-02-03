import numpy as np
import os
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# ✅ GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class model_AE(nn.Module):
    def __init__(self):
        super(model_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding='same'),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding='same')
        )
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        x = self.decoder(x_encoded)
        return x, x_encoded


if __name__ == '__main__':
    pipeline = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_datasets = torchvision.datasets.MNIST(
        root='../Datasets/MNIST',
        train=True,
        transform=pipeline,
        download=True
    )
    test_datasets = torchvision.datasets.MNIST(
        root='../Datasets/MNIST',
        train=False,
        transform=pipeline,
        download=True
    )
    
    train_size = int(len(train_datasets) * 0.8)
    val_size = len(train_datasets) - train_size
    
    train_datasets, val_datasets = random_split(train_datasets, (train_size, val_size))
    
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=True)
    
    # ✅ 모델을 GPU로 이동
    model = model_AE().to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_epoch = 5
    for epoch in range(n_epoch):
        for data in train_loader:
            img, _ = data  # label은 사용하지 않음 (AutoEncoder)
            img = img.to(device)  # ✅ 데이터를 GPU로 이동
            
            output, _ = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {loss.item():.4f}')
    
    print("Training Complete")
    
    # ✅ 테스트 (GPU에서 CPU로 변환 후 시각화)
    for idx, (images, _) in enumerate(test_loader):
        images = images.to(device)  # ✅ 데이터를 GPU로 이동
        fake_images, latent_space = model(images) # Latent Space를 보자
        gap = nn.AdaptiveAvgPool2d((28, 28))
        latent_space = gap(latent_space)
        latent_space = latent_space[0]
        latent_space_img = latent_space.cpu().detach().numpy()
        plt.imshow(latent_space.reshape(28, 28), cmap='gray')
        plt.show()
        plt.savefig(os.path.join('./cnn_latent_space.png'))
        plt.close()

        # ✅ GPU -> CPU로 변환 후 numpy로 변환 (detach 필요)
        orig = images[0].cpu().detach().numpy()
        recon = fake_images[0].cpu().detach().numpy()

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(orig.reshape(28, 28), cmap='gray')
        axes[0].set_title("Original")
        axes[1].imshow(recon.reshape(28, 28), cmap='gray')
        axes[1].set_title("Reconstructed")
        plt.show()
            
        plt.savefig(os.path.join('./', f"cnn_result_{idx}.png"))
        plt.close(fig)
        
        if idx == 5:
            break