import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ✅ GPU 설정
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Variational AutoEncoder (VAE) 모델 정의
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1), # (14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels = 2, out_channels=4, kernel_size=3, stride=2, padding=1), # (7, 7)
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1) # (4, 4)
        )
        self.mu_layer = nn.Linear(4*4*8, 2)  # 평균 (mu)
        self.log_var_layer = nn.Linear(4*4*8, 2)  # 분산 (log_var)
        self.decoder_input = nn.Linear(2, 4*4*8) # decoder에 넣기 전에 할 작업.
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 표준편차 계산
        eps = torch.randn_like(std)  # 표준정규분포에서 샘플링
        return mu + eps * std  # 재파라미터화 트릭 적용

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)  # 잠재 공간에서 샘플링
        z = self.decoder_input(z)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


# ✅ 손실 함수 (VAE는 Reconstruction Loss + KL Divergence)
def loss_function(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL Divergence
    return recon_loss + kl_divergence * 0.0001  # KL 손실 가중치 조절


if __name__ == '__main__':
    # ✅ 데이터 전처리 및 로드
    pipeline = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(root='../Datasets/MNIST', train=True, transform=pipeline, download=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # ✅ 모델 및 학습 설정
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)



    # ✅ 학습 루프
    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            img, _ = batch
            img = img.view(img.size(0), -1).to(device)

            optimizer.zero_grad()
            recon_img, mu, log_var = model(img)
            loss = loss_function(recon_img, img, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss:.4f}")

    print("Training Complete!")

    # ✅ 테스트 (이미지 복원 및 저장)
    save_dir = "./vae_results"
    os.makedirs(save_dir, exist_ok=True)

    for idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        fake_images, _, _ = model(images)

        orig = images[0].cpu().detach().numpy().reshape(28, 28)
        recon = fake_images[0].cpu().detach().numpy().reshape(28, 28)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title("Original")
        axes[1].imshow(recon, cmap='gray')
        axes[1].set_title("Reconstructed")

        plt.savefig(os.path.join(save_dir, f"vae_result_{idx}.png"))
        plt.close(fig)

        if idx == 5:
            break

    print(f"Saved reconstructed images in '{save_dir}' directory.")
