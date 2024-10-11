import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


class myModel(nn.Module()):
    def __init__(self):
        super(self, myModel).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding='same')
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 32, 3, 1, 'same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 100)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    
    


if __name__ == '__main__':
    best_accuracy_list = []
    def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, lr, max_patience = 2):
        patience = 0
        min_val_loss = float('inf')
        best_model = model.state_dict()
        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad() # 각 파라미터 기울기 초기화.
                outputs = model(inputs) # 학습을 진행 (순전파, forward)
                loss = criterion(outputs, targets) # 손실값 계산.
                loss.backward() # 손실값으로 각 파라미터 가중치 계산.
                optimizer.step() # 가중치로 각 파라미터 기울기 업데이트.
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1) # 예측된 값을 tensor로 구함.
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(dim=0)
            
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, dim=1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(dim=0)
            
            print(f'Train Accuracy : {(train_correct/train_total)}.3f, Val Accuracy : {(val_correct/val_total)}:.3f, Train loss : {train_loss}:.3f, Val loss : {val_loss}:.3f')
            
            if val_loss < min_val_loss:
                patience = 0
                min_val_loss = val_loss
                best_model = model.state_dict()
            else:
                patience += 1
                if patience >= max_patience:
                    break
        torch.save(best_model, f'MNIST_adam_relu_best_CE_{lr}.pth')
    
    
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_datasets = torchvision.datasets.MNIST(
        root='pytorch/MNIST',
        train=True,
        transform=pipeline,
        download=True
    )
    test_datasets = torchvision.datasets.MNIST(
        root='pytorch/MNIST',
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
    
    lr_list = [0.0001, 0.001, 0.01]


    for lr in lr_list:
        optimizer = torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        model = myModel()
        train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50, lr = lr, max_patience=2)
    