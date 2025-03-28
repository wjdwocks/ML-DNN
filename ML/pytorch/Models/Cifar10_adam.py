import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import matplotlib.pyplot as plt

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.maxpool1 = nn.MaxPool2d((3,3), 2) # (16, 16)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 45, kernel_size=3, padding='same')  # 입력 채널을 64로 수정
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*8*45, 100)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 10)
    
    def ResidualStage1(self, x):
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        # bn
        # relu
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        # bn
        # out = out + x
        # out = relu(out)
        # return out
        
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
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    train_datasets = torchvision.datasets.CIFAR10(
        root='ML/pytorch/Datasets/CIFAR10',
        train=True,
        transform=pipeline,
        download=True
    )
    
    test_datasets = torchvision.datasets.CIFAR10(
        root='ML/pytorch/Datasets/CIFAR10',
        train=False,
        transform=pipeline,
        download=True
    )
    
    train_length = int(len(train_datasets) * 0.8)
    val_length = len(train_datasets) - train_length
    
    train_datasets, val_datasets = random_split(train_datasets, (train_length, val_length))
    
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_datasets, batch_size=64, shuffle=False)
    
    model = myModel()
    
    # GPU가 사용 가능하면 모델을 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracys = []
    
    def train_model(model, train_loader, test_loader, val_loader, optimizer, criterion, epochs, max_patience):
        patience = 0
        best_model = model.state_dict()
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 입력과 타겟을 GPU로 이동
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
                
            train_accuracy = (train_correct / train_total)
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)  # 입력과 타겟을 GPU로 이동
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, dim=1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)
                    
            val_accuracy = (val_correct / val_total)
            val_loss /= len(val_loader)
            losses.append(val_loss)
            accuracys.append(val_accuracy)
            
            print(f'{epoch+1} : train_accuracy - {train_accuracy:.3f}, train_loss - {train_loss:.3f}, val_accuracy - {val_accuracy:.3f}, val_loss - {val_loss:.3f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_model = model.state_dict()
            else:
                patience += 1
                if patience >= max_patience:
                    break
        
        torch.save(best_model, 'ML/pytorch/Cifar-10_Adam.pth')
    
    train_model(model, train_loader, test_loader, val_loader, optimizer, criterion, epochs=20, max_patience=2)
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 입력과 타겟을 GPU로 이동
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
            
    print(f'Test Set의 Accuracy : {(test_correct/test_total):.3f}, Test Set의 loss : {(test_loss / len(test_loader)):.3f}')
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(range(1, len(losses) + 1), losses)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('Validation Loss')
    ax[0].set_title('Cifar-10 Adam lr=0.001')
    ax[1].plot(range(1, len(losses) + 1), accuracys)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('Validation Acc')
    ax[1].set_title('Cifar-10 Adam lr=0.001')
    plt.show()
