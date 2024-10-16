import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import matplotlib.pyplot as plt

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, padding='same') # 3채널 이미지 데이터를 3x3커널 15개와? 합성곱 연산을 수행하여 out_channel을 15로 만듦.
        # samepadding으로 나온 특성맵도 32*32 크기를 유지하도록 함.
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(15, 45, kernel_size=3, padding='same')
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(8*8*45, 100)
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
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 각 채널(R, G, B)의 픽셀값을 mean=0.5, std=0.5로 정규화한다.
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
    
    train_datasets, val_datasets = torch.utils.data.random_split(train_datasets, (train_length, val_length))
    
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=64, shuffle=True)
    
    model = myModel()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
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
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(dim=0)
                
            train_accuracy = (train_correct / train_total)
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, dim=1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(dim=0)
                    
            val_accuracy = (val_correct / val_total)
            val_loss /= len(val_loader)
            losses.append(val_loss)
            
            print(f'{epoch+1}s : train_accuracy - {train_accuracy:.3f} train_loss - {train_loss:.3f}, val_accuracy - {val_accuracy:.3f}, val_loss - {val_loss:.3f}')
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                patience = 0
                best_model = model.state_dict()
            else :
                patience += 1
                if patience >= max_patience:
                    break
        
        torch.save(best_model, 'ML/pytorch/Cifar-10_best_model.pth')
    train_model(model, train_loader, test_loader, val_loader, optimizer, criterion, epochs=20, max_patience=2)
    
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(dim=0)
            
    print(f'Test Set의 Accuracy : {(test_correct/test_total):.3f}, Test Set의 loss : {(test_loss / len(test_loader)):.3f}')
            
                
    
        
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('epoch')
    plt.ylabel('Validation Loss')
    plt.title('Cifar-10 Validation Loss in SGD + Momentum')
    plt.show()