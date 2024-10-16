import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding='same')
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 30, kernel_size=3, padding='same')
        # pooling
        # relu
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*30, 100)
        # relu
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax
    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x
    
if __name__ == '__main__':
    pipeline = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))])
    # 각 데이터를 Tensor의 형태로 변경 → Normalize는 각 채널의 평균과 표준편차를 지정해서 표준점수로 바꿔준다. 여기선 gray_scale이라서 1채널임.
    # mean = (0.5,)와 mean = (0.5)는 다른게, (0.5,) 는 1개의 원소를 가진 튜플을 의미하고, (0.5)는 실수임.
    
    train_datasets = torchvision.datasets.MNIST( # 각 데이터는 ((img, label), ... ) 의 형태로 있고, 각각이 60000개가 있다.
        root='ML/pytorch/Datasets/MNIST',
        train=True,
        download=True,
        transform=pipeline
    )
    test_datasets = torchvision.datasets.MNIST(
        root='ML/pytorch/Datasets/MNIST',
        train=False,
        download=True,
        transform=pipeline        
    )
    
    print(len(train_datasets))
    train_size = (int)(0.8 * len(train_datasets))
    val_size = len(train_datasets) - train_size
    train_datasets, val_datasets = random_split(train_datasets, (train_size, val_size))
    print(val_size, len(test_datasets))
    
    train_loader = DataLoader(dataset=train_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, shuffle=True, batch_size=64)
    
    model = model()
    
    criterion = nn.CrossEntropyLoss() # criterion은 손실함수의 판단 기준을 의미함.
    optimizer = torch.optim.Adam(lr=0.0005, params=model.parameters()) # params를 넘겨주는 이유는 순전파 진행 시 생기는 가중치들의 파라미터들의 기울기를 역전파 때 얻고, 업데이트 해주기 위함임.
    losses = []
    
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=2):
        patience_counter = 0
        best_val_loss = float('inf')
        best_model = model.state_dict()
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Train Accuracy
                _, predicted = torch.max(outputs, dim=1) # predicted is index.
                correct_train += (predicted == targets).sum().item()
                total_train += targets.size(dim=0)
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # Validation Accuracy
                    _, predicted = torch.max(outputs, dim=1)
                    correct_val += (predicted == targets).sum().item()
                    total_val += targets.size(dim=0)
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            losses.append(val_loss)
            
            print(f'{epoch+1}th - Train_accuracy : {(correct_train/total_train):.3f}, Train_loss : {train_loss:.3f}, Validation Accuracy : {(correct_val/total_val):.3f}, Validation loss : {val_loss:.3f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                     break
        torch.save(best_model, 'pytorch/MNIST_relu_adam_CE.pth')
    train_model(model, train_loader, val_loader, criterion, optimizer, 20, 2)
    
        
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss in Each Epochs')
    plt.show()