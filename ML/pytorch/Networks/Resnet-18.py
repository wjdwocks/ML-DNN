import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 1번째 Conv Layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 여기서 만약 2Stage 이상의 첫 번째 블록이었다면 stride = 2였을 것임.
        # 두 번재 블록이었다면 stride는 기본값인 1이었을 것이고.
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 2번째 Conv Layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 입력과 출력 채널이 다를 경우 차원을 조정해줌.
        # stride != 1 의 의미 : Stage 2, 3, 4의 첫 번째 블록에서는 stride=2로 설정되므로, 
        # 첫 번째 Residual Block에서 stride = 2인 경우 다운샘플링이 필요하기에 
        # shortcut 경로에서도 같은 stride를 사용하여 출력과 입력의 해상도를 일치시킴

        # in_channels != out_channels 의 의미
        # 각 Stage의 첫 번째 Residual Block은 이전 Stage와 채널 수가 다름.
        # ex) Stage 1에서 out_channels = 64로 끝나면 Stage 2에서는 in_channels = 128 로 시작해야 함.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels: 
            self.shortcut = nn.Sequential( # 그렇기 때문에 입력 채널 수를 kernel_size = 1로 하여 채널 수만 2배로 늘려준다.
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # 즉, 각 Stage의 첫 번재 블록에서 stride = 2가 되고, 각 특성맵의 크기는 반으로 줄어드는 대신, 채널의 깊이가 2배가 된다.

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x))) # 각 블록의 첫 번째 Conv층
        out = self.bn2(self.conv2(out)) # 두 번째 Conv층
        out += self.shortcut(x) # 단차 더해줌 여기서 Conv를 겪으며 바뀐 특성맵의 크기(해상도) out과 x가 다르면 안되기 때문에 shortcut(x)로 그 특성맵의 크기를 맞춰준다.
        # 여기서 각 Stage의 첫 번째 블록에서는 특성맵의 크기가 다르기 때문에 out += shortcut(x) 가 되고,
        # 두 번째 블록에서는 out += x가 되는 것이다.
        out = nn.functional.relu(out) # 활성화 함수 적용
        return out # 반환
    


class Resnet18(nn.Module):
    def __init__(self, num_classes = 10):
        super(Resnet18, self).__init__()
        # 초기 Conv 레이어 
        # 4개의 Stage를 지나야 하기 때문에 첫 번째 Conv에서는 해상도를 유지하도록 함.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        # maxPooling도 해상도가 반으로 줄어드니 사용하지 않음.
        self.stage1 = self._make_stage(64, 64, num_blocks = 2, stride=1) 
        self.stage2 = self._make_stage(64, 128, num_blocks = 2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks = 2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks = 2, stride=2)

        # Adaptive Pooling과 FC Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 출력되는 특성 맵의 목표 크기를 tuple로 넘겨줌.
        self.fc = nn.Linear(512, num_classes)

    # 각 스테이지를 꾸리는 함수
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 초기 Conv층, BN, ReLU 지나는데 
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        # Pooling은 제외 cifar10이 32x32 크기라 너무 작아짐.

        # 각 Residual Stage를 건넘
        out = self.stage1(out) # 32, 32
        out = self.stage2(out) # 16, 16
        out = self.stage3(out) # 8, 8
        out = self.stage4(out) # 4, 4

        # 
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = transforms([
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
    
    train_datasets, val_datasets = torch.utils.data.random_split(train_datasets, (train_length, val_length))
    
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=64, shuffle=True)
    
    model = Resnet18().to(device)  # 모델을 GPU로 이동
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.001)
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
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
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
                    inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, dim=1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(dim=0)
                    
            val_accuracy = (val_correct / val_total)
            val_loss /= len(val_loader)
            losses.append(val_loss)
            accuracys.append(val_accuracy)
            print(f'{epoch+1}s : train_accuracy - {train_accuracy:.3f} train_loss - {train_loss:.3f}, val_accuracy - {val_accuracy:.3f}, val_loss - {val_loss:.3f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_model = model.state_dict()
            else:
                patience += 1
                if patience >= max_patience:
                    break
        
        torch.save(best_model, 'ML/pytorch/Cifar-10_Adagrad.pth')

        train_model(model, train_loader, test_loader, val_loader, optimizer, criterion, epochs=20, max_patience=2)

    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(dim=0)
            
    print(f'Test Set의 Accuracy : {(test_correct/test_total):.3f}, Test Set의 loss : {(test_loss / len(test_loader)):.3f}')

    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(range(1, len(losses)+1), losses)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('Validation Loss')
    ax[0].set_title('Cifar-10 Adagrad lr=0.001')
    ax[1].plot(range(1, len(losses)+1), accuracys)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('Validation Acc')
    ax[1].set_title('Cifar-10 Adagrad lr=0.001')
    plt.show()