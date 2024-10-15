import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 32, kernel_size=3, padding='same')
        # 7 x 7 x 32의 특성맵 크기.
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 32, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        # self.softmax = nn.Softmax(dim=1) # dim=1은 axis를 말하며, 1번 차원에 대해 소프트맥스를 사용, 각 클래스의 확률들에
    def forward(self, x):
        # print(x.shape) # (64, 1, 28, 28)
        # x = self.pooling(nn.functional.relu(self.conv1(x))) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        # print(x.shape) # (64, 10, 14, 14)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        # print(x.shape) # (64, 32, 7, 7)
        x = self.flatten(x)
        # print(x.shape) # (64, 1568)
        x = self.fc(x)
        x = self.relu(x)
        x = nn.functional.dropout(x, 0.2)
        # print(x.shape) # (64, 100)
        x = self.fc2(x)
        # print(x.shape) # (64, 10)
        # x = nn.functional.softmax(x)
        return x
    
    
if __name__ == '__main__':
    # 데이터 전처리?
    transform = transforms.Compose([ # 파이프 라인을 만듦.
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Fashion_Mnist 훈련 및 테스트 데이터셋 불러오기.
    train_dataset = torchvision.datasets.FashionMNIST( # FashionMNIST를 불러올거
        root='./fashionMNIST', # 저장 위치 : ./data (여기 폴더의 data라는 파일로)
        train=True, # True면 Train Set으로 False면 Test Set임.
        download=True,  # 다운로드 받을건지, 이미 있으면 False로 하자.
        transform=transform # 데이터 선처리
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./fashionMNIST',
        train=False,
        download=False,
        transform=transform
    )
    
    # 검증 세트 분리
    train_size = int(0.8 * len(train_dataset))
    val_size = int(len(train_dataset) - train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 모델 생성
    model = myModel()
    
    ## result
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs = 50, patience = 2):
        # best_model_state = None # None으로 놓아도 되는데 공부중이니까 아레와 같이 놓자
        best_model_state = model.state_dict()
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs): # 각 에포크마다 아레를 반복.
            model.train() # 모델을 훈련 모드로 설정
            ## model.eval() ## 평가 -> 자동으로 가중치 고정
            train_loss = 0 # 초기 훈련 손실값 0
            correct_train = 0
            total_train = 0
            correct_val = 0
            total_val = 0
            for inputs, targets in train_loader: # 각 훈련 세트를 입력과 출력으로 나눔. (64, 1, 28, 28)
                optimizer.zero_grad() # 이번 배치의 각 파라미터의 가중치를 초기화함.?
                outputs = model(inputs) ## 순전파 (이번 배치 세트의 학습을 시작함.)
                loss = criterion(outputs, targets) ## 손실 계산 +, * , / , softmax 
                loss.backward()  ## 역전파, 각 파라미터의 기울기 정보 계산
                optimizer.step() ## loss.backward()에서 계산한 기울기로 파라미터를 업데이트(가중치 변경)
                train_loss += loss.item()
                # 정확도 계산하기 위한 식
                _, predicted = torch.max(outputs, dim=1)    # torch.max는 텐서로 온 값 (여기선 64 x 10, 왜냐면 softmax를 통과하면 64개(배치)데이터의 (10개 클래스의 확률)이 나옴.)
                                                            # 그런데 dim=1로 줘서 두번 째 차원의 값이 나옴. (각 확률 중 최댓값), _에 그 확률, predicted에 그 index가 들어감.
                correct_train += (predicted == targets).sum().item() # predicted : (64, ), targets : (64, ) 의 텐서라서 각각의 값이 같으면 True(1)로 다 더하고, .item()으로 정수로 바꿈.
                total_train += targets.size(dim=0) # targets는 (64, )의 크기라 dim=0으로 해서 배치의 사이즈 만큼 전체 데이터 개수를 추적.
                
            
            model.eval()
            val_loss =  0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # 정확도
                    _, predicted = torch.max(outputs, dim=1)
                    correct_val = correct_val + (predicted == targets).sum().item()
                    total_val += targets.size(dim=0)
                    
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {(correct_train/total_train):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {(correct_val/total_val):.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict() # model.state_dict()가 현재 모델의 각 파라미터의 가중치를 저장하고 있는 것을 의미함.
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            
        torch.save(best_model_state, 'pytorch/MNIST_best_model_RMSprop.pth') ## 이름 겹치지 않게 하기 덮어씌워지면 이제 지옥
        print('Model training completed and saved.')

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
    
    test_correct = 0
    test_total = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(dim=0)
            
    print(f'Test Acc = {(test_correct/test_total):.3f}, Test loss = {(test_loss / len(test_loader)):.3f}') # Test Acc = 0.858, Test loss = 0.420
        
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss in Each Epochs')
    plt.show()