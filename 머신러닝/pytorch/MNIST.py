import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, ramdom_split

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 30)
        # pooling
        # relu
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28*30, 100)
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
        print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
if __name__ == '__main__':
    pipeline = nn.Compo