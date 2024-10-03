import torch.nn as nn
model = nn.Sequential(
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128, 10)
)