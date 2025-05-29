import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_blk = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_blk(x)
        p = self.pool(x)
        return x, p # x는 Encoder Block에서 나온 Featrue Map (Skip Connection 용도) Deocder에 넣을 Skip Connection용도
                    # p는 Pooling 결과 (다음 블록 입력용, Downsample된 Feature)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_blk = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Padding if necessary
        if x.size() != skip.size():
            x = F.pad(x, [0, skip.size(3) - x.size(3), 0, skip.size(2) - x.size(2)])
        x = torch.cat([x, skip], dim=1)
        return self.conv_blk(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.b = ConvBlock(512, 1024)
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_act = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1) # 이런식으로 P만 다음 Encoder로 넣음.
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3) # 점점 작게 만들다가
        b = self.b(p4) # 전에 봤던것처럼 BottleNeck으로 해상도는 유지하고, 채널 수를 증가시킴.
        d1 = self.d1(b, s4) # 다시 ConvTranspose를 하면서 해상도를 올림. + Encoder에서 출력한 x값으로 Skip Connection.
        d2 = self.d2(d1, s3) 
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1) 
        out = self.outc(d4) # 최종 출력에 맞게 channel을 맞춰줌.
        return self.final_act(out) # 최종적으로 나온 이미지를 보고 원하는 Task에 적용