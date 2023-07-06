import torch
from torch import nn
import math

# 处理图像，残差神经网络
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        y = torch.nn.functional.relu(self.conv(x) + x)
        return y

# 搭建ResNet
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self. conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7), # 256*3*256*256 256*16*250*250
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 256*16*250*250 256*16*125*125
        )
        self.res_block1 = ResidualBlock(16)
        self. conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6), # 256*16*125*125 256*32*120*120
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 256*32*120*120 256*32*60*60
        )
        self.res_block2 = ResidualBlock(32)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(32*60*60, 64)
    # 正向传播过程
    def forward(self, x):
        y = self.conv1(x) # 256*3*32*32 256*16*125*125
        # print(y.shape)
        y = self.res_block1(y)
        # print(y.shape)
        y = self.conv2(y) # 256*16*125*125 256*32*60*60
        y = self.res_block2(y)
        output = self.linear(self.dropout(y.view(x.shape[0], -1)))
        return output # 256*64


        
# 处理文本

class TextModel(nn.Module):
    def __init__(self, max_len):
        super(TextModel, self).__init__()
        # print(max_len)
        self.embedding = nn.Embedding(max_len, 128, 0)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(130 * 128, 64)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x[0])
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 维度转换，将batch维放在第1维度
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 维度转换，将batch维放回第0维度
        # print(x.shape)
        # print('`````')
        # x = x.mean(dim=1)  # 池化操作，取平均值
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.linear(self.dropout(x))
        # x = self.softmax(x)
        # print(x[0])
        return x


# 表征合并，多模态融合
class DualStreamModel(nn.Module):
    def __init__(self, max_len, mode = 'all'):
        super(DualStreamModel, self).__init__()
        self.mode = mode
        self.hidden_size = 64
        if mode == 'all':
            self.hidden_size = 128
        self.image_model = ImageModel()
        self.text_model = TextModel(max_len)
        self.linear = nn.Linear(self.hidden_size, 3)
        

    def forward(self, x_img, x_txt):
        # print('```')
        if self.mode == 'img':
            image_output = self.image_model(x_img)
            combined_output = image_output
        elif self.mode == 'txt':
            text_output = self.text_model(x_txt)
            combined_output = text_output
        else:
            image_output = self.image_model(x_img)
            text_output = self.text_model(x_txt)
            # print(image_output.shape)
            # print(text_output.shape)
            combined_output = torch.cat((image_output, text_output), dim=1)
        output = self.linear(combined_output)
        
        return output