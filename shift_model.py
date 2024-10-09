import torch.nn as nn
import torch.nn.functional as F

#artifacts learning
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        # 将压缩层合并到 features 中
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(1, 5), stride=(1, 5), padding=0),  # 压缩宽度从50到10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 9, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, 7, stride=(1, 4), padding=(3, 0))
        )
    def forward(self, x):
        x = self.features(x)
        return x
