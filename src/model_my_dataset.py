from torch import nn


# 创建网络模型
class Mff(nn.Module):

    def __init__(self):
        super(Mff, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.model(x)
        return x
