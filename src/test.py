import torch
import torchvision
from PIL import Image
from torch import nn
# from model import Mff

image_path = '../imgs/dog.png'
image = Image.open(image_path)  # PIL 类型的图片
print(image)  # 打印图像，观察图像类型
"""
因为 png 格式是四个通道，除了 RGB 三个通道外，还有一个透明度通道。所以，要调用 image = image.convert("RGB")，保留其颜色通道。
当然，如果图片本身就是三个颜色的通道，经过此操作，不变。
加上这一步后，可以适应 png，jpg 各种格式的图片。
"""
image = image.convert("RGB")

# 图像大小，只能是模型中的 (32, 32)，然后转换为 tensor 类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])

image = transform(image)  # 应用 transform
print(image)  # 打印图像，观察图像类型
print(image.shape)  # 打印图像大小


class Mff(nn.Module):
    """
    陷阱，如果是自己的模型，就需要在加载模型之前，把模型的 class 重新写一遍，但并不需要实例化，即可
    这个陷阱，也是可以避免的，比如在最上面导入模型：from model import Mff，就是在做这个事情，避免出现错误
    """

    def __init__(self):
        super(Mff, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# mff = torch.load("../checkpoints_dir/mff_20.pth")
# 有的时候如果在gpu上训练的模型，在cpu上加载模型要把它映射到cpu上，使用 map_location=torch.device('cpu')
mff = torch.load("../checkpoints_dir/mff_20.pth", map_location=torch.device('cpu'))
print(mff)

# 由于训练的时候有 batch_size，是 4 维的图像，所以预测的时候也要与训练的时候保持维度一致，因此也需要 reshape 到 4 维
image = torch.reshape(image, (-1, 3, 32, 32))

mff.eval()

with torch.no_grad():  # 推理过程中不需要计算梯度更新参数，节约内存
    output = mff(image)

print(output)
print(output.argmax(1))
