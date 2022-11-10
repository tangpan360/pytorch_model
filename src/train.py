import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Mff


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../dataset',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../dataset',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获取数据集的的长度并打印提示
train_data_length = len(train_data)
test_data_length = len(test_data)
print(f"训练集的长度为：{train_data_length}")
print(f"测试集的长度为：{test_data_length}")

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 网络模型实例化
mff = Mff()
mff = mff.to(device)  # 送入 gpu 训练

# 创建损失函数
loss_cross = nn.CrossEntropyLoss()
loss_cross = loss_cross.to(device)

# 定义优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(mff.parameters(), lr=learning_rate)

# 设置训练当中的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
epoch = 20  # 训练的轮数

# 添加 tensorboard
writer = SummaryWriter(log_dir='../logs')

# for 循环 进行训练和测试
for i in range(epoch):
    print(f"-----第 {i + 1} 轮（epoch）训练开始-----")

    # 训练步骤开始
    mff.train()  # model.train() 只对一部分层起作用，比如 dropout 层；如果有这些特殊的层才需要调用这个语句
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = mff(imgs)
        loss = loss_cross(outputs, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1  # 此处是每个 batch_size 为一步

        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}, loss: {loss.item()}")  # loss.item() 是吧 tensor 数据转化为真实的数字
            writer.add_scalar("train_loss", scalar_value=loss.item(), global_step=total_train_step)

    # 测试步骤开始
    mff.eval()  # model.eval() 只对一部分层起作用，比如 dropout 层；如果有这些特殊的层才需要调用这个语句
    total_test_loss = 0
    total_accuracy_num = 0
    with torch.no_grad():  # 测试过程中不需要计算梯度更新参数，节约内存
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = mff(imgs)
            loss = loss_cross(outputs, targets)
            total_test_loss += loss.item()

            accuracy_num = (outputs.argmax(1) == targets).sum()
            total_accuracy_num += accuracy_num

    total_test_step += 1  # 此处是每个 epoch 为一步

    print(f"整体测试集上的 Loss：{total_test_loss}")
    print(f"整体测试集上的正确率 Accuracy：{total_accuracy_num / train_data_length}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy_num, total_test_step)

    torch.save(mff, f"../checkpoints_dir/mff_{i + 1}.pth")  # 需要自己创建 checkpoints_dir 文件夹
    print("模型已保存")

writer.close()
