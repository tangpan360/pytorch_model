import glob

import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class MyDataset(Dataset):

    def __init__(self, all_imgs_path_list, all_labels_path_list, transform):
        """对数据集进行初始化设置

        :param all_imgs_path_list: 所有图片路径组成的列表
        :param all_labels_path_list: 所有标签文件路径组成的列表
        :param transform: 图片的格式转换（可以尺寸变换，转换成 tensor 等）
        """
        self.all_imgs_path_list = all_imgs_path_list  # 所有图片列表
        self.all_labels_path_list = all_labels_path_list  # 所有标签路径列表
        self.transforms = transform  # 图片变换方式

    def __getitem__(self, idx):  # 依次获取图像及其对应标签
        img = self.all_imgs_path_list[idx]
        img = Image.open(img)
        img = img.convert("RGB")
        img = self.transforms(img)

        label = self.all_labels_path_list[idx]
        label = int(label)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.all_imgs_path_list)  # 返回数据集长度


def split_train_test(all_imgs_path: str = '../dataset/original_imgs',
                     all_labels_path: str = '../dataset/labels',
                     train_size: float = 0.8):
    """用来划分训练集和测试集。该函数是对 Dataset 类的进一步封装调用（因为 Dataset 类没有划分训练集和测试集），
    该函数可直接使用，无需再单独调用 Dataset 类，已经在该函数内部调用。

    图片存储形式：
                -| 001.jpg
                -| 002.jpg
    标签存储形式：
                -| 001.txt (文本内标签内容：0)
                -| 002.txt (文本内标签内容；1)

    示例：train_ds, test_ds = split_train_test()

    :param all_imgs_path: 所有图片所在的的存放路径（不包含图片）
    :param all_labels_path: 所有标签文本所在的的存放路径（不包含标签文本）
    :param train_size: 训练集所占的比例
    :return: 分别返回 训练集和测试集，需两个参数来接收
    """

    all_imgs_path_list = glob.glob(all_imgs_path + '/*')  # 获取所有图片的位置列表

    all_labels_path_list = glob.glob(all_labels_path + '/*')  # 获取所有标签文件的位置列表

    # 获取所有标签文件里的具体内容组成列表
    all_labels = []
    for label_path in all_labels_path_list:
        with open(label_path) as f:
            label = f.readline()
            all_labels.append(label)

    # 图片处理方式
    transform = transforms.Compose([transforms.Resize((128, 128)),  # resize 图片尺寸
                                    transforms.ToTensor()])  # 将图片转成 tensor

    index = np.random.permutation((len(all_imgs_path_list)))  # 获取全部图片的索引，并重新打乱顺序
    print(index)

    all_imgs_path = np.array(all_imgs_path_list)[index]  # 打乱顺序后的所有图片位置列表
    print("bbbb", type(all_imgs_path))
    print(all_imgs_path)
    all_labels = np.array(all_labels)[index]  # 打乱顺序后的所有标签列表（由于用的索引一致为 index，所以与图片相对应）
    print("aaaa", type(all_labels))
    print(all_labels)

    # 80% 作为训练集
    s = int(len(all_imgs_path) * train_size)

    train_imgs = all_imgs_path[:s]  # 作为训练集的所有图片
    train_labels = all_labels[:s]  # 作为训练集的所有标签
    test_imgs = all_imgs_path[s:]  # 作为测试集的所有图片
    test_labels = all_labels[s:]  # 作为测试集的所有标签

    train_ds = MyDataset(train_imgs, train_labels, transform)  # 利用 Dataset 类构建训练集
    test_ds = MyDataset(test_imgs, test_labels, transform)  # 利用 Dataset 类构建测试集

    return train_ds, test_ds


if __name__ == "__main__":

    train_ds, test_ds = split_train_test()

    print(len(train_ds))
    print(len(test_ds))

    train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    print(next(iter(train_dataloader)))
    test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=True, drop_last=True)



