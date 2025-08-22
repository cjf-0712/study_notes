from torchvision import transforms
from torchvision.datasets import FashionMNIST
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#数据集准备
train_data = FashionMNIST(
    root = "./data/FashionMNIST",
    train = True,
    transform = transforms.ToTensor(),
    download = True #如果没下载数据，就下载数据；如果已经下载好，就换为False
)
test_data = FashionMNIST(
    root = "./data/FashionMNIST",
    train = False,
    transform = transforms.ToTensor(),
    download = True #如果没下载数据，就下载数据；如果已经下载好，就换为False
)

train_data_x=train_data.data
train_data_y=train_data.targets
test_data_x=test_data.data
test_data_y=test_data.targets

print(train_data_x.shape)
print(train_data_y.shape)
print(type(train_data_x))
print(type(train_data_y))
print(test_data_x.shape)
print(test_data_y.shape)

def function1(x):
    for i in range(x):
        print(i)
function1(7)
