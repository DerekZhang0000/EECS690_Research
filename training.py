import torch
import time
import hiddenlayer as hl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from digitclassifier import BackpropNet, FFNet

device = torch.device("cuda:0")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

def record_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}s")
        return result

    return wrapper

def plot_losses(losses):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def train_backpropnet(train_data):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    bn = BackpropNet()
    losses = record_time(bn.train(train_loader, learning_rate=0.01, epochs=25))
    plot_losses(losses)

def train_ffnet(train_data):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    ff = FFNet()
    losses = record_time(ff.train(train_loader, learning_rate=0.01, epochs=25))
    plot_losses(losses)

if __name__ == "__main__":
    data = MNIST("./data/", download=True, train=True, transform=transform)
    train_data, test_data = torch.utils.data.random_split(data, [50000, 10000])

    # train_backpropnet(train_data)
    train_ffnet(train_data)
    # model = BackpropNet()
    # model = FFNet()

    # model.load_state_dict(torch.load("ff_model.pt"))
    # test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    # model.accuracy(test_loader)
    # for i in range(64):
    #     model.visualize_feature(i)