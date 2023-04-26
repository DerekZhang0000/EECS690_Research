import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from digitclassifier import BackpropNet, FFNet, FFNetOvA

device = torch.device("cuda:0")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

def plot_losses(losses):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def train_backpropnet(train_data):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    bn = BackpropNet()
    losses = bn.train(train_loader, epochs=25)
    plot_losses(losses)

def train_ffnet(train_data):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    ffn = FFNet()
    losses = ffn.train(train_loader, epochs=25)
    plot_losses(losses)

def train_ffnet_ova(train_data, train_category):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    ffnova = FFNetOvA(train_category)
    losses = ffnova.train(train_loader, epochs=25)
    # plot_losses(losses)

def display_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    data = MNIST("./data/", download=True, train=True, transform=transform)
    train_data, test_data = torch.utils.data.random_split(data, [50000, 10000])

    # train_backpropnet(train_data)
    # train_ffnet(train_data)
    for category in range(1, 10):
        train_ffnet_ova(train_data, category)

    # model = FFNetOvA()
    # model.load_state_dict(torch.load("ff_model_0.pt"))

    # test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    # model.accuracy(test_loader)

    # for category in range(32):
    #     model.visualize_feature(category)