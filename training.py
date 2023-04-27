import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from digitclassifier import BPNet, BPNetOvA, FFNet, FFNetOvA

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
    bn = BPNet()
    losses = bn.train(train_loader, epochs=25)
    plot_losses(losses)

def train_bpnet_ova(train_data, train_category):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    bnova = BPNetOvA(train_category)
    losses = bnova.train(train_loader, epochs=25)
    plot_losses(losses)

def bp_accuracy_ova():
    models = [BPNetOvA(index) for index in range(10)]
    for index in range(10):
        models[index].load_state_dict(torch.load(f"./Backprop Models/bp_model_{index}.pt"))
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    for batch in test_loader:
        batch_predictions = []
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        for image in images:
            temp_predictions = []
            for model in models:
                temp_predictions.append(torch.argmax(model.forward(image)).item())
            if temp_predictions.count(1) != 1:
                batch_predictions.append(-1)
            else:
                batch_predictions.append(temp_predictions.index(1))
        total += labels.size(0)

        for prediction, label in zip(batch_predictions, labels):
            if prediction == label:
                correct += 1
    print(f"Accuracy: {correct / total}")

def train_ffnet(train_data):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    ffn = FFNet()
    losses = ffn.train(train_loader, epochs=25)
    plot_losses(losses)

def train_ffnet_ova(train_data, train_category):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    ffnova = FFNetOvA(train_category)
    losses = ffnova.train(train_loader, epochs=25)
    plot_losses(losses)

def ff_accuracy_ova():
    models = [FFNetOvA(index) for index in range(10)]
    for index in range(10):
        models[index].load_state_dict(torch.load(f"./FF Models/ff_model_{index}.pt"))
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    for batch in test_loader:
        batch_predictions = []
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        for image in images:
            temp_predictions = []
            for model in models:
                temp_predictions.append(model.predict(image))
            if temp_predictions.count(-1) != 9:
                batch_predictions.append(-1)
            else:
                batch_predictions.append([num for num in temp_predictions if num != -1][0])
        total += labels.size(0)

        for prediction, label in zip(batch_predictions, labels):
            if prediction == label:
                correct += 1
    print(f"Accuracy: {correct / total}")

def display_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    data = MNIST("./data/", download=True, train=True, transform=transform)
    train_data, test_data = torch.utils.data.random_split(data, [50000, 10000])

    bp_accuracy_ova()
    ff_accuracy_ova()

    # for category in range(64):
    #     model.visualize_feature(category)