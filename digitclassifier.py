import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from math import e as eulers_num
from random import randint

class BackpropNet(nn.Module):
    def __init__(self):
        super(BackpropNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.R = nn.ReLU()
        self.to("cuda:0")

    def forward(self, x):
        # Flatten input
        x = x.view(x.shape[0], -1)
        x = F.normalize(x, p=2, dim=1)

        # Forward pass
        x = self.R(self.fc1(x))
        x = self.R(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def backward(self, x, y, learning_rate):
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(x, y)
        self.zero_grad()

        # Backwards pass
        loss.backward()

        # Update weights and biases
        with torch.no_grad():
            for param in self.parameters():
                param.data -= learning_rate * param.grad
        self.zero_grad()

        return loss.item()

    def train(self, train_loader, learning_rate=0.01, epochs=10):
        losses = []
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                predictions = self.forward(images)
                cumulative_loss += self.backward(predictions, labels, learning_rate)

            loss = cumulative_loss / len(train_loader)
            losses.append(loss)
            print(f"Epoch {epoch} Loss: {loss}")

        torch.save(self.state_dict(), "bp_model.pt")
        return losses

    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                predictions = self.forward(images)
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {correct / total * 100}%")
        return correct / total

    def visualize_feature(self, node_index):
        self.to("cpu")

        weight = self.fc1.weight.data[node_index]
        output_grid = weight.view(28, 28)
        output_grid -= output_grid.min()
        output_grid /= output_grid.max()

        plt.imshow(output_grid, cmap="gray")
        plt.title(f"Node {node_index} Feature Visualization")
        plt.axis("off")
        plt.show()

class FFNet(nn.Module):
    class FFLayer(nn.Linear):
        def __init__(self, inputs, outputs, threshold):
            super(FFNet.FFLayer, self).__init__(inputs, outputs)
            self.R = nn.ReLU(inplace=False)
            self.threshold = threshold

        def forward(self, x):
            # Flatten input
            x = x.view(x.shape[0], -1)

            x_norm = F.normalize(x, p=2, dim=1)
            x_transform = torch.matmul(x_norm, self.weight.t()) + self.bias.view(1, -1).t()
            x_relu = self.R(x_transform)
            return x_relu

        def train(self, positive_data, negative_data, learning_rate):
            def goodness(x):
                return torch.sum(x ** 2)

            layer_loss = 0

            # Positive pass
            x = self.forward(positive_data)
            loss = torch.log(1 + torch.exp(goodness(x) - self.threshold))
            layer_loss += loss.item()

            # backward() does not want to work, so we calculate gradients manually here
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.parameters(), grads):
                    param.grad = -1 * grad * learning_rate
            self.zero_grad()

            # Negative pass
            x = self.forward(negative_data)
            loss = torch.log(1 + torch.exp(self.threshold - goodness(x)))
            layer_loss += loss.item()

            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.parameters(), grads):
                    param.grad = -1 * grad * learning_rate
            self.zero_grad()

            return layer_loss / 2

    def __init__(self):
        super(FFNet, self).__init__()
        self.fc1 = self.FFLayer(28 * 28, 64, threshold=2)  # Thresholds are arbitrary
        self.fc2 = self.FFLayer(64, 64, threshold=2)
        self.fc3 = self.FFLayer(64, 10, threshold=2)
        self.layers = (self.fc1, self.fc2, self.fc3)
        self.to("cuda:0")

    def encode_label(self, image, label):
        # Encodes the image label into the image
        for i in range(10):
            if i == label:
                image[0][0][label] = 1
            else:
                image[0][0][label] = -1

        return image

    def train(self, train_loader, learning_rate=0.01, epochs=10):
        losses = []
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                positive_images, positive_labels = images[:len(images) // 2], labels[:len(labels) // 2]
                positive_images = torch.stack([self.encode_label(image, label) for image, label in zip(positive_images, positive_labels)])
                negative_images, negative_labels = images[len(images) // 2:], labels[len(labels) // 2:]
                negative_images = torch.stack([self.encode_label(image, (label + randint(1, 9)) % 10) for image, label in zip(negative_images, negative_labels)]) # Misencodes the label for negative data

                for image_pair in zip(positive_images, negative_images):
                    temp_loss = 0
                    for layer in self.layers:
                        temp_loss += layer.train(image_pair[0], image_pair[1], learning_rate=learning_rate)
                    cumulative_loss += temp_loss / len(self.layers)

            loss = cumulative_loss / len(train_loader)
            losses.append(loss)
            print(f"Epoch {epoch} Loss: {loss}")

        torch.save(self.state_dict(), "ff_model.pt")
        return losses

    def visualize_feature(self, node_index):
        self.to("cpu")

        weight = self.fc1.weight.data[node_index]
        output_grid = weight.view(28, 28)
        output_grid -= output_grid.min()
        output_grid /= output_grid.max()

        plt.imshow(output_grid, cmap="gray")
        plt.title(f"Node {node_index} Feature Visualization")
        plt.axis("off")
        plt.show()
