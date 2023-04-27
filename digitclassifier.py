import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from random import randint

class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.R = nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
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

    def backward(self, x, y):
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(x, y)

        # Backwards pass
        self.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def train(self, train_loader, epochs=10):
        losses = []
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                predictions = self.forward(images)
                cumulative_loss += self.backward(predictions, labels)

            loss = cumulative_loss / len(train_loader)
            losses.append(loss)
            print(f"Epoch {epoch} Loss: {loss}")

        torch.save(self.state_dict(), "./Backprop Models/bp_model.pt")
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

class BPNetOvA(BPNet):
    def __init__(self, train_category=0):
        super(BPNetOvA, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)
        self.train_category = train_category
        self.to("cuda:0")

    def train(self, train_loader, epochs=10):
        losses = []
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")

                # Set target labels to 1 for the target class and 0 for all other classes
                labels = (labels == self.train_category).long()

                while len(labels[labels == 1]) > len(labels[labels == 0]):  # Removes non-target labels until the number of targets and non-targets are equal
                    index = randint(0, len(labels) - 1)
                    if labels[index] == 0:
                        labels = torch.cat((labels[:index], labels[index + 1:]))

                predictions = self.forward(images)
                cumulative_loss += self.backward(predictions, labels)

            loss = cumulative_loss / len(train_loader)
            losses.append(loss)
            print(f"Epoch {epoch} Loss: {loss}")

        torch.save(self.state_dict(), f"./Backprop Models/bp_model_{self.train_category}.pt")
        return losses

    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                labels[labels == self.train_category] = -1
                labels[labels != -1] = 0
                labels[labels == -1] = 1

                for image, label in zip(images, labels):
                    prediction = torch.argmax(self.forward(image))
                    if prediction == label:
                        correct += 1

                total += len(labels)
        print(f"Accuracy: {correct / total * 100}%")
        return correct / total

class FFNet(nn.Module):
    class FFLayer(nn.Linear):
        def __init__(self, inputs, outputs, threshold):
            super(FFNet.FFLayer, self).__init__(inputs, outputs)
            self.R = nn.ReLU()
            self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
            self.threshold = threshold
            self.to("cuda:0")

        def forward(self, x):
            # Flatten input
            x = x.view(x.shape[0], -1)

            # Normalize and transform input
            x_direction = x / (x.norm(2, 1) + 1e-4).unsqueeze(1)
            x_transform = torch.matmul(x_direction, self.weight.T) + self.bias.unsqueeze(0)

            # Forward pass
            x_relu = self.R(x_transform)
            return x_relu

        def train(self, x_positive, x_negative):
            def goodness(x):
                return x.pow(2).mean(1)

            loss = torch.log(1 + torch.exp(torch.cat([
                -goodness(self.forward(x_positive)) + self.threshold,
                goodness(self.forward(x_negative)) - self.threshold
            ]))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            return self.forward(x_positive).detach(), self.forward(x_negative).detach(), loss.item()

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
                image[0][0][i] = 1
            else:
                image[0][0][i] = -1

        return image

    def train(self, train_loader, epochs=10):
        losses = []
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                positive_images = torch.stack([self.encode_label(image, label) for image, label in zip(images, labels)])
                negative_images = torch.stack([self.encode_label(image, (label + randint(1, 9)) % 10) for image, label in zip(images, labels)]) # Misencodes the label for negative data

                for image_pair in zip(positive_images, negative_images):
                    x_positive, x_negative = image_pair
                    for layer in self.layers:
                        x_positive, x_negative, temp_loss = layer.train(x_positive, x_negative)
                        cumulative_loss += temp_loss

            loss = cumulative_loss / (len(train_loader) * len(self.layers))
            losses.append(loss)
            print(f"Epoch {epoch} Loss: {loss}")

        torch.save(self.state_dict(), "./FF Models/ff_model.pt")
        return losses

    def predict(self, x):
        with torch.no_grad():
            def goodness(x):
                return x.pow(2).mean(1)

            x_possibilities = [self.encode_label(x, i).clone() for i in range(10)]  # Creates 10 copies of the image with different labels
            most_good_index = 0
            most_goodness = 0
            for index, x in enumerate(x_possibilities):
                x_goodness = 0
                for layer in self.layers:
                    x = layer.forward(x)
                    x_goodness += torch.sum(goodness(x)).item()
                if x_goodness > most_goodness:
                    most_goodness = x_goodness
                    most_good_index = index

            return most_good_index

    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                for image, label in zip(images, labels):
                    prediction = self.predict(image)
                    total += 1
                    if prediction == label:
                        correct += 1

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

class FFNetOvA(FFNet):
    class FFLayerOvA(FFNet.FFLayer):
        pass

    def __init__(self, train_category=0):
        super(FFNetOvA, self).__init__()
        self.fc1 = self.FFLayerOvA(28 * 28, 32, threshold=2)    # Thresholds are arbitrary
        self.fc2 = self.FFLayerOvA(32, 32, threshold=2)
        self.fc3 = self.FFLayerOvA(32, 2, threshold=2)
        self.layers = (self.fc1, self.fc2, self.fc3)
        self.train_category = train_category
        self.to("cuda:0")

    def train(self, train_loader, epochs=10):
        losses = []
        for epoch in range(epochs):
            cumulative_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                positive_images = images[labels == self.train_category]
                negative_images = images[labels != self.train_category]
                negative_images = negative_images[:len(positive_images)]  # Makes sure that the number of positive and negative images are equal

                for image_pair in zip(positive_images, negative_images):
                    x_positive, x_negative = image_pair
                    for layer in self.layers:
                        x_positive, x_negative, temp_loss = layer.train(x_positive, x_negative)
                        cumulative_loss += temp_loss

            loss = cumulative_loss / (len(train_loader) * len(self.layers))
            losses.append(loss)
            print(f"Epoch {epoch} Loss: {loss}")

        torch.save(self.state_dict(), f"./FF Models/ff_model_{self.train_category}.pt")
        return losses

    def predict(self, x):
        def goodness(x):
            return x.pow(2).mean(1)

        total_goodness = 0
        for layer in self.layers:
            x = layer.forward(x)
            total_goodness += torch.sum(goodness(x)).item()

        return self.train_category if total_goodness >= sum([layer.threshold for layer in self.layers]) else -1

    def accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
                for image, label in zip(images, labels):
                    prediction = self.predict(image)
                    if label == self.train_category and prediction == self.train_category:
                        correct += 1
                    elif label != self.train_category and prediction == -1:
                        correct += 1
                total += labels.size(0)

        print(f"Accuracy: {correct / total * 100}%")
        return correct / total