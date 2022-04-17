import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
epochs = 50


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 5)),
            nn.BatchNorm2d(10),
            nn.Sigmoid()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(10),
            nn.Sigmoid()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Sequential(
            nn.Linear(10 * 12 * 12, 500),
            nn.BatchNorm1d(500),
            nn.Sigmoid()
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(500, 12),
            nn.BatchNorm1d(12),
            nn.Sigmoid()
        )
        self.linear_3 = nn.Sequential(
            nn.Linear(12, 10),
            nn.BatchNorm1d(10),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        conv_layer_1 = self.conv_1(x)
        pool_2_2 = self.max_pool(conv_layer_1)
        conv_layer_2 = self.conv_2(pool_2_2)
        flatten_layer = conv_layer_2.reshape(x.size(0), -1)
        linear_layer_1 = self.linear_1(flatten_layer)
        linear_layer_2 = self.linear_2(linear_layer_1)
        linear_layer_3 = self.linear_3(linear_layer_2)
        output_layer = self.output(linear_layer_3)

        return output_layer


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True
    )

    model = ConvNet().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.05)
    optimizer = optim.Adam(model.parameters())
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # criterion = F.nll_loss

    train_loss, test_loss = float("inf"), float("inf")
    epoch = 0
    while epoch < epochs:
        # while (train_loss > 1e-2 or test_loss > 1e-2) and epoch < epochs:
        epoch += 1
        # Train
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = np.zeros((targets.shape[0], 10), dtype=np.float32)
            for index, target in enumerate(targets):
                labels[index, target] = 1.0
            labels = torch.from_numpy(labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if (i + 1) % (10000 // batch_size) == 0:
                print(
                    f'Train {epoch}'
                    f'[{100 * i / len(train_loader):.0f}%]: '
                    f' loss:{train_loss:.6f}')
        print(
            f'Train {epoch}'
            f'[100%]: '
            f' loss:{train_loss:.6f}')

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                labels = np.zeros((targets.shape[0], 10), dtype=np.float32)
                for index, target in enumerate(targets):
                    labels[index, target] = 1.0
                labels = torch.from_numpy(labels).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                # test_loss += F.nll_loss(outputs, targets, reduction='sum').item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()

        # test_loss /= len(test_loader.dataset)
        print(f'Test {epoch}: Avg_loss:{test_loss:.4f} '
              f'Acc:{correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset):.2f})'
              f'\n')

    model.eval()
    model.to(device)
    torch.save(model, "MNIST.pth")
    dummy_input = torch.zeros(1, 1, 28, 28, requires_grad=True).to(device)
    torch.onnx.export(model, dummy_input, "MNIST.onnx", verbose=True, opset_version=13, input_names=["input"],
                      output_names=["output"])
