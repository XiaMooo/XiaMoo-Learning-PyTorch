import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def SiLU(x):
    return x * (1 / (1 + np.exp(-x)))


def my_pow(x):
    return x * x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 100)
        self.hidden_layer2 = nn.Linear(100, 100)
        self.hidden_layer3 = nn.Linear(100, 100)
        self.hidden_layer4 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)

    def forward(self, x):
        # x = F.leaky_relu(self.hidden_layer1(x), 0.05)
        # x = F.leaky_relu(self.hidden_layer2(x), 0.05)
        # x = F.leaky_relu(self.hidden_layer3(x), 0.05)
        # x = F.leaky_relu(self.hidden_layer4(x), 0.05)
        x = F.silu(self.hidden_layer1(x))
        x = F.silu(self.hidden_layer2(x))
        x = F.silu(self.hidden_layer3(x))
        x = F.silu(self.hidden_layer4(x))
        predict_y = self.output_layer(x)
        return predict_y


if __name__ == '__main__':
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.00005)
    print(model.parameters())

    criterion = nn.MSELoss()
    epochs = 6000

    x = np.random.randint(-1600, 1600, size=(50, 1)) / 100
    noise = np.random.rand(50, 1)
    y = SiLU(x) + noise * 1

    x_train = x.astype(np.float32)
    y_train = y.astype(np.float32)

    for i in range(epochs):
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)
        outputs = model(inputs)
        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"epoch {i}: loss {loss.data.item():.4f}")

    with torch.no_grad():
        inputs = torch.from_numpy(np.linspace(-16, 16, 1000).reshape(-1, 1).astype(np.float32)).to(device)
        outputs = model(inputs)
        plt.rcParams['xtick.direction'] = 'in'#???x?????????????????????????????????
        plt.rcParams['ytick.direction'] = 'in'#???y??????????????????????????????

        xl = np.linspace(-16, 16, 1000)
        yl = SiLU(xl)
        y_pred = outputs.cpu().numpy()
        plt.plot(xl, yl)
        plt.scatter(x, y, s=2, alpha=0.5, c="g")
        plt.plot(xl, y_pred, alpha=0.5, c="r")
        plt.savefig("out.png", dpi=300)

    model.eval()
    torch.save(model, "linear.pth")
    dummy_input = torch.randn(1, 1, requires_grad=True).to(device)
    torch.onnx.export(model, dummy_input, "linear.onnx", verbose=True, opset_version=13, input_names=["input"],
                      output_names=["output"])
