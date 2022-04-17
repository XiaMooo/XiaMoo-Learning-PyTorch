import torch
from sklearn import datasets, model_selection
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(4, 8)
        self.hidden_layer2 = torch.nn.Linear(8, 15)
        self.hidden_layer3 = torch.nn.Linear(15, 100)
        self.hidden_layer4 = torch.nn.Linear(100, 15)
        self.output_layer = torch.nn.Linear(15, 3)

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
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    x_train, x_test, yi_train, yi_test = model_selection.train_test_split(x, y, test_size=0.2)

    y_train = np.zeros((yi_train.shape[0], 3))
    for i, j in enumerate(yi_train):
        y_train[i, j] = 1.0

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.10)
    criterion = nn.MSELoss()

    epoch = 0
    mse_loss = 1.0
    while mse_loss > 1e-2:
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)
        outputs = model(inputs)
        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        mse_loss = loss.data.item()
        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss {mse_loss:.4f}")
        epoch += 1

    torch.save(model.state_dict(), "MultipleClassifier-Iris.pth")

    with torch.no_grad():
        inputs = torch.from_numpy(x_test).to(device)
        outputs = model(inputs)

        y_pred = [list(i) for i in outputs.cpu().numpy()]
        for i in range(len(y_pred)):
            print(yi_test[i], y_pred[i].index(max(y_pred[i])), y_pred[i])

    model.eval()
    torch.save(model, "MultipleClassifier-Iris.pth")
    dummy_input = torch.randn(1, 10, requires_grad=True).to(device)
    torch.onnx.export(model, dummy_input, "MultipleClassifier-Iris.onnx", verbose=True, opset_version=13, input_names=["input"],
                      output_names=["output"])
