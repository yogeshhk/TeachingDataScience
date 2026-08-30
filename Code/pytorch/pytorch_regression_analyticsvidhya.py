# https://medium.com/analytics-vidhya/implement-linear-regression-on-boston-housing-dataset-by-pytorch-c5d29546f938

from sklearn.datasets import fetch_openml  # load_boston has ethical issues
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

np.random.seed(12)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(size_average=False)


def get_data():
    data = fetch_openml(name="house_prices", as_frame=True)
    data_ames = pd.DataFrame(data.data, columns=data.feature_names)
    data_ames['SalePrice'] = data.target
    numeric_features = data_ames.dtypes[data_ames.dtypes != 'object'].index
    data_ames = data_ames[numeric_features]
    data_ames = data_ames.apply(lambda x: (x - x.mean()) / (x.std()))
    data_ames = data_ames.fillna(0)

    X = data_ames.drop('SalePrice', axis=1).values
    Y = data_ames['SalePrice'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    n_train = X_train.shape[0]
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
    y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)

    x_train_tensor = X_train.to(device)
    y_train_tensor = y_train.to(device)
    x_test_tensor = X_test.to(device)
    y_test_tensor = y_test.to(device)

    num_feat = X_train.shape[1]
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

    return train_loader, test_loader, num_feat


class LinearRegressionModel(nn.Module):

    def __init__(self, num):
        super(LinearRegressionModel, self).__init__()
        # Hidden layer
        n_hidden = 32
        self.l1 = nn.Linear(num, n_hidden)
        # Output layer
        self.l2 = nn.Linear(n_hidden, 1)
        # ReLU func
        self.relu = nn.ReLU(inplace=True)

        self.net = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        y_pred = self.net(x)
        return y_pred


def train_network(net, train_data):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    learn_pro = np.zeros((0, 2))
    for epoch in range(500):
        train_loss = 0
        for X, y in train_data:
            # Forward pass: Compute predicted y by passing
            # x to the model
            X = X.to(device)
            y = y.to(device)

            y_pred = net(X)

            # Compute and print loss
            loss = torch.sqrt(criterion(y_pred, y))  # <- RMSE # loss = criterion(pred_y, y)

            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learn_pro = np.vstack((learn_pro, np.array([epoch, loss.item()])))
            train_loss += loss
        print('epoch {}, loss {}'.format(epoch, train_loss.item()))


def evaluate_network(model, testloader):

    with torch.no_grad():
        train_loss = 0
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = torch.sqrt(criterion(y_pred, y))  # <- RMSE # loss = criterion(pred_y, y)
            train_loss += loss
        print('loss {}'.format(train_loss.item()))


if __name__ == "__main__":
    train_iter, test_iter, num_features = get_data()
    model = LinearRegressionModel(num_features)
    model = model.to(device)
    train_network(model, train_iter)
    evaluate_network(model,test_iter)
