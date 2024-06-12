from torch import nn
import torch
import numpy as np


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


if __name__ == '__main__':
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, 1.0]])
    Y = corr2d(X, K)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 1e-2

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()

        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

    print(conv2d.weight.data.reshape((1, 2)))