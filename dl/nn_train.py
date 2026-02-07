import numpy as np


def grad_descent_linear_regression_batch(n: int, data_iter, lr=0.01, epochs=3):
    # 随机初始化参数
    w = np.random.random(n)
    b = 0.0
    for _ in range(epochs):
        # x: [batch, n], y: [batch]
        for x, y in data_iter:
            y_pred = x @ w + b
            diff = y_pred - y
            # loss = 0.5 * (diff ** 2).mean()
            # 正确的向量化梯度：grad_w = X^T @ diff / batch_size
            grad_w = x.T @ diff / x.shape[0]
            grad_b = diff.mean()
            w -= lr * grad_w
            b -= lr * grad_b
