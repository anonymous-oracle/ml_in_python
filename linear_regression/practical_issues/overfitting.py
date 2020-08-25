import numpy as np
import matplotlib.pyplot as plt

# # make up some data and plot it
# n = 100
# x = np.linspace(0, 6 * np.pi, n)
# y = np.sin(x)
#
#
# # plt.plot(x, y)
# # plt.show()

def make_poly(x, deg):
    n = len(x)
    data = [np.ones(n)]
    # adding polynomial terms to data like in polynomial regression
    for d in range(deg):
        data.append(x ** (d + 1))
    # np.vstack is used to append N-sized 'deg' number of dimensions along with bias terms; transpose is done to convert
    # (deg x N) matrix into (N x deg) matrix
    return np.vstack(data).T


def fit(x, y):
    return np.reshape(np.linalg.solve(x.T.dot(x), y.T.dot(x)), newshape=(np.shape(x)[1], 1))


def fit_and_display(x, y, sample, deg):
    n = len(x)
    train_idx = np.random.choice(n, sample)
    # the above choice() function chooses 'sample' number of samples from a specified 1D array or int, where it is equivalent to
    # range(n); train_idx will contain a list of random int indices where the samples for respective splitting can be taken
    xtrain = x[train_idx]
    ytrain = y[train_idx]

    plt.scatter(xtrain, ytrain)
    plt.show()

    # fit polynomial
    xtrain_poly = make_poly(xtrain, deg)
    w = fit(xtrain_poly, ytrain)

    # display the polynomial
    x_poly = make_poly(x, deg)
    y_hat = np.squeeze(x_poly.dot(w))
    plt.plot(x, y)
    plt.plot(x, y_hat)
    plt.scatter(xtrain, ytrain)
    plt.title("deg = {}".format(deg))
    plt.show()


def get_mse(y, y_hat):
    d = y - y_hat
    return d.dot(d) / len(d)


def plot_train_vs_test_curves(x, y, sample=80, max_deg=20):
    n = len(x)
    train_idx = np.random.choice(n, sample)
    xtrain = x[train_idx]
    ytrain = y[train_idx]

    test_idx = [idx for idx in range(n) if idx not in train_idx]

    xtest = x[test_idx]
    ytest = y[test_idx]

    mse_trains = []
    mse_tests = []
    for deg in range(max_deg + 1):
        xtrain_poly = make_poly(xtrain, deg)
        w = fit(xtrain_poly, ytrain)
        yhat_train = np.squeeze(xtrain_poly.dot(w))
        mse_train = get_mse(ytrain, yhat_train)

        xtest_poly = make_poly(xtest, deg)
        yhat_test = np.squeeze(xtest_poly.dot(w))
        mse_test = get_mse(ytest, yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # make up some data and plot it
    N = 100
    X = np.linspace(0, 6 * np.pi, N)
    Y = np.sin(X)

    plt.plot(X, Y)
    plt.show()

    for deg in (5, 6, 7, 8, 9):
        fit_and_display(X, Y, 80, deg)
    plot_train_vs_test_curves(X, Y)
