import numpy as np
from scipy.spatial import distance
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def read_data(path):
    file = np.loadtxt(path)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    y = np.array(file[:, 1])
    data = np.vstack((x1, x2, y)).T
    return data


def splitToFeaturesLabels(data):
    X = np.array(data[:, 0:2])
    Y = np.array(data[:, -1])
    return X, Y


def draw(trainFeatures, trainLabels):
    fig, ax = plt.subplots()
    fig = plt.gcf()
    ax = fig.gca()

    # Draw points:
    for i in range(trainFeatures.shape[0]):
        if trainLabels[i] == 1:  # Positive points:
            plt.scatter(trainFeatures[i, 0], trainFeatures[i, 1], color="green", s=30)
        else:  # Negative points:
            plt.scatter(trainFeatures[i, 0], trainFeatures[i, 1], color="blue", s=30)
        ax.grid()

    # Add legend to colors:
    green_patch = patches.Patch(color='green', label='Positive points (male)')
    blue_patch = patches.Patch(color='blue', label='Negative points (female)')
    plt.legend(handles=[green_patch, blue_patch])
    plt.show()


"""
In the experiment we randomly divide the data-points for each iteration 
into 65 training points and 65 test points
"""


def splitTestTrain(data):
    np.random.shuffle(data)
    train = data[: 65, :]
    test = data[66:, :]
    return train, test


"""
Split into 2 columns of features (Body temperature, Heart rate)
and one vector of labels (gender)
"""


def UpdateNeighbors(neighbors, point, distance, k):
    if k > len(neighbors):
        neighbors.append([distance, point])

    if neighbors[-1][0] > distance:
        neighbors[-1] = [distance, point]

    neighbors.sort(key=lambda x: x[0])
    return neighbors


def knn(train, point, p, k):
    error = 0.0
    neighbors = []
    man = 0
    woman = 0

    for pointTrain in train:
        dis = distance.minkowski(pointTrain[0:2], point[0:2], p)
        neighbors = UpdateNeighbors(neighbors, pointTrain, dis, k)

    for i in range(k):
        if neighbors[i][1][2] == 2.0:
            woman += 1
        else:
            man += 1
    if woman > man:
        return 2

    return 1


def knnExperiment(data):
    errorTest = 0.0
    errorTrain = 0.0
    bestK = ""
    bestP = ""
    bestError = np.inf
    for p in [1, 2, np.inf]:
        print("P = ", p)
        print("--------")
        for k in range(1, 10, 2):
            errorTest = 0.0
            errorTrain = 0.0
            for i in range(500):
                train, test = splitTestTrain(data)
                for pointTest in test:
                    guess = knn(train, pointTest, p, k)
                    if guess != pointTest[2]:
                        errorTest += 1

                for pointTrain in train:
                    guess = knn(train, pointTrain, p, k)
                    if guess != pointTrain[2]:
                        errorTrain += 1

            if errorTest < bestError:
                bestK = k
                bestP = p
                bestError = errorTest

            print("Test: for k:", k, "the Error is: ", (errorTest / 500) / 65)
            print("Train: for k:", k, "the Error is: ", (errorTrain / 500) / 65)
            print("------------------------------------------------------------")

    print("Best result only about test: error =", (bestError / 500) / 65, "% , p = ", bestP, ", k = ", bestK)


if __name__ == '__main__':
    # initiailize data set
    path = 'C:/Users/HC_Body_Temperature'
    data = read_data(path)
    X, Y = splitToFeaturesLabels(data)
    draw(X, Y)
    knnExperiment(data)
