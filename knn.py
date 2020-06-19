import numpy as np
from scipy.spatial import distance


def read_data(path):
    file = np.loadtxt(path)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    y = np.array(file[:, 1])
    data = np.vstack((x1, x2, y)).T
    return data


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
        if pointTrain[0] == point[0]:
            continue
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
    for p in [1, 2, np.inf]:
        print("P = ", p)
        for k in range(1, 10, 2):
            for i in range(1):
                train, test = splitTestTrain(data)
                for pointTest in test:
                    guess = knn(train, pointTest, p, k)
                    if guess != pointTest[2]:
                        errorTest += 1
                for pointTrain in train:
                    guess = knn(train, pointTrain, p, k)
                    if guess != pointTrain[2]:
                        errorTrain += 1

            print("Test: for k:", k, "the Error is: ", (errorTest / 1) / 65)
            print("Train: for k:", k, "the Error is: ", (errorTrain / 1) / 65)
            print("------------------------------------------------------------")
            errorTest = 0.0
            errorTrain = 0.0

if __name__ == '__main__':
    # initiailize data set
    path = 'C:/Users/Roi Abramovitch/Downloads/לימודים מדעי המחשב/שנה ג/למידת מכונה/מטלות/מטלה 3/HC_Body_Temperature'
    data = read_data(path)
    knnExperiment(data)
