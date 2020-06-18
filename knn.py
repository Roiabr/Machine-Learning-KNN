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


def splitToFeaturesLabels(data):
    X = np.array(data[:, 0:2])
    Y = np.array(data[:, -1])
    return X, Y


def UpdateNeighbors(neighbors, point, distance, k):
    if k > len(neighbors):
        neighbors.append([distance, point])

    if neighbors[-1][0] > distance:
        neighbors[-1] = [distance, point]

    neighbors.sort(key=lambda x: x[0])
    return neighbors


def knn(train, test, p, k):
    error = 0.0
    if p == 3:
        p = np.inf
    neighbors = []
    man = 0
    woman = 0
    for pointTest in test:
        for pointTrain in train:
            dis = distance.minkowski(pointTrain[0:2], pointTest[0:2], p)
            # print(dis)
            neighbors = UpdateNeighbors(neighbors, pointTrain, dis, k)
            print(neighbors)
    #     for i in range(k):
    #         if neighbors[i][1][2] == 2.0:
    #             woman += 1
    #         else:
    #             man += 1
    #     # print(man, woman)
    #     if (woman > man and pointTest[2] == 1) or man > woman and pointTest[2] == 2:
    #         error += 1
    # return error


def knnExperiment(data):
    error = 0.0
    for p in range(1, 4):
        print("# P = ", p)
        for k in range(1, 10, 2):
            for i in range(500):
                train, test = splitTestTrain(data)
                error = knn(train, test, p, k)
            #print((error / 500) / 65)


if __name__ == '__main__':
    # initiailize data set
    path = 'C:/Users/Roi Abramovitch/Downloads/לימודים מדעי המחשב/שנה ג/למידת מכונה/מטלות/מטלה 3/HC_Body_Temperature'
    data = read_data(path)
    knnExperiment(data)
