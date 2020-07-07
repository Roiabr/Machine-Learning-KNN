# KNN Algorithm  
![knn](https://user-images.githubusercontent.com/44756354/86769164-63f66000-c057-11ea-8449-fea8d3392ab7.png)




## About the project:
In this project I implemented the KNN algorithm in the machine learning course.
The dataset is the Hope College Temperature data set.
The data described by body temperature in degrees Fahrenheit, the gender (1 = male, 2 = female) and the heart rate in beats per minute.

We are running the knn algorithm 500 times for each of k = 1, 3, 5, 7, 9 when k is the number of neighbours.
For each run, randomly dividing the points into 50% base points  and 50% test points.
Then run knn on p when p is the minkowski distance(1, 2, infinity), and after computing the final best neighbours, find its error T on every points in test and traing.
Dataset contains 130 data points. The label (1 and -1) will be the gender, and the temperature and heartrate define the 2-dimensional point.

##
Roi Abramovitch
