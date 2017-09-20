import kNN

group,labels=kNN.createDataSet()
a=kNN.classify0([0,0], group, labels, 3)
print(a)