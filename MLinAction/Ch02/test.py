import kNN

# #1测试简单分类器
# group,labels=kNN.createDataSet()
# a=kNN.classify0([0,0], group, labels, 3)
# print(a)


# #2测试约会数据处理器
# datingDataMat,datingLabels = kNN.file2matrix(r'c:\Users\Administrator\Desktop\ING\pytest\MLinAction\Ch02\datingTestSet2.txt')   
# # print(datingDataMat)
# # print(datingLabels[0:20])


# #3测试样本数据中2和3列，matplotlib散点图
# import matplotlib
# import matplotlib.pyplot as plt
# from numpy import *

# datingDataMat,datingLabels = kNN.file2matrix(r'c:\Users\Administrator\Desktop\ING\pytest\MLinAction\Ch02\datingTestSet2.txt')   
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))      #2,3列数据
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))      #1,2列数据
# plt.show()          #显示图


# # 4测试归一化数值
# datingDataMat,datingLabels = kNN.file2matrix(r'c:\Users\Administrator\Desktop\ING\pytest\MLinAction\Ch02\datingTestSet2.txt')   
# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)


# #5测试分类器对约会网站的错误率
# kNN.datingClassTest()


# #6测试约会网站
# kNN.classifyPerson()


# #7测试img2vector函数
# testVector = kNN.img2vector(r'c:\Users\Administrator\Desktop\ING\pytest\MLinAction\Ch02\testDigits\0_13.txt')
# print(testVector[0,0:31])
# print(testVector[0,32:63])

#8测试手写识别函数
kNN.handwritingClassTest()
