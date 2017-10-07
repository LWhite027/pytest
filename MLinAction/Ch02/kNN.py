'''
@author: LWhite027
'''
from numpy import *
import operator     #运算符模块



#创建训练样本
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#输入：训练样本向量，目标样本矩阵，便签，最近邻数
#输出：分类后的结果
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]              #取样本训练集矩阵第一维的长度（行数）
#计算向量xA和xB之间欧式距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #重复构造数组inX，使其有dataSize行，1列，最后两矩阵相减
    sqDiffMat = diffMat**2
    sqDsitances = sqDiffMat.sum(axis=1)         #普通sum默认参数为axis=0为普通相加，axis=1为一行的行向量相加
    distances = sqDsitances**0.5
    sortedDistIndicies = distances.argsort()    #argsort返回数值从小到大的索引值（数组索引0,1,2,3）
#选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #根据排序结果的索引值返回最靠近的第k个标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   #依次计算各个标签出此案的频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     #排序频率
    #itemgetter(1)按照第一维度排序（0,1,2,3）
    #reverse默认顺序，此时需reverse=True使其逆序
    return sortedClassCount[0][0]   #找出频率最高的

#输入：训练样本数据文件
#输出：训练样本矩阵和类标签
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)        #得到文件行数
    returnMat = zeros((numberOfLines,3))    #创建返回的NumpyPy矩阵（初始化为[文件]行，3列的0矩阵）
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()                         #删除空白符
        listFromLine = line.split('\t')             #split制定分隔符对数据进行切片
        returnMat[index, :] = listFromLine[0:3]     #选取前3个元素（特征）存储在返回矩阵中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#归一化数值
#归一化公式：（当前值-最小值）/（最大值-最小值）
def autoNorm(dataSet):
    minVals = dataSet.min(0)        #存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    maxVals = dataSet.max(0)        #存放每列最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))         #初始化归一化矩阵为读取的dataSet
    m = dataSet.shape[0]            #m保存dataSet行数
    #特征矩阵是3x1000，min max range是1x3 因此采用tile将变量内容复制成输入矩阵同大小
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#测试约会网站分类结果代码
def datingClassTest():
    hoRatio = 0.10          #设置测试样本比例占10%
    datingDataMat,datingLabels = file2matrix(r'c:\Users\Administrator\Desktop\ING\pytest\MLinAction\Ch02\datingTestSet2.txt')      #读取数据
    normMat, ranges, minVals = autoNorm(datingDataMat)                  #归一化特征值
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 4)    #设置近邻数k=4
        print("the classifier came back with: %d, th real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): 
            errorCount += 1.0
            print("different!")
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))





