'''
@author: LWhite027
'''
from numpy import *
import operator     #运算符模块




def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

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




