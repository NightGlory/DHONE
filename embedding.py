from collections import defaultdict
import numpy as np
import sklearn.decomposition as skd
import math
import time


class HONEM:
    def __init__(self):
        self.indexDict = dict()
        self.adjacencyMatrices = defaultdict(dict)  # 邻接矩阵
        self.embeddedMatrix = [[]]                  # 最终嵌入矩阵
    
    def EmbeddingMatrix(self, RuleFilePath, IndexPyFile, EmbeddedSavePath, Dimension):
        '''
        获取嵌入矩阵
        '''
        # 重建索引并保存
        self.indexDict = RebuildIndexes(RuleFilePath, IndexPyFile)
        self.adjacencyMatrices = AdjacencyMatricesOfDifferentOrders(RuleFilePath, IndexPyFile)
        combinedMatrix = self.CombineMatrix()
        self.embeddedMatrix = TruncatedSVD(combinedMatrix, Dimension)
        # 保存到文件
        np.savetxt(EmbeddedSavePath, self.embeddedMatrix, delimiter=",", fmt='%.6f')
        
        return self.embeddedMatrix

    def CombineMatrix(self):
        '''
        合并矩阵, e^-k
        '''
        combinedMatrix = [[]]

        for order in range(1,len(self.adjacencyMatrices)+1):
            print("合并进度：%d/%d" % (order, len(self.adjacencyMatrices)))
            currentMatrix = self.adjacencyMatrices[order]

            if order==1:
                combinedMatrix = currentMatrix
            else:
                combinedMatrix = currentMatrix * math.e**(1-order) + combinedMatrix
        print("合并矩阵完成")
        nodeSize = len(combinedMatrix)
        print("嵌入时计算的节点个数：", nodeSize)
        return combinedMatrix/nodeSize
    

class Method1:
    def __init__(self):
        self.indexDict = dict()
        self.adjacencyMatrices = defaultdict(dict)  # 邻接矩阵
        self.embeddedMatrix = [[]]                  # 最终嵌入矩阵
        self.densityList = []
    
    def EmbeddingMatrix(self, RuleFilePath, IndexPyFile, EmbeddedSavePath, Dimension):
        '''
        获取嵌入矩阵
        '''
        # 获取各阶密度
        self.densityList = DensityStatistics(RuleFilePath)
        # 重建索引并保存
        self.indexDict = RebuildIndexes(RuleFilePath, IndexPyFile)
        self.adjacencyMatrices = AdjacencyMatricesOfDifferentOrders(RuleFilePath, IndexPyFile)
        combinedMatrix = self.CombineMatrix()
        self.embeddedMatrix = TruncatedSVD(combinedMatrix, Dimension)
        # 保存到文件
        np.savetxt(EmbeddedSavePath, self.embeddedMatrix, delimiter=",", fmt='%.6f')
        
        return self.embeddedMatrix

    def CombineMatrix(self):
        '''
        合并矩阵, density
        '''
        combinedMatrix = [[]]

        for order in range(1,len(self.adjacencyMatrices)+1):
            print("合并进度：%d/%d" % (order, len(self.adjacencyMatrices)))
            currentMatrix = self.adjacencyMatrices[order]
            if order==1:
                combinedMatrix = currentMatrix * parabolic(self.densityList[0])
            else:
                combinedMatrix = currentMatrix * parabolic(self.densityList[order-1]) + combinedMatrix

        print("合并矩阵完成")
        nodeSize = len(combinedMatrix)
        print("嵌入时计算的节点个数：", nodeSize)
        return combinedMatrix/nodeSize

def DensityStatistics(rulesFile):
    densityList = []
    # 各阶规则个数分布
    orderDistribution = np.zeros((20,), dtype=np.int)
    with open(rulesFile, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            orderDistribution[(len(line)-2)] += 1
    print()
    for i in orderDistribution:
        densityList.append(i/orderDistribution[0])
    return densityList

def AdjacencyMatricesOfDifferentOrders(FilePath, IndexFilePath):
    '''
    不同阶的邻接矩阵，返回的数据结构为[阶数,[邻接矩阵]]
    '''
    adjacencyMatrices = defaultdict(dict)
    indexDict = np.load(IndexFilePath, allow_pickle=True).item()

    with open(FilePath) as f:
        lines = f.readlines()
        # 先获取总的节点个数
        nodeSize = len(indexDict.keys())
        print("总节点个数为：%d" % nodeSize)
        # 再处理邻接矩阵
        for line in lines:
            path = line.strip().split(" ")
            order = len(path)-1
            # 判断字典中有没有这个阶数，没有则在字典中添加
            if order not in adjacencyMatrices.keys():
                # 则新建邻接矩阵
                adjacency = np.zeros((nodeSize, nodeSize), np.int)
                adjacencyMatrices[order] = adjacency
            lastVertex = indexDict[path[-1]]
            lastSecondVertex = indexDict[path[-2]]
            adjacencyMatrices[order][lastSecondVertex, lastVertex] = 1
        print("最高阶：%d" % len(adjacencyMatrices))
        return adjacencyMatrices

def TruncatedSVD(matrix, k):
        '''
        截断SVD，matrix为高阶依赖关系矩阵，k为降低的维度
        '''
        embeddedMatrix = [[]]
        start = time.time()
        # 奇异值分解
        trsvd = skd.TruncatedSVD(n_components=k)
        X_transformed = trsvd.fit_transform(matrix)
        U = X_transformed / trsvd.singular_values_
        S = np.diag(trsvd.singular_values_)
        embeddedMatrix  = np.dot(U,np.sqrt(S))       # 嵌入矩阵，为截断后的U*sqrt(Sigma)

        print("嵌入矩阵的大小："+str(np.shape(embeddedMatrix)))
        print("嵌入矩阵计算完成")
        end = time.time()
        print("SVD耗时: %f s" % (end - start))
        return embeddedMatrix


def RebuildIndexes(RuleFilePath, IndexPyFile):
    indexDict = dict()
    currentIndex = 0
    with open(RuleFilePath) as f:
        lines = f.readlines()
        for line in lines:
            nodes = line.strip().split(" ")
            for node in nodes:
                if node not in indexDict.keys():
                    indexDict[node] = currentIndex
                    currentIndex += 1
    # 保存字典
    np.save(IndexPyFile, indexDict)

    return indexDict

def softsign(x):
    return x / (1. +abs(x))

def softsign_variety(x):
    return 2. * softsign(x)

def parabolic(x):
    '''
    抛物线
    '''
    return math.sqrt(x)