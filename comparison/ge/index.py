import numpy as np

def RebuildIndexes(EdgeFilePath, IndexPyFile, IndexEdgeFile):
    indexDict = dict()
    currentIndex = 0
    with open(EdgeFilePath, "r") as rf:
        lines = rf.readlines()
        with open(IndexEdgeFile, "w") as wf:
            for line in lines:
                nodes = line.strip().split(" ")
                for node in nodes:
                    if node not in indexDict.keys():
                        indexDict[node] = currentIndex
                        currentIndex += 1
                   
                # 写入改写成index的edge文件
                wf.write(str(indexDict[nodes[0]])+" "+str(indexDict[nodes[1]])+"\n")
    # 保存字典
    np.save(IndexPyFile, indexDict)
    # 写入改写成index的edge文件             

    return indexDict