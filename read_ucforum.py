import numpy as np
import itertools

def generate_movements(rawData):
    # 初始化
    movementsDict = dict()
    RawTrajectories = []
    TotalLines = 0
    with open(rawData, "r") as f:
        lines = f.readlines()
        TotalLines = len(lines)
        print("轨迹总数：%d" % TotalLines)
        for line in lines:
            data = line.strip().split(" ")
            fromNode = data[0]
            toNode = data[1]
            if fromNode not in movementsDict.keys():
                movementsDict[fromNode] = [toNode]
            else:
                movementsDict[fromNode].append(toNode)
        
        pathIndex = 100000
        wf = open("data/ucforum/tracject.txt", "w")
        for val in movementsDict.values():

            movements = [key for key,_ in itertools.groupby(val)] # 去除相邻重复

            # 保存路径到文本

            pathLen = len(movements)
            if pathLen>1:
                traj = " ".join(movements)
                wf.write(traj+"\n")

                RawTrajectories.append([pathIndex, np.array(movements)])
                pathIndex += 1
        
        wf.close()

    return RawTrajectories, TotalLines

def get_fon_file(ruleFile, fonFile):
    with open(fonFile, 'w') as wf:
        with open(ruleFile) as rf:
            lines = rf.readlines()
            for line in lines:
                length = len(line.split(" "))
                if length==2:
                    wf.write(line)
    wf.close()

def GenerateIndexes(fonFile, indexFile):
    indexDict = {}
    current = 1
    with open(fonFile) as f:
        lines = f.readlines()
        for line in lines:
            node = line.strip().split(" ")
            for i in node:
                if i not in indexDict.keys():
                    indexDict[i] = current
                    current +=1
    print(indexDict)
    np.save(indexFile, indexDict)

if __name__ == "__main__":
    get_fon_file("data/ucforum/rulesFile.txt", "data/ucforum/fonData.txt")