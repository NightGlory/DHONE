import os
import itertools

DELIMINATOR = ' '               # 分隔符
LASTSTEPSHOLDOUTFORTEST = 0     # 路径最后多少个是测试数据, 为0则不划分数据集
MINLENGTHFORTRAIN = 1           # 训练集最小长度

def ReadSequentialData(DataPath):
    '''
    读取数据，返回为 轨迹list.
    输入：原始文件位置
    输出：原始数据
    全局变量：LASTSTEPSHOLDOUTFORTEST/MINLENGTHFORTRAIN
    '''
    print('读取原始数据...')

    excelList = ExcelList(DataPath)

    # 获取总记录数据量
    TotalLines = 0
    for excel in excelList:
        if os.path.isdir(DataPath+excel):
            continue
        with open(DataPath+excel) as f:
            lines = f.readlines()
            TotalLines += len(lines)
    print("数据总行数：%d" % TotalLines)

    RawTrajectories = []
    LoopCounter = 0
    for excel in excelList:
        if os.path.isdir(DataPath+excel):
            continue
        with open(DataPath+excel) as f:
            lines = f.readlines()
            for line in lines:
                fields = line.strip().split(DELIMINATOR)
                ## 在出租车轨迹数据中，每一行的第一个数据是出租车ID，其他为地理位置
                ## 格式如下：[Tax 1] [Position 1] [Position 2] [Position 3]...
                taxId = fields[0]
                movements = fields[1:]

                LoopCounter += 1
                if LoopCounter % 5000 == 0:
                    print("当前进度：%d%%" % (LoopCounter*100/TotalLines))
                elif LoopCounter == TotalLines:
                    print("当前进度：100%")

                ## 数据的预处理可以在下面添加

                ## 过滤噪声
                MinMovementLength = MINLENGTHFORTRAIN + LASTSTEPSHOLDOUTFORTEST
                if len(movements) < MinMovementLength:
                    continue

                RawTrajectories.append([taxId, movements])
    return RawTrajectories, TotalLines

def ExcelList(path):
    '''
    获取文件夹中文件名的list
    '''
    filenames = os.listdir(path)
    list = []
    for filename in filenames:
        list.append(filename)
    return list

def GenerateTrajectoryFile(DataPath, WritePath):
    print('读取原始数据...')

    excelList = ExcelList(DataPath)

    wf = open(WritePath, "w")

    for excel in excelList:
        if os.path.isdir(DataPath+excel):
            continue
        with open(DataPath+excel) as f:
            lines = f.readlines()
            for line in lines:
                fields = line.strip().split(DELIMINATOR)
                ## 在出租车轨迹数据中，每一行的第一个数据是出租车ID，其他为地理位置
                ## 格式如下：[Tax 1] [Position 1] [Position 2] [Position 3]...
                movements = fields[1:]
                movements = [key for key,_ in itertools.groupby(movements)] # 去除相邻重复
                ## 过滤噪声
                MinMovementLength = MINLENGTHFORTRAIN + LASTSTEPSHOLDOUTFORTEST
                if len(movements) < MinMovementLength:
                    continue

                # 将movements保存到文件
                contentLine = "\t".join(movements)
                wf.write(contentLine+"\n")
    

    wf.close()


if __name__ == "__main__":
    GenerateTrajectoryFile("data/porto-taxi/", "data/porto-taxi/result/trajectory.txt")