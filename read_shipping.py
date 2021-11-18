import numpy as np

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
    # 获取总记录数据量
    with open(DataPath) as f:
        lines = f.readlines()
        totalLines = len(lines)
    print("数据总行数：%d" % totalLines)

    RawTrajectories = []
    LoopCounter = 0

    with open(DataPath) as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split(DELIMINATOR)
            ## 在出租车轨迹数据中，每一行的第一个数据是出租车ID，其他为地理位置
            ## 格式如下：[Tax 1] [Position 1] [Position 2] [Position 3]...
            shipId = fields[0]
            movements = fields[1:]

            LoopCounter += 1
            if LoopCounter % 5000 == 0:
                print("当前进度：%d%%" % (LoopCounter*100/totalLines))
            elif LoopCounter == totalLines:
                print("当前进度：100%")

            ## 数据的预处理可以在下面添加

            ## 过滤噪声
            MinMovementLength = MINLENGTHFORTRAIN + LASTSTEPSHOLDOUTFORTEST
            if len(movements) < MinMovementLength:
                continue

            RawTrajectories.append([shipId, movements])
    return RawTrajectories, totalLines
