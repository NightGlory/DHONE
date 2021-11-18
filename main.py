
import read_porto as ReadPortoFile
import read_shipping as ReadShipping
import read_ucforum as ReadUC
import extract_rules as ExtractRules
import embedding as Embedding
import itertools
import time
import numpy as np

# 算法中的一些参数定义
# ------------------------porto的文件参数----------------------------------------------
inputFileName = "data/porto-taxi/0.csv"
dataPath = "data/porto-taxi/"
outputRulesFile = "data/porto-taxi/result/rulesFile.txt"
outputNetworkFile = "data/porto-taxi/result/networkFile.txt"
rulesPyFile = "data/porto-taxi/result/temp/rules.npy"
embeddedSavePath = "data/porto-taxi/result/embeddedMatrix.txt"
dimension = 5
fonFile = "data/porto-taxi/result/fonFile.txt"
indexFile = "data/porto-taxi/result/temp/index.npy"

# ------------------------shipping的文件参数----------------------------------------------
# inputFileName = "data/shipping/data.csv"
# outputRulesFile = "data/shipping/result/rulesFile.txt"
# rulesPyFile = "data/shipping/result/temp/rules.npy"
# embeddedSavePath = "data/shipping/result/embeddedMatrix.txt"
# fonFile = "data/shipping/result/fonFile.txt"
# dimension = 6
# indexFile = "data/shipping/result/temp/index.npy"

# ------------------------startups的文件参数----------------------------------------------
# inputFileName = "data/start-ups/rewritedDataSource.txt"

# ------------------------facebook的文件参数----------------------------------------------
# inputFileName = "data/facebook/data.txt"
# rulesPyFile = "data/facebook/result/temp/rules.npy"
# outputRulesFile = "data/facebook/result/rulesFile.txt"
# indexFile = "data/facebook/result/temp/index.npy"
# fonFile = "data/facebook/result/fonFile.txt"
# embeddedSavePath = "data/facebook/result/embeddedMatrix.txt"
# dimension = 100

# ------------------------UC Forum----------------------------------------------
# inputFileName = "data/ucforum/data.txt"
# rulesPyFile = "data/ucforum/temp/rules.npy"
# outputRulesFile = "data/ucforum/rulesFile.txt"
# indexFile = "data/ucforum/temp/index.npy"
# fonFile = "data/ucforum/fonData.txt"
# embeddedSavePath = "data/ucforum/embeddedMatrix.txt"
# dimension = 25

###########################################
# 超参数
MAXORDER = 5
MINSUPPORT = 1
LASTSTEPSHOLDOUTFORTEST = 0  # 路径最后多少个是测试数据, 为0则不划分数据集

###########################################
# Util Functions
###########################################

def BuildTrainingAndTesting(rawTrajectories):
    """
    划分训练与测试数据.
    输入：轨迹
    输出：两个(训练/测试)集
    LASTSTEPSHOLDOUTFORTEST
    """
    print('划分训练与测试数据...')
    Training = []
    Testing = []
    for trajectory in rawTrajectories:
        tax, movement = trajectory
        movement = [key for key,_ in itertools.groupby(movement)] # 去除相邻重复
        if LASTSTEPSHOLDOUTFORTEST > 0:
            Training.append([tax, movement[:-LASTSTEPSHOLDOUTFORTEST]])
            Testing.append([tax, movement[-LASTSTEPSHOLDOUTFORTEST]])
        else:
            Training.append([tax, movement])
    return Training, Testing

###########################################
# Main function
###########################################

if __name__ == "__main__":
    begin = time.clock()
    # 读取数据
    rawTrajectories, totalLines = ReadPortoFile.ReadSequentialData(dataPath)
    # rawTrajectories, totalLines = ReadShipping.ReadSequentialData(inputFileName)
    # rawTrajectories = ReadStartups.InvestmentTrack(inputFileName)
    # rawTrajectories, totalLines = ReadFacebook.generate_movements(inputFileName)
    # rawTrajectories, totalLines = ReadUC.generate_movements(inputFileName)

    # 划分训练与测试数据
    trainingTrajectory, testingTrajectory = BuildTrainingAndTesting(rawTrajectories)
    print("有效数据集长度：%d" % len(trainingTrajectory))
    rules = ExtractRules.ExtractRules(trainingTrajectory, MAXORDER, MINSUPPORT)
    np.save(rulesPyFile, rules)
    # 将Rules[字典]保存到txt文本
    ExtractRules.DumpRules(rules, outputRulesFile)

    # 读取Rules
    ReadRules = np.load(rulesPyFile, allow_pickle=True).item()

    # 嵌入
    honem = Embedding.Method1()
    embeddedMatrix = honem.EmbeddingMatrix(outputRulesFile, indexFile, embeddedSavePath, dimension)
    
    end = time.clock()
    print("共用时：%.2f 分钟" % ((end-begin)/60))

