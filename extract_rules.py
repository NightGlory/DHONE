
from collections import defaultdict

import math

ThresholdMultiplier = 1                         # 阈值的乘数

Count = defaultdict(lambda: defaultdict(int))   # 高阶关系的轨迹统计矩阵
Rules = defaultdict(dict)
Distribution = defaultdict(dict)                # 已有路径的下一个节点的概率分布
SourceToExtSource = defaultdict(set)
divergences = []
Verbose = True
StartingPoints = defaultdict(set)               # 这是一个集合字典，key是已有的路径，value是下一个节点对应的第几个轨迹的第几个节点
Trajectory = []
MinSupport = 1

def Initialize():
    global Count
    global Rules
    global Distribution
    global SourceToExtSource
    global StartingPoints

    Count = defaultdict(lambda: defaultdict(int))
    Rules = defaultdict(dict)
    Distribution = defaultdict(dict)
    SourceToExtSource = defaultdict(set)
    StartingPoints = defaultdict(set)

def ExtractRules(T, MaxOrder, MS):
    '''
    提取高阶规则
    输入：T:轨迹集合；MaxOrder：最高阶数；MS：最小支持度
    输出：
    '''
    # 初始化全局变量
    Initialize()
    # 全局变量的赋值
    global Trajectory
    global MinSupport
    Trajectory = T
    MinSupport = MS
    # 建立高阶关系
    BuildOrder(order=1, Trajectory=Trajectory, MinSupport=MinSupport)
    GenerateAllRules(MaxOrder, Trajectory, MinSupport)
    return Rules


def BuildOrder(order, Trajectory, MinSupport):
    '''
    输入：（阶数，轨迹，最小支持度）
    '''
    # 统计该阶的路线出现次数矩阵
    BuildObservations(Trajectory, order)
    # 统计该阶的路线的下一节点分布矩阵
    BuildDistributions(MinSupport, order)


def BuildObservations(Trajectory, order):
    '''
    用于统计不同阶路线对应的下个节点的分布数量,结果存在全局的StartingPoints中
    输入：（轨迹，阶数）
    '''
    print('赋值%d阶数据统计矩阵' % order)
    LoopCounter = 0
    trajectorySize = len(Trajectory)
    for Tindex in range(trajectorySize):   # 遍历所有轨迹
        LoopCounter += 1
        if LoopCounter % 10000 == 0:
            print("当前进度：%d%%" % (LoopCounter*100/trajectorySize))
        elif LoopCounter == trajectorySize:
                print("当前进度：100%")
        # 删除存储在第一个元素中的元数据
        # 这一步可以扩展为包含更丰富的信息
        trajectory = Trajectory[Tindex][1]      # 将出租车ID移除，只保留轨迹信息

        for index in range(len(trajectory) - order):
            # 获取轨迹中n阶的路线(不包括尾节点)
            Source = tuple(trajectory[index:index+order])       # 将当前轨迹切割成所有符合要求的order-1阶子链,order=1代表只有头节点
            Target = trajectory[index+order]                    # k阶子链的尾节点，即Source子链的下一节点，组合起来代表一个order阶子链
            Count[Source][Target] += 1                          # 获取高阶关系的轨迹统计矩阵，将出现次数保存在【global】Count中
            StartingPoints[Source].add((Tindex, index))         # 这是一个集合字典，key是已有的路径，value是下一个节点对应的第几个轨迹的第几个节点


def BuildDistributions(MinSupport, order):
    '''
    将当前轨迹的下个节点的分布概率保存在全局的Distribution，即观察到的转移概率
    输入：（最小支持度，阶数）
    '''
    print("以最小支持度%d和阈值乘数%.2f构造分布统计矩阵"%(MinSupport, ThresholdMultiplier))
    for Source in Count:            # 遍历Count， 每一个代表轨迹中n阶的路线(不包括尾节点)
        if len(Source) == order:    # 如果路线长度已经是我们的阶数，order=1时，涉及到两个节点，但是source = order = 1，所以恒为真
            # 下面的循环是将统计个数小于最小支持度的数据清零，即去噪
            for Target in Count[Source].keys():
                if Count[Source][Target] < MinSupport:
                    Count[Source][Target] = 0
            # 下面的循环是为了获取路径的下个节点的分布概率
            for Target in Count[Source]:
                if Count[Source][Target] > 0:
                    Distribution[Source][Target] = 1.0 * Count[Source][Target] / sum(Count[Source].values())


def GenerateAllRules(MaxOrder, Trajectory, MinSupport):
    '''
    生成高阶依赖规则
    输入：(最高阶，轨迹，最小支持度)
    '''
    print('生成高阶依赖规则...')
    progress = len(Distribution)
    print("已有路径的个数：%d" % progress)
    LoopCounter = 0
    for Source in tuple(Distribution.keys()):       # 对于每一个已有路径Source
        AddToRules(Source)
        ExtendRule(Source, Source, 1, MaxOrder, Trajectory, MinSupport)
        LoopCounter += 1
        if LoopCounter % 10 == 0:
            print('正在生成规则...  进度：%d/%d' % (LoopCounter, progress))


def ExtendRule(Valid, Curr, order, MaxOrder, Trajectory, MinSupport):
    '''
    根据已有规则拓展
    '''
    if order >= MaxOrder:
        AddToRules(Valid)
    else:
        Distr = Distribution[Valid]

        if KLD(MaxDivergence(Distribution[Curr]), Distr) < KLDThreshold(order+1, Curr):     # 如果散度小于动态阈值
            AddToRules(Valid)
        else:
            NewOrder = order + 1
            Extended = ExtendSourceFast(Curr)
            if len(Extended) == 0:
                AddToRules(Valid)
            else:
                for ExtSource in Extended:
                    ExtDistr = Distribution[ExtSource]
                    divergence = KLD(ExtDistr, Distr)
                    if divergence > KLDThreshold(NewOrder, ExtSource):
                        # NewOrder存在高阶依赖关系，继续比较高阶与当前阶的概率分布
                        ExtendRule(ExtSource, ExtSource, NewOrder, MaxOrder, Trajectory, MinSupport)
                    else:
                        # NewOrder不存在高阶依赖关系，继续比较高阶概率分布与已知阶概率分布
                        ExtendRule(Valid, ExtSource, NewOrder, MaxOrder, Trajectory, MinSupport)


def MaxDivergence(Distr):
    '''
    获取最大散度
    '''
    MaxValKey = sorted(Distr, key=Distr.__getitem__)
    d = {MaxValKey[0]: 1}
    return d


def AddToRules(Source):
    '''
    输入：已有路径
    拓展规则
    '''
    for order in range(1, len(Source)+1):
        s = Source[0:order]     # 获取从起始开始的每一段  
        
        if not s in Distribution or len(Distribution[s]) == 0:  # 若当前段不在分布上
            ExtendSourceFast(s[1:])     # 当前轨迹除了第一个节点
        for t in Count[s]:
            if Count[s][t] > 0:
                Rules[s][t] = Count[s][t]

def ExtendSourceFast(Curr):
    '''
    输入：当前段，当不在Distribution中时
    拓展Source
    '''
    if Curr in SourceToExtSource:           # 第一次实验：SourceToExtSource = {}
        return SourceToExtSource[Curr]
    else:
        ExtendObservation(Curr)
        if Curr in SourceToExtSource:
            return SourceToExtSource[Curr]
        else:
            return []

def ExtendObservation(Source):
    '''
    输入：当前已有的路径
    拓展计数
    '''
    if len(Source) > 1:     # 非空的情况,2阶的时候全都为0？
        if (not Source[1:] in Count) or (len(Count[Source]) == 0):      # 当前路径除去第一个节点，如果剩下的路径不在Count中（即已有路径不存在该条），第二个条件似乎永远为假
            # 超过4阶（包含）才会运行到这里
            ExtendObservation(Source[1:])

    order = len(Source)                         # 获取当前阶
    C = defaultdict(lambda: defaultdict(int))   # 初始化

    for Tindex, index in StartingPoints[Source]:
        if index - 1 >= 0 and index + order < len(Trajectory[Tindex][1]):       # 如果index是不第一个（永真，因为我们如此设计）且index的后order跳存在
            ExtSource = tuple(Trajectory[Tindex][1][index - 1:index + order])
            Target = Trajectory[Tindex][1][index + order]
            C[ExtSource][Target] += 1
            StartingPoints[ExtSource].add((Tindex, index - 1))

    if len(C) == 0:
        return
    
    for s in C:
        for t in C[s]:
            if C[s][t] < MinSupport:    # 支持度
                C[s][t] = 0
            Count[s][t] += C[s][t]
        CsSupport = sum(C[s].values())
        for t in C[s]:
            if C[s][t] > 0:
                Distribution[s][t] = 1.0 * C[s][t] / CsSupport
                SourceToExtSource[s[1:]].add(s)

def KLD(a, b):
    '''
    计算散度
    '''      
    divergence = 0
    for target in a:
        divergence += GetProbability(a, target) * math.log((GetProbability(a, target)/GetProbability(b, target)), 2)
    return divergence

def KLDThreshold(NewOrder, ExtSource):
    '''
    动态阈值
    '''
    return ThresholdMultiplier * NewOrder / math.log(1 + sum(Count[ExtSource].values()), 2)

def GetProbability(d, key):
    '''
    获取分布概率
    '''
    if key not in d:
        return 0
    else:
        return d[key]

def PrintRules(Rules, count):
    '''
    打印规则
    '''
    print("--------- 打印规则 ---------")
    # global Rules
    temp = 0
    for key,value in Rules.items():
        temp+=1
        if temp<count:
            print('{key}: {value}'.format(key = key, value = value))
            print()
        else:
            break

def DumpRules(Rules, OutputRulesFile):
    '''
    将高阶规则存入文件
    '''
    print('将规则存入文件...')
    with open(OutputRulesFile, 'w') as f:
        total = len(Rules)
        current = 0
        for Source in Rules:
            current += 1
            if current%10000==0:
                print("已保存 %d/%d ..." % (current, total))
            for Target in Rules[Source]:
                # 格式：已有路径 => 下个节点 个数
                # 除了最后的个数，其余总体为高阶依赖路径
                f.write(' '.join([' '.join([str(x) for x in Source]), Target]) + '\n')
        f.close()
    Rules=None
    print("存入完成")