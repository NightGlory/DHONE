# -*- coding:utf-8 -*-

from comparison.ge.walker import RandomWalker
from gensim.models import Word2Vec


class DeepWalk:
    
    def __init__(self, graph, walk_length, num_walks, workers=1):
        '''
        初始化DeepWalk参数。
        @graph：nx.read_edgelist图，通过边表添加
        @walk_length：游走长度（固定）
        @num_walks：游走的次数
        @worker：线程
        '''
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        # walker对象
        self.walker = RandomWalker(
            graph, p=1, q=1, )
        # 路径
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        '''
        @embed_size：嵌入向量维度
        @window_size：词向量上下文最大距离
        @workers：训练模型时使用的线程数。
        @iter：随机梯度下降法中迭代的最大次数，默认是5。
        @**kwargs：超参列表
        :模型训练
        '''
        kwargs["sentences"] = self.sentences                        # 路径
        kwargs["min_count"] = kwargs.get("min_count", 0)            # 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。
        kwargs["vector_size"] = embed_size                          # 向量维度
        kwargs["sg"] = 1                                            # 1 for skip gram, 0 for CBOW
        kwargs["hs"] = 1                                            # deepwalk use 1 for Hierarchical Softmax, otherwise, 0 for Negative Sampling
        kwargs["workers"] = workers                                 # 训练模型时使用的线程数。
        kwargs["window"] = window_size                              # 词向量上下文最大距离
        kwargs["epochs"] = iter                                     # 随机梯度下降法中迭代的最大次数，默认是5。

        print("正在学习嵌入向量...")
        model = Word2Vec(**kwargs)          # 啊这……调包侠
        print("训练完成")

        self.w2v_model = model              # 传入模型
        return model

    def get_embeddings(self,):
        '''
        将训练好的矩阵放入字典中
        '''
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
