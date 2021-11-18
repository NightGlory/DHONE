import numpy as np
import sys
import time
import networkx as nx

from comparison.models import *
from comparison.ge.utils import read_graph
from comparison.ge.index import RebuildIndexes


INDEXDICTFILE = "code/comparison/index.npy"
INDEXEDGEFILE = "code/comparison/index_edge.txt"

# -------------------------start-ups的文件参数---------------------------------------------
# edgeFile = "data/start-ups/fonData.txt"
# embed_size = 200
# order = 7
# iterations = 20

# ------------------------porto的文件参数----------------------------------------------
edgeFile = "data/porto-taxi/result/FonFile.txt"
embed_size = 5
order = 5
iterations = 20

# -------------------------shipping的文件参数---------------------------------------------
# edgeFile = "data/shipping/result/FonFile.txt"
# embed_size = 6
# order = 5
# iterations = 20

# -------------------------facebook的文件参数---------------------------------------------
# edgeFile = "data/facebook/result/FonFile.txt"
# embed_size = 20
# order = 6
# iterations = 20

# -------------------------ucforum---------------------------------------------
# edgeFile = "data/ucforum/fonData.txt"
# embed_size = 25
# order = 5
# iterations = 20


def GetEmbeddings(method, edgeFile):
    embedded_matrix = []
    if method is not "GraRep":
        G = nx.read_edgelist(edgeFile, create_using=nx.DiGraph(), nodetype=None)
        if method is "DeepWalk":
            # 使用DeepWalk
            model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
            model.train(window_size=5, iter=3, embed_size=embed_size)
        elif method is "Node2Vec":
            # 使用Node2Vec
            model = Node2Vec(G, walk_length=10, num_walks=80,
                            p=0.25, q=4, workers=1, use_rejection_sampling=0)
            model.train(window_size = 5, iter = 3, embed_size=embed_size)
        elif method is "LINE":
            # 使用LINE
            model = LINE(G, embedding_size=embed_size, order='second')
            model.train(batch_size=1024, epochs=50, verbose=2)

        embeddings = model.get_embeddings()
        # 转矩阵
        for key in embeddings.keys():
            embedded_matrix.append(embeddings[key])
    else:
        # 使用GraRep
        A, number_of_nodes = read_graph(edgeFile)
        model = GraRep(A, dimensions=embed_size, order=order, iterations = iterations, seed = 42)
        model.optimize()
        # 将矩阵reshape为两维
        embedded_matrix = np.array(model.embeddings).reshape(number_of_nodes, embed_size*order)
    return embedded_matrix


if __name__ == "__main__":
    begin = time.clock()

    # edge file to index file
    dict = RebuildIndexes(edgeFile, INDEXDICTFILE, INDEXEDGEFILE)
    
    embedded_matrix = GetEmbeddings("DeepWalk", INDEXEDGEFILE)

    end = time.clock()
    print("共用时：%.2f 分钟" % ((end-begin)/60))