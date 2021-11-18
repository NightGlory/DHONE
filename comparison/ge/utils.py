from networkx.classes import graph
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

def preprocess_nxgraph(graph):
    '''
    获取节点的字典和列表
    '''
    node2idx = {}                       # 字典{key=节点:value=索引}
    idx2node = []                       # list[node]，index是索引
    node_size = 0
    for node in graph.nodes():          # 遍历节点
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]

def read_graph(edge_path):
    """
    方法读取图并创建目标矩阵。
    :param edge_path: 到边列表的路径。
    :return A: 目标矩阵。
    """
    edges = pd.read_table(edge_path, header=None, sep= " ").values.tolist()
    A = normalize_adjacency(edges)
    graph = nx.from_edgelist(edges)
    return A, graph.number_of_nodes()

def normalize_adjacency(edges):
    """
    计算稀疏度归一化邻接矩阵的方法。——稀疏矩阵解法
    :param edges: 图的边列表。
    :return A: 规范化邻接矩阵。
    """
    # 从边列表创建一个逆矩阵。
    D_1 = create_inverse_degree_matrix(edges)

    index_1 = [int(edge[0]) for edge in edges] + [int(edge[1]) for edge in edges]           # 因为是无向图，a->b肯定有b->a。所以先拼接起来
    index_2 = [int(edge[1]) for edge in edges] + [int(edge[0]) for edge in edges]           # 同上
    values = [1.0 for edge in edges] + [1.0 for edge in edges]
    
    A = sparse.coo_matrix((values, (index_1, index_2)),
                          shape=D_1.shape, dtype=np.float32)
    A = A.dot(D_1)
    return A

def create_inverse_degree_matrix(edges):
    """
    从边列表创建一个逆矩阵。
    :param edges: 边列表.
    :return D_1: 逆度矩阵.
    """
    graph = nx.from_edgelist(edges)
    ind = range(len(graph.nodes()))
    
    # degs = [1.0/graph.degree(node) for node in range(1, graph.number_of_nodes()+1)]
    degs = [1.0/graph.degree(node) for node in range(graph.number_of_nodes())]

    print(len(degs))

    D_1 = sparse.coo_matrix((degs, (ind, ind)),             # 稀疏矩阵
                            shape=(graph.number_of_nodes(),
                            graph.number_of_nodes()),
                            dtype=np.float32)
    return D_1
