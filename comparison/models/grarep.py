"""GraRep model implementation."""

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

class GraRep(object):
    """
    GraRep Model 对象
    GraRep的稀疏感知实现。
    For details see the paper: https://dl.acm.org/citation.cfm?id=2806512
    """
    def __init__(self, A, dimensions, order, iterations, seed):
        """
        :param A: 稀疏邻接矩阵
        :param args: Arguments object.
        """
        self.A = A
        self.dimensions = dimensions
        self.order = order
        self.iterations = iterations
        self.seed = seed
        self._setup_base_target_matrix()

    def _setup_base_target_matrix(self):
        """
        创建要相乘的基矩阵。
        """
        values = [1.0 for _ in range(self.A.shape[0])]
        indices = [i for i in range(self.A.shape[0])]
        self.A_hat = sparse.coo_matrix((values, (indices, indices)),
                                       shape=self.A.shape,
                                       dtype=np.float32)

    def _create_target_matrix(self):
        """
        创建一个对数转换的目标矩阵。
        :return target_matrix: 矩阵分解SVD。
        """
        self.A_hat = sparse.coo_matrix(self.A_hat.dot(self.A))
        scores = np.log(self.A_hat.data)-math.log(self.A.shape[0])
        rows = self.A_hat.row[scores < 0]
        cols = self.A_hat.col[scores < 0]
        scores = scores[scores < 0]
        target_matrix = sparse.coo_matrix((scores, (rows, cols)),
                                          shape=self.A.shape,
                                          dtype=np.float32)
        return target_matrix

    def optimize(self):
        """
        学习嵌入
        """
        print("\n优化开始...\n")
        self.embeddings = []            # 嵌入矩阵
        for step in tqdm(range(self.order)):           # tqdm为进度显示
            target_matrix = self._create_target_matrix()

            svd = TruncatedSVD(n_components=self.dimensions,
                               n_iter=self.iterations,
                               random_state=self.seed)

            svd.fit(target_matrix)
            embedding = svd.transform(target_matrix)
            self.embeddings.append(embedding)

    def save_embedding(self):
        """
        保存嵌入。
        """
        print("\nSave embedding.\n")
        self.embeddings = np.concatenate(self.embeddings, axis=1)
        column_count = self.order*self.dimensions
        columns = ["ID"] + ["x_" + str(col) for col in range(column_count)]
        ids = np.array([i for i in range(self.A.shape[0])]).reshape(-1,1)
        self.embeddings = np.concatenate([ids, self.embeddings], axis=1)
        self.embeddings = pd.DataFrame(self.embeddings, columns=columns)
        self.embeddings.to_csv(self.args.output_path, index=None)
