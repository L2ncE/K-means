# %%
import math as mt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.manifold import TSNE


# %%


def euDistance(x, y):  # 欧式距离的平方
    return np.square(((x - y) ** 2).sum())


def Min_Distance(vet, centers):  # 计算每个到所有簇心的距离中最短的那个
    MIN = mt.inf
    for i in centers:
        distance = euDistance(vet, i)
        if distance < MIN:
            MIN = distance
    return MIN


def KMeansPPlus(X, K):
    n = len(X)
    clusters = np.zeros((1, X.shape[1]))
    clusters[0] = X[np.random.randint(0, n)]  # 随机选择一个向量作为中心向量
    D = np.zeros(len(X))
    for k in range(1, K):
        for i, vector in enumerate(X):
            D[i] = Min_Distance(vector, clusters)  # 计算离所有簇心最短的距离
        clusters = np.row_stack((clusters, X[np.argmax(D)]))  # 将距离最远的新簇心压入簇数组

    return clusters


class KMeans:
    times = 0  # loop times

    def __init__(self, data, KK):
        self.K = KK  # 初始化的簇数
        self.X = data  # 导入的数据
        self.N = len(data)  # 数据的量数
        self.dimension = len(data[0])  # 数据的维数
        self.mu = KMeansPPlus(data, KK)  # 初始化选择中心
        self.mu_record = []  # 初始化中心记录（这里不会影响后面的计算，只是为了初始化）
        self.R = np.zeros((self.N, KK))  # 初始化状态 默认 全0
        self.SSEdata = []

    def kmeans(self):
        while True and self.times < 10:
            self.times += 1  # 统计循环次数 可以设置最大迭代次数
            # 以下为根据mu中心向量来更新R的操作，也即是Kmeans的第一步
            for n in range(self.N):
                distance = np.zeros(self.K)
                for k in range(self.K):
                    distance[k] = euDistance(self.X[n], self.mu[k])
                self.R[n, np.argmin(distance)] = 1
            # 一下为根据状态来更新mu的操作，也就是Kmeans的第二步
            for k in range(self.K):
                for j in range(self.dimension):
                    self.mu_record = self.mu
                    self.mu[k, j] = np.sum(
                        self.R[:, k] * self.X[:, j]) / np.sum(self.R[:, k])
            # 由于Python没有do while循环这里使用while-if-break来进行控制，知道与上一次的记录差小于误差值时停止迭代
            if euDistance(self.mu_record, self.mu) <= 0.000001:
                break

    def lower(self, flag=None):  # 利用PCA对数据进行降维，便于可视化
        pca = PCA(n_components=2)
        self.X = pca.fit_transform(self.X)

    def sseValue(self):
        Value = 0
        for i in range(self.N):
            Value += euDistance(self.X[i], self.mu[self.R[i].argmax()])
            # 这里edDistance为欧式距离，标签采用一个N维向量来表示。若x属于第2类则r=[0,1,0,0,0,...]
        return Value

    def show(self):  # 展示函数
        X_range = [np.min(self.X[:, 0] * 1.25), np.max(self.X[:, 0] * 1.25)]
        # X轴的展示范围
        Y_range = [np.min(self.X[:, 1] * 1.25), np.max(self.X[:, 1] * 1.25)]
        # Y轴的展示范围
        for k in range(self.K):
            for j in range(2):
                self.mu[k, j] = np.sum(
                    self.R[:, k] * self.X[:, j]) / np.sum(self.R[:, k])
        # 计算簇心位置 高纬度数据降维之后可能不在中心
        colors = ['red', 'green', 'yellow', 'blue', 'purple',
                  'brown', 'coral', 'fuchsia', 'gray', 'crimson']
        for k in range(self.K):
            plt.plot(self.X[self.R[:, k] == 1, 0], self.X[self.R[:, k] == 1, 1],
                     marker='o', markerfacecolor=colors[k], markeredgecolor='k',
                     markersize=6, alpha=0.5, linestyle='none')
            plt.plot(self.mu[k, 0], self.mu[k, 1],
                     marker='*', markerfacecolor=colors[k],
                     markersize=10, markeredgewidth=1, markeredgecolor='k'
                     )
        plt.xlim(X_range)
        plt.ylim(Y_range)
        plt.grid(True)
        plt.show()

    def show_original(self):
        X_range = [np.min(self.X[:, 0] * 1.25), np.max(self.X[:, 0] * 1.25)]
        Y_range = [np.min(self.X[:, 1] * 1.25), np.max(self.X[:, 1] * 1.25)]

        plt.plot(self.X[:, 0], self.X[:, 1],
                 marker='o', markersize=6, alpha=0.5, linestyle='none'
                 )

        plt.xlim(X_range)
        plt.ylim(Y_range)
        plt.grid(True)
        plt.show()


def SSE(X, K=None):
    data = X
    if K is None:
        K = 10
    test = []
    test.append(None)
    sseValue = [None]
    for k in range(1, K):
        test.append(KMeans(data, k))
    for k in range(1, K):
        test[k].kmeans()
        sseValue.append(test[k].sseValue())
    X_Range = [x + 1 for x in range(K)]
    ssePicture = pd.DataFrame(sseValue, index=X_Range, columns=['SSE'])
    ssePicture.plot()
    plt.show()


# %%

def iris():
    tsne = TSNE()
    test_iris = datasets.load_iris()
    data_iris = test_iris.data
    SSE(data_iris)

    test1 = KMeans(data_iris, 3)
    test1.kmeans()
    # test1.lower()
    test1.X = tsne.fit_transform(test1.X)
    test1.show_original()
    test1.show()


def wine():
    tsne = TSNE()
    test_wine = datasets.load_wine()
    data_wine = test_wine.data
    SSE(data_wine)

    wine_test = KMeans(data_wine, 3)
    wine_test.kmeans()
    wine_test.X = tsne.fit_transform(wine_test.X)
    wine_test.show_original()
    wine_test.show()


def winequality_white():
    tsne = TSNE()
    # 导入数据集，指定;做分隔号
    test_winequality_white = pd.read_csv('winequality-white.csv', sep=';')
    # 删除结果标签
    test_winequality_white = test_winequality_white.drop(['quality'], axis=1)
    # 取值
    data_winequality = test_winequality_white.values

    # SSE轮廓系数
    SSE(data_winequality)
    kmeans_wineq = KMeans(data_winequality, 10)
    kmeans_wineq.kmeans()
    # kmeans_wineq.X = tsne.fit_transform(kmeans_wineq.X)
    pca = PCA(n_components=2)
    kmeans_wineq.X = pca.fit_transform(kmeans_wineq.X)
    kmeans_wineq.show()


def winequality_red():
    tsne = TSNE()
    # 导入数据集，指定;做分隔号
    test_winequality_red = pd.read_csv('winequality-red.csv', sep=';')
    # 删除结果标签
    test_winequality_red = test_winequality_red.drop(['quality'], axis=1)
    # 取值
    data_winequality = test_winequality_red.values
    # SSE轮廓系数
    SSE(data_winequality)
    kmeans_wineq = KMeans(data_winequality, 7)
    kmeans_wineq.kmeans()
    pca = PCA(n_components=2)
    kmeans_wineq.X = pca.fit_transform(kmeans_wineq.X)
    kmeans_wineq.show()


def absent_time():
    data_ab = pd.read_csv('Absenteeism_at_work.csv', sep=';')
    abtime = data_ab['Absenteeism time in hours']
    abtimevalue = pd.DataFrame(abtime.value_counts())
    abtimevalue = abtimevalue.reindex(index=sorted(abtimevalue.index))
    abtimevalue.plot.bar()
    data_ab_droped = data_ab.drop('Absenteeism time in hours', axis=1)
    data_ab_droped = data_ab_droped.values
    SSE(data_ab_droped)
    pca = PCA(n_components=2)
    kmeans_ab = KMeans(data_ab_droped, 4)
    kmeans_ab.kmeans()
    kmeans_ab.X = pca.fit_transform(kmeans_ab.X)
    kmeans_ab.show()


def main():
    iris()
    wine()
    winequality_white()
    winequality_red()
    absent_time()


if __name__ == "__main__":
    main()
