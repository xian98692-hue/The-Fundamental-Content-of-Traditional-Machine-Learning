import numpy as np
'''
对于待预测的样本，先找到训练集中与它最相似的K个样本（K近邻），
再对这K个样本的连续型标签取平均值，作为该样本的回归预测结果。
'''
class KNNRegressor:
    def __init__(self, k=3):
        """
        初始化K近邻回归器
        k: 近邻个数，默认3
        """
        self.k = k
        self.X_train = None  # 训练特征
        self.y_train = None  # 训练标签

    def fit(self, X_train, y_train):
        """
        训练阶段：仅存储训练数据（KNN是惰性学习，无训练过程）
        :param X_train: 训练特征数组，形状为 (n_samples, n_features)
        :param y_train: 训练标签数组，形状为 (n_samples,)
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """
        预测阶段：对每个测试样本计算K近邻并回归
        :param X_test: 测试特征数组，形状为 (m_samples, n_features)
        :return: 预测结果数组，形状为 (m_samples,)
        """
        X_test = np.array(X_test)
        # 初始化预测结果数组
        y_pred = np.zeros(X_test.shape[0])#shape[0]获取数组第一个维度的长度

        # 遍历每个测试样本
        for i, x in enumerate(X_test):#遍历二维数组按行处理得到一维数组，i是样本的索引
            # 1. 计算当前测试样本与所有训练样本的欧氏距离
            # 欧氏距离：√(Σ(xi - yi)²)，用numpy广播加速计算
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))#按列计算，得到距离矩阵

            # 2. 按距离升序排序，取前K个近邻的索引
            k_indices = np.argsort(distances)[:self.k]

            # 3. 取K个近邻的标签，计算平均值作为预测值
            k_neighbors_y = self.y_train[k_indices]
            y_pred[i] = np.mean(k_neighbors_y)

        return y_pred

# ------------------- 简单使用示例 -------------------
if __name__ == "__main__":
    # 构造训练数据（特征：二维，标签：一维）
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    y_train = np.array([2.5, 3.5, 4.5, 5.5, 6.5])

    # 初始化KNN回归器（K=2）
    knn = KNNRegressor(k=2)
    # 训练（仅存储数据）
    knn.fit(X_train, y_train)

    # 构造测试数据
    X_test = np.array([[2.5, 3.5], [4.5, 5.5]])
    # 预测
    y_pred = knn.predict(X_test)

    # 输出结果
    print("测试样本特征：")
    print(X_test)
    print("\nK=2时的预测结果：")
    print(y_pred)