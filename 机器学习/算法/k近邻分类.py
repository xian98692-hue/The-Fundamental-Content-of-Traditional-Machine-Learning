import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        # 初始化K值，默认选择3个近邻
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # 训练过程仅存储训练集数据
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _calculate_distance(self, x):
        # 计算单个样本与所有训练样本的欧氏距离
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _vote(self, distances):
        # 按距离排序，取前k个样本的标签并投票
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        # 返回出现次数最多的标签
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def predict(self, X_test):
        # 对测试集批量预测
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            distances = self._calculate_distance(x)
            pred = self._vote(distances)
            predictions.append(pred)
        return np.array(predictions)

# ---------------- 测试代码 ----------------
if __name__ == "__main__":
    # 构造简单的分类数据集
    X_train = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]]
    y_train = [0, 0, 0, 1, 1]
    X_test = [[2.5, 3.5], [5.5, 6.5]]

    # 初始化并训练KNN模型
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    # 预测测试集
    preds = knn.predict(X_test)
    print("预测结果:", preds)  # 输出: [0 1]