from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 加载数据并训练模型
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)
tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
print("训练集的准确率：{:.3f}".format(tree.score(X_train, y_train)))
print("测试集的准确率：{:.3f}".format(tree.score(X_test, y_test)))
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(range(len(cancer.feature_names)), tree.feature_importances_)
plt.yticks(range(len(cancer.feature_names)), cancer.feature_names)
plt.show()
