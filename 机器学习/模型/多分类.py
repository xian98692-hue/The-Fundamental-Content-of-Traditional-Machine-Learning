import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
'''
OneVsRestClassifier（也叫一对其余分类器）是 sklearn 中处理多分类 / 多标签任务的核心工具，本质是将多分类问题拆解为多个二分类问题来解决
'''
# 1. 加载数据
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# 2. 兼容所有版本的Softmax回归写法
try:
    # 高版本：显式指定multi_class='multinomial'（Softmax）
    softmax_clf = LogisticRegression(
        multi_class='multinomial',  # 强制Softmax
        solver='lbfgs',             # 必须配lbfgs/sag/saga求解器
        random_state=42
    )
except TypeError:
    # 低版本：无multi_class参数，用OneVsRest+逻辑回归（效果接近Softmax）
    from sklearn.multiclass import OneVsRestClassifier
    softmax_clf =LogisticRegression(solver='lbfgs', random_state=42)

# 3. 训练+预测
softmax_clf.fit(X_iris_train, y_iris_train)
y_iris_pred = softmax_clf.predict(X_iris_test)

# 4. 评估
print("Softmax回归准确率：", accuracy_score(y_iris_test, y_iris_pred))
# 查看第一个测试样本的类别概率（Softmax会输出所有类别的概率，且和为1）
print("类别概率分布：", softmax_clf.predict_proba(X_iris_test[:1]))