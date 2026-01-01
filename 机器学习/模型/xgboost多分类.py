import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
汉明损失（Hamming Loss）是多标签分类任务中核心的评估指标，用来衡量模型预测结果与真实标签的「不一致程度」
汉明损失越小，模型在多标签任务上的表现越好（最小值 0，最大值 1）
'''
# 加载数据集并划分
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost多分类器（互斥多分类）
xgb_multi_clf = xgb.XGBClassifier(
    objective='multi:softmax',  # 直接输出类别
    num_class=3,  # 类别数
    random_state=42
)
xgb_multi_clf.fit(X_train, y_train)

# 预测与评估
y_pred = xgb_multi_clf.predict(X_test)
print("XGBoost互斥多分类准确率：", accuracy_score(y_test, y_pred))
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss

# 构造多标签数据集（类别非互斥）
np.random.seed(42)
X_multi = np.random.randn(100, 5)
y_multi = np.random.randint(0, 2, (100, 3))  # 3个类别，样本可同时属于多个
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# 用One-vs-Rest包装XGBoost处理非互斥多分类
xgb_ovr_clf = OneVsRestClassifier(xgb.XGBClassifier(objective='binary:logistic', random_state=42))
xgb_ovr_clf.fit(X_train_multi, y_train_multi)

# 预测与评估（汉明损失）
y_multi_pred = xgb_ovr_clf.predict(X_test_multi)
print("XGBoost非互斥多分类汉明损失：", hamming_loss(y_test_multi, y_multi_pred))