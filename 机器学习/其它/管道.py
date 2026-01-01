import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.linear_model import LogisticRegression  # 分类模型
from sklearn.pipeline import Pipeline  # 核心管道类

# 1. 加载数据并划分训练/测试集
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 定义管道：步骤为「标准化 → 逻辑回归」
# 格式：[(步骤名1, 步骤对象1), (步骤名2, 步骤对象2), ...]
pipe = Pipeline([
    ("scaler", StandardScaler()),  # 第一步：数据标准化（特征缩放）
    ("classifier", LogisticRegression(random_state=42))  # 第二步：分类模型
])

# 3. 训练管道（一步执行所有步骤）
pipe.fit(X_train, y_train)

# 4. 评估模型
train_score = pipe.score(X_train, y_train)
test_score = pipe.score(X_test, y_test)
cv_score = cross_val_score(pipe, X, y, cv=5).mean()  # 5折交叉验证

print(f"训练集准确率：{train_score:.4f}")
print(f"测试集准确率：{test_score:.4f}")
print(f"5折交叉验证准确率：{cv_score:.4f}")

# 5. 用管道预测新数据
new_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]])
pred = pipe.predict(new_data)
print(f"新数据预测结果：{[iris.target_names[p] for p in pred]}")