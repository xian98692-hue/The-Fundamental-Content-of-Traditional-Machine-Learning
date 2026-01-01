import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. 数据加载与划分
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 定义管道+候选模型超参
# 格式：(管道, 超参网格),l1_ratio=0 + 保留默认 C	仅使用 L2 正则
models = [
    # 逻辑回归
    (Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=10000))]),
     { "clf__l1_ratio": [0.1,0.5,0.7,0.8,1], "clf__solver": ["saga"]}),
    # SVM
    (Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
     {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]}),
    # 随机森林
    (Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())]),
     {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 10]})
]

# 3. 网格搜索找最优模型
best_score = 0
best_model = None

for pipe, params in models:
    grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    # 更新最优模型
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_

# 4. 输出结果
print(f"最优模型：{best_model.named_steps['clf'].__class__.__name__}")
print(f"最优交叉验证得分：{best_score:.4f}")
print(f"测试集得分：{best_model.score(X_test, y_test):.4f}")
print(f"最优超参数：{best_model.get_params()['clf']}")