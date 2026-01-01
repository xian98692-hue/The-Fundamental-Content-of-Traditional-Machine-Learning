import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 生成3分类数据
X, y = make_classification(
    n_samples=1000, n_features=5, n_classes=3,#n_classes=3指定标签有3个类别，所以模型分3个类
    n_informative=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 适配1.5+版本：移除multi_class参数，默认使用multinomial
lr_multi = LogisticRegression(
    solver='saga',       # 必须选支持multinomial的solver（lbfgs/saga/newton-cg）
    max_iter=1000,       # 增加迭代次数确保收敛（多分类易需更多迭代）
    random_state=42      # 固定随机种子
)
lr_multi.fit(X_train, y_train)

# 预测与评估（和之前逻辑完全一致）
y_pred = lr_multi.predict(X_test)
y_pred_proba = lr_multi.predict_proba(X_test)  # 每类概率和为1

print(f"多分类准确率：{accuracy_score(y_test, y_pred):.3f}")
print(f"前3个样本的类别概率：\n{y_pred_proba[:3].round(3)}")
print(f"模型系数形状：{lr_multi.coef_.shape}")  # (n_classes, n_features)，符合multinomial逻辑