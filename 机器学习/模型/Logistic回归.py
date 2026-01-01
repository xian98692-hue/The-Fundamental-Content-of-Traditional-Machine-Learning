import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
'''
线性假设：仅能捕捉特征与标签的线性关系，无法处理非线性数据（需结合特征工程，如多项式特征、离散化）。
'''
# ===================== 1. 数据准备 =====================
# 生成模拟二分类数据（特征数=5，正样本占比30%）
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3,
    weights=[0.7, 0.3], random_state=42
)
# 划分训练/测试集（分层抽样保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 特征归一化（加速收敛，逻辑回归对尺度不敏感但归一化更稳定）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 2. 训练逻辑回归模型 =====================
# 初始化模型（关键参数：正则化、求解器、类别权重）
lr = LogisticRegression(
    penalty='l2',  # L2正则化（缓解过拟合），可选L1（需配合solver='saga'）
    C=1.0,         # 正则化强度（C越小，正则化越强）
    solver='lbfgs',# 求解器（lbfgs适合小数据，saga适合大数据/L1正则）
    class_weight='balanced',  # 平衡正负样本权重（适合不平衡数据）
    random_state=42
)
# 训练模型
lr.fit(X_train_scaled, y_train)

# ===================== 3. 模型输出与解读 =====================
# 3.1 核心参数：系数和截距
print("=== 模型参数 ===")
print(f"特征系数(w)：{lr.coef_[0]}")  # 每个特征的权重（5个特征对应5个系数）
print(f"截距(b)：{lr.intercept_[0]}")  # 偏置项

# 3.2 预测结果（类别+概率）
y_pred = lr.predict(X_test_scaled)  # 预测类别（0/1）
y_pred_proba = lr.predict_proba(X_test_scaled)  # 预测概率（[负类概率, 正类概率]）
y_pred_proba_pos = y_pred_proba[:, 1]  # 提取正类概率

print("\n=== 前5个样本预测结果 ===")
for i in range(5):
    print(f"样本{i+1}：真实标签={y_test[i]}, 预测类别={y_pred[i]}, 正类概率={y_pred_proba_pos[i]:.3f}")

# ===================== 4. 模型评估 =====================
print("\n=== 模型评估指标 ===")
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(cm)
# 核心指标
accuracy = accuracy_score(y_test, y_pred)  # 准确率
precision = precision_score(y_test, y_pred)  # 精确率
recall = recall_score(y_test, y_pred)  # 召回率
roc_auc = roc_auc_score(y_test, y_pred_proba_pos)  # ROC-AUC

print(f"准确率：{accuracy:.3f}")
print(f"精确率：{precision:.3f}")
print(f"召回率：{recall:.3f}")
print(f"ROC-AUC：{roc_auc:.3f}")