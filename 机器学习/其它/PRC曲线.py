import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据准备（二分类不平衡数据） =====================
# 生成数据：正样本占比20%，模拟真实场景
X, y = make_classification(
    n_samples=1000, n_classes=2, weights=[0.8, 0.2], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===================== 2. 训练模型并获取正类评分（概率/分数） =====================
# 训练逻辑回归（输出概率）
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)[:, 1]  # 取正类概率作为评分

# ===================== 3. 计算PRC曲线核心指标 =====================
# 计算每个阈值对应的精确率、召回率
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# 计算PRC曲线下面积（AUC-PR）
auc_pr = auc(recall, precision)

# ===================== 4. 绘制PRC曲线 =====================
plt.figure(figsize=(8, 6))

# 绘制PRC曲线
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'PRC Curve (AUC-PR = {auc_pr:.3f})')

# 绘制随机猜测基线（正样本占比）
pos_ratio = np.sum(y_test) / len(y_test)
plt.plot([0, 1], [pos_ratio, pos_ratio], color='navy', lw=2, linestyle='--',
         label=f'Random Guess (pos ratio = {pos_ratio:.3f})')

# 图表美化与标注
plt.xlabel('Recall (查全率)', fontsize=12)
plt.ylabel('Precision (查准率)', fontsize=12)
plt.title('Precision-Recall Curve (PRC)', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.xlim([0.0, 1.0])  # 横轴范围0-1
plt.ylim([0.0, 1.05]) # 纵轴范围0-1.05（避免曲线贴顶）
plt.grid(alpha=0.3)
plt.show()