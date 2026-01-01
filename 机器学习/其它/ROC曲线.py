import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
'''
维度	    ROC 曲线	                PRC 曲线
核心关注	模型整体区分能力（正负样本）	模型对正类的识别质量
横轴/纵轴	FPR/TPR	                Recall / Precision
随机基线	对角线（0.5）	            正样本占比（如 1%）
类别不平衡  鲁棒性强  	            敏感（更真实）
适用场景	正负样本均衡，关注整体区分	正负样本失衡 / 关注正类识别
'''
# 1. 生成数据（不平衡二分类）
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
model = LogisticRegression().fit(X, y)
y_score = model.predict_proba(X)[:, 1]

# 2. 计算ROC指标
fpr, tpr, thresholds = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)

# 3. 绘制ROC曲线
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')  # 随机基线
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR/Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()