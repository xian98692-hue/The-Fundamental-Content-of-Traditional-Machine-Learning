import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 生成数据
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7,0.3], random_state=42)
model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)

# 分层5折交叉验证（保证每折中类别比例一致）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 计算5轮的AUC-ROC（多分类用ovr策略）
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr')

print(f"5折交叉验证的AUC-ROC：{scores.round(3)}")
print(f"平均AUC-ROC：{np.mean(scores):.3f}（标准差：{np.std(scores):.3f}）")