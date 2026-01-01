from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
#这类模拟数据通常会刻意设计 “特征可线性区分正负类”，逻辑回归（线性模型）无需复杂调参就能拟合得很好
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7,0.3],random_state=42)
'''
定义 “超参数网格”（要尝试的参数组合）；
对每个参数组合，用交叉验证评估性能；
选择交叉验证得分最高的参数组合作为最优参数。
'''
# 1. 定义超参数网格（要尝试的参数组合）
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # 正则化强度（越小正则化越强）
    'solver': ['lbfgs', 'saga'],  # 求解器
    'penalty': ['l2', 'l1']  # 正则化方式
}

# 2. 网格搜索（结合5折分层交叉验证）
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),  # 交叉验证策略
    scoring='roc_auc_ovr',  # 评估指标
    n_jobs=1  # 并行计算（利用所有CPU核心）
)

# 3. 执行网格搜索
grid_search.fit(X, y)

# 4. 输出结果
print(f"最优参数组合：{grid_search.best_params_}")
print(f"最优参数的交叉验证得分：{grid_search.best_score_:.3f}")
print(f"所有参数组合的得分：")
for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    print(f"参数：{params} → 得分：{score:.3f}")