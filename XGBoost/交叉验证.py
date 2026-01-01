import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集（回归任务）
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 转换为XGBoost专用的DMatrix格式（提升效率）
dtrain = xgb.DMatrix(X, label=y)
# 定义参数（回归任务）
params = {
    'objective': 'reg:squarederror',  # 回归目标函数（分类用binary:logistic/multi:softmax）
    'max_depth': 3,                  # 树的最大深度
    'learning_rate': 0.1,            # 学习率
    'subsample': 0.8,                # 训练样本采样率
    'colsample_bytree': 0.8,         # 特征采样率
    'eval_metric': 'rmse',           # 评估指标（分类用auc/error）
    'seed': 42                       # 随机种子（保证可复现）
}

# 分类任务参数示例（二分类）
# params = {
#     'objective': 'binary:logistic',
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     'eval_metric': 'auc',
#     'seed': 42
# }
# 执行5折交叉验证
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,  # 迭代次数（树的数量）
    nfold=5,              # 折数
    stratified=False,     # 分类任务设为True（分层抽样），回归设为False
    early_stopping_rounds=10,  # 早停（验证集性能10轮无提升则停止）
    verbose_eval=10,      # 每10轮打印一次结果
    show_stdv=True        # 显示标准差（反映结果稳定性）
)

# 打印CV结果
print("\n交叉验证结果汇总：")
print(cv_results.tail())

# 提取最优结果
best_rmse = cv_results['test-rmse-mean'].min()
best_std = cv_results['test-rmse-std'].loc[cv_results['test-rmse-mean'].idxmin()]
print(f"\n最优验证集RMSE：{best_rmse:.4f} ± {best_std:.4f}")
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

'''
结合sklearn接口
'''
# 初始化XGBoost回归器
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 定义5折交叉验证（分层KFold适用于分类）
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 执行交叉验证（评估指标为负均方误差，越小越好）
scores = cross_val_score(
    estimator=xgb_model,
    X=X,
    y=y,
    cv=kfold,
    scoring='neg_mean_squared_error'  # 分类可用'accuracy'/'roc_auc'
)

# 转换为RMSE
rmse_scores = np.sqrt(-scores)
print(f"各折RMSE：{rmse_scores.round(4)}")
print(f"平均RMSE：{rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

'''
交叉验证 + 超参数调优
'''
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

# 初始化网格搜索（5折CV）
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=1
)

# 执行网格搜索
grid_search.fit(X, y)

# 输出最优参数和结果
print("\n最优参数：", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_rmse = np.sqrt(-grid_search.best_score_)
print(f"最优参数下的平均RMSE：{best_rmse:.4f}")
'''
随机搜索
'''
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定义随机参数分布
param_dist = {
    'max_depth': randint(2, 6),          # 整数随机值：2-5
    'learning_rate': uniform(0.01, 0.3), # 浮点随机值：0.01-0.31
    'n_estimators': randint(50, 200),    # 整数随机值：50-199
    'subsample': uniform(0.6, 0.4)       # 浮点随机值：0.6-1.0
}

# 初始化随机搜索
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,  # 随机采样20组参数
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    random_state=42,
    verbose=1
)

# 执行随机搜索
random_search.fit(X, y)

# 输出结果
print("\n随机搜索最优参数：", random_search.best_params_)
best_rmse_random = np.sqrt(-random_search.best_score_)
print(f"随机搜索最优RMSE：{best_rmse_random:.4f}")