import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
'''
原生API
'''
# 1. 准备数据（延续之前的糖尿病回归任务）
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
dtrain = xgb.DMatrix(X, label=y)

# 2. 训练模型（含交叉验证确定最优迭代次数）
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'rmse',
    'seed': 42
}

# 先通过CV确定最优n_estimators
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=5,
    early_stopping_rounds=10,
    verbose_eval=False
)
best_rounds = cv_results.shape[0]

# 训练最终模型
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_rounds
)

# 3. 提取特征重要性
# 方式1：原生属性（返回字典）
importance_dict = model.get_score(importance_type='gain')  # 可选 'weight'/'cover'

# 方式2：转换为DataFrame（更易处理）
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

print("特征重要性（Gain）：")
print(importance_df)


'''
scikit-learn接口
'''
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold

# 1. 初始化模型+交叉验证调优
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# 定义参数网格
param_grid = {
    'max_depth': [2, 3],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100]
}

# 5折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X, y)

# 2. 提取最优模型的特征重要性
best_model = grid_search.best_estimator_

# 方式1：直接获取（默认Gain）
importance_vals = best_model.feature_importances_  # 数组，顺序与X.columns一致

# 方式2：转换为DataFrame
importance_df_sk = pd.DataFrame({
    'feature': X.columns,
    'importance': importance_vals
}).sort_values('importance', ascending=False)

print("\nScikit-learn API 特征重要性：")
print(importance_df_sk)
'''
特征重要性可视化
'''
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df_sk['feature'][::-1],  # 逆序显示（从高到低）
    importance_df_sk['importance'][::-1],
    color='skyblue'
)
plt.xlabel('Gain（平均损失减少量）')
plt.ylabel('特征')
plt.title('XGBoost 特征重要性（Gain）')
plt.grid(axis='x', alpha=0.3)
plt.show()