from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer  # 缺失值填充
from sklearn.preprocessing import PolynomialFeatures  # 多项式特征
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline

# 1. 加载数据（模拟缺失值）
wine = load_wine()
X, y = wine.data, wine.target
# 随机插入10%的缺失值（模拟真实场景）
np.random.seed(42)
mask = np.random.rand(*X.shape) < 0.1
X[mask] = np.nan

# 2. 定义管道（多步特征工程 + 模型）
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # 步骤1：填充缺失值
    ("poly", PolynomialFeatures(degree=2)),  # 步骤2：多项式特征（扩展特征）
    ("rf", RandomForestClassifier(random_state=42))  # 步骤3：随机森林
])

# 3. 定义超参数网格（指定管道中步骤的参数）
# 格式：管道步骤名__参数名
param_grid = {
    "imputer__strategy": ["mean", "median"],  # 缺失值填充策略
    "poly__degree": [1, 2, 3],  # 多项式特征的阶数
    "rf__n_estimators": [50, 100, 200],  # 随机森林树的数量
    "rf__max_depth": [None, 5, 10]  # 随机森林树的最大深度
}

# 4. 网格搜索（交叉验证+超参数调优）
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# 5. 输出最优结果
print(f"最优超参数：{grid_search.best_params_}")
print(f"最优交叉验证准确率：{grid_search.best_score_:.4f}")
print(f"测试集准确率（最优模型）：{grid_search.score(X, y):.4f}")