# 1. 安装依赖（首次运行需执行）
# !pip install openml scikit-learn pandas numpy matplotlib

# 2. 导入核心库
import openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 3. 从OpenML加载数据集（以经典的成人收入数据集为例，ID=1590）
dataset = openml.datasets.get_dataset(1590)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute
)

# 查看数据集基本信息
print("数据集形状:", X.shape)
print("\n前5行数据:")
print(X.head())
print("\n目标变量分布:")
print(y.value_counts())

# 4. 数据预处理
# 分离数值特征和分类特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 定义预处理管道：数值特征填充缺失值，分类特征编码+填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))  # 数值特征用中位数填充
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 分类特征用众数填充
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 划分训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify保证目标变量分布一致
)

# 6. 构建并训练随机森林模型（集成预处理+模型）
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,  # 决策树数量
        max_depth=10,      # 树最大深度
        random_state=42,
        n_jobs=-1         # 并行计算（使用所有CPU核心）
    ))
])

# 训练模型
print("\n开始训练随机森林模型...")
rf_pipeline.fit(X_train, y_train)

# 7. 模型评估
# 预测测试集
y_pred = rf_pipeline.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)

# 特征重要性可视化（仅数值特征，分类特征编码后维度过多）
numeric_model = rf_pipeline.named_steps['classifier']
numeric_importances = pd.Series(
    numeric_model.feature_importances_[:len(numeric_features)],
    index=numeric_features
).sort_values(ascending=False)

# 绘制数值特征重要性
plt.figure(figsize=(10, 6))
numeric_importances.plot(kind='bar')
plt.title('随机森林 - 数值特征重要性')
plt.ylabel('特征重要性')
plt.xlabel('特征名称')
plt.tight_layout()
plt.show()

# 8. 超参数调优（可选，耗时较长）
print("\n开始超参数调优...")
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# 网格搜索+交叉验证
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 训练调优模型
grid_search.fit(X_train, y_train)

# 输出最优参数和最优分数
print(f"\n最优参数: {grid_search.best_params_}")
print(f"最优交叉验证分数: {grid_search.best_score_:.4f}")

# 用最优模型评估
best_model = grid_search.best_estimator_
y_best_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_best_pred)
print(f"调优后模型测试集准确率: {best_accuracy:.4f}")