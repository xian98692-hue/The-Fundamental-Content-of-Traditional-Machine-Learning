import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成回归数据集
X, y = datasets.make_regression(n_samples=100, n_features=1, n_informative=3, noise=10, random_state=42)

# 数据划分与标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练线性回归模型
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = linear_reg.predict(X_test_scaled)
print(f"线性回归均方误差: {mean_squared_error(y_test, y_pred):.2f}")
print(f"线性回归系数: {linear_reg.coef_[0]:.2f}, 截距: {linear_reg.intercept_:.2f}")
print('test R^2:{:.2f}'.format(linear_reg.score(X_test_scaled, y_test)))

# 可视化拟合效果
plt.figure(figsize=(8, 6))
plt.scatter(X_train_scaled, y_train, label='训练数据', alpha=0.7)
plt.plot(X_train_scaled, linear_reg.predict(X_train_scaled), color='red', linewidth=2, label='线性回归拟合')
plt.title('线性回归拟合效果')
plt.xlabel('特征')
plt.ylabel('目标值')
plt.legend()
plt.grid(True)
plt.show()