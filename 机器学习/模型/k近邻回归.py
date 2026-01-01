import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成回归数据集
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# 数据划分与标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练KNN回归器
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = knn_regressor.predict(X_test_scaled)
print('test set R^2:{:.2f}'.format(knn_regressor.score(X_test_scaled, y_test)))
print('test set MSE:{:.2f}'.format(mean_squared_error(y_test, y_pred)))

# 可视化拟合效果
plt.figure(figsize=(8, 6))
X_plot = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 100).reshape(-1, 1)
y_plot = knn_regressor.predict(X_plot)

plt.scatter(X_train_scaled, y_train, label='训练数据', alpha=0.7)
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='KNN回归拟合')
plt.title('K近邻回归拟合效果')
plt.xlabel('特征')
plt.ylabel('目标值')
plt.legend()
plt.grid(True,axis='x')
plt.show()