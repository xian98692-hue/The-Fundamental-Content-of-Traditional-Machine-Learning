import os
# 设置OMP_NUM_THREADS为2（或1，也可避免泄漏）
# 避免多线程导致数据泄漏，不需要MKL库
os.environ["OMP_NUM_THREADS"] = "1"
'''
只能处理球形簇
针对原生 KMeans 的缺点，衍生出多个实用变种，按需选择：
变种	            核心改进	                                                适用场景
MiniBatchKMeans	用 “迷你批次样本” 代替全量样本迭代，速度提升 10 倍 +	        大数据集（n>10 万）、实时流数据
KMeans++	    初始化时让聚类中心尽可能远离，避免局部最优	                    所有场景（原生 KMeans 已默认用）
Bisecting KMeans 二分 KMeans：先聚成 2 簇，再对每个簇递归二分，直到 K 个簇	    避免原生 KMeans 的局部最优，聚类效果更稳定
Kernel KMeans	用核函数（如 RBF）将数据映射到高维，处理非球形簇	                非球形簇、复杂分布数据
'''

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 生成模拟数据（模拟制造业设备的两个特征：转速、温度）
np.random.seed(42)  # 固定随机种子，保证结果可复现
# 簇1：低转速、低温度
cluster1 = np.random.normal(loc=[100, 30], scale=[10, 3], size=(100, 2))
# 簇2：中转速、中温度
cluster2 = np.random.normal(loc=[200, 50], scale=[15, 4], size=(150, 2))
# 簇3：高转速、高温度
cluster3 = np.random.normal(loc=[300, 70], scale=[12, 5], size=(120, 2))
# 合并数据
data = np.vstack([cluster1, cluster2, cluster3])

# 3. 数据标准化（KMeans对特征尺度敏感，标准化后聚类效果更好）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 4. 构建KMeans模型（指定聚类数k=3）
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # n_init避免初始中心的随机性
labels = kmeans.fit_predict(data_scaled)
centers = kmeans.cluster_centers_  # 获取聚类中心

# 5. 模型评估（轮廓系数，取值[-1,1]，越接近1聚类效果越好）
sil_score = silhouette_score(data_scaled, labels)
print(f"聚类轮廓系数: {sil_score:.3f}")

# 6. 结果可视化
plt.figure(figsize=(10, 6))
# 绘制数据点
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
# 绘制聚类中心（反标准化回原始尺度）
centers_original = scaler.inverse_transform(centers)
plt.scatter(centers_original[:, 0], centers_original[:, 1],
            marker='*', s=200, c='red', label='聚类中心')
plt.xlabel('设备转速 (r/min)')
plt.ylabel('设备温度 (°C)')
plt.title('制造业设备运行参数 KMeans 聚类结果')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()