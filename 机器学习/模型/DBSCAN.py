'''
能聚类任意形状的簇（如环形、不规则形状）
无需指定 k，自动识别簇数
对参数（ε、MinPts）敏感，参数选不好会导致聚类效果极差
MinPts（最小点数）：
经验值：MinPts = 数据维度 + 1（如二维设备参数数据，MinPts=3）
ε（邻域半径）：
方法：绘制 “k - 距离图”（计算每个点到第 MinPts 近邻点的距离，排序后找拐点），拐点对应的距离即为最优 ε
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 2. 生成制造业设备数据（转速、温度）
np.random.seed(42)
# 正常工况簇1：低转速、低温度（核心点）
cluster1 = np.random.normal(loc=[150, 40], scale=[8, 2], size=(100, 2))#正态分布，均值为[150,40]，方差为[8,2]，二维数据
# 正常工况簇2：中转速、中温度（核心点）
cluster2 = np.random.normal(loc=[220, 55], scale=[12, 4], size=(80, 2))
# 异常工况（边界点）
border = np.random.normal(loc=[200, 50], scale=[5, 2], size=(20, 2))
# 噪声点（设备故障极端值）
noise = np.random.uniform(low=[300, 80], high=[350, 90], size=(10, 2))
# 合并数据
data = np.vstack([cluster1, cluster2, border, noise])#行在叠一起
# 标准化（DBSCAN对尺度敏感）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. 调优参数：用k-距离图找最优ε
min_pts = 3  # 数据维度2+1
neigh = NearestNeighbors(n_neighbors=min_pts)
nbrs = neigh.fit(data_scaled)
distances, _ = nbrs.kneighbors(data_scaled)
# 计算每个点到第min_pts近邻的距离，排序
distances = np.sort(distances[:, min_pts-1], axis=0)
# 绘制k-距离图
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('数据点（按距离排序）')
plt.ylabel(f'到第{min_pts}近邻的距离')
plt.title('k-距离图（找拐点）')
plt.grid(True, alpha=0.5)
plt.show()

# 4. 训练DBSCAN（根据k-距离图，选拐点ε=0.3）
dbscan = DBSCAN(eps=0.3, min_samples=min_pts)
labels = dbscan.fit_predict(data_scaled)

# 5. 结果分析
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 噪声标签为-1
n_noise = list(labels).count(-1)
print(f"识别出的簇数：{n_clusters}")
print(f"噪声点（异常）数量：{n_noise}")
sil_score = silhouette_score(data_scaled, labels)
print(f"轮廓系数：{sil_score:.3f}")

# 6. 可视化结果
plt.figure(figsize=(10, 6))
# 绘制簇和噪声
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声点用黑色
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = data[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'簇{k}' if k!=-1 else '噪声（异常）', alpha=0.7)

plt.xlabel('设备转速 (r/min)')
plt.ylabel('设备温度 (°C)')
plt.title('DBSCAN 设备工况聚类+异常检测结果')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()