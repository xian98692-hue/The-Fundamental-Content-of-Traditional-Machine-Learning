import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ===================== 1. 生成制造业设备数据 =====================
np.random.seed(42)
# 正常工况簇1：低转速、低温度
cluster1 = np.random.normal(loc=[150, 40], scale=[8, 2], size=(100, 2))
# 正常工况簇2：中转速、中温度
cluster2 = np.random.normal(loc=[220, 55], scale=[12, 4], size=(80, 2))
# 噪声点（设备异常）
noise = np.random.uniform(low=[300, 80], high=[350, 90], size=(10, 2))
data = np.vstack([cluster1, cluster2, noise])#垂直堆叠
data_scaled = StandardScaler().fit_transform(data)  # 标准化

# ===================== 2. 计算k-距离 =====================
min_pts = 3  # DBSCAN最小点数（维度+1）
nn = NearestNeighbors(n_neighbors=min_pts, metric='euclidean')
nn.fit(data_scaled)
distances, _ = nn.kneighbors(data_scaled)#NearestNeighbors返回的是距离和索引，_ 是 Python 中约定俗成的 “占位符变量”，用来接收不需要使用的返回值
k_dist = distances[:, min_pts - 1]  # 每个样本到第3个近邻的距离,第一个数组（distances）：形状为 [样本数, n_neighbors]，每个样本对应的 K 个最近邻的距离
k_dist_sorted = np.sort(k_dist)  # 升序排序


# ===================== 3. 自动找拐点（核心逻辑） =====================
def find_inflection_point(k_dist_sorted):
    """
    自动识别k-距离图拐点
    返回：拐点索引、最优ε值
    """
    # 步骤1：计算一阶导数（斜率）
    dx = 1  # 横轴步长（样本索引差）
    dy = np.diff(k_dist_sorted)  #一阶导数	np.diff(k_dist_sorted)	计算曲线每一点的上升斜率，斜率越大，曲线越陡
    slope = dy / dx  # 一阶导数（斜率）

    # 步骤2：计算二阶导数（曲率，反映斜率变化率）
    d_slope = np.diff(slope)#计算斜率的变化率，曲率最大值对应 “斜率突然变大” 的位置
    curvature = d_slope / dx

    # 步骤3：找曲率最大的点（核心拐点）
    if len(curvature) == 0:
        inflection_idx = len(k_dist_sorted) // 2  # 兜底：取中间点
    else:
        max_curvature_idx = np.argmax(curvature)#最大曲率索引
        # 修正索引（两次diff后长度减2，需还原）
        inflection_idx = max_curvature_idx + 2#屈折索引

        # 兜底逻辑：若曲率无明显峰值，找斜率超过均值2倍的第一个点
    slope_mean = np.mean(slope)#slope是斜率，mean()是求平均值
    slope_threshold = slope_mean * 2
    over_threshold_idx = np.where(slope > slope_threshold)[0]#后缀 [0] 是因为 np.where 返回的是元组（即使只有一维），取第一个才能拿到数组
    if len(over_threshold_idx) > 0 and over_threshold_idx[0] < inflection_idx:#找斜率超过均值2倍的第一个点,over_threshold_idx超阈值索引
        inflection_idx = over_threshold_idx[0] + 1  # 还原索引，slope 是 k_dist_sorted 差分后的结果，长度减了1

    # 步骤4：计算最优ε
    optimal_epsilon = k_dist_sorted[inflection_idx]

    return inflection_idx, optimal_epsilon


# 执行自动找拐点
inflection_idx, optimal_epsilon = find_inflection_point(k_dist_sorted)
print(f"自动识别的拐点样本索引：{inflection_idx}")
print(f"最优ε值：{optimal_epsilon:.4f}")

# ===================== 4. 可视化拐点 =====================
plt.figure(figsize=(10, 6))
# 绘制k-距离曲线
plt.plot(k_dist_sorted, color='blue', label='k-距离曲线')
# 标注拐点
plt.axvline(x=inflection_idx, color='red', linestyle='--', linewidth=2,
            label=f'拐点（索引{inflection_idx}）')
plt.axhline(y=optimal_epsilon, color='green', linestyle='--', linewidth=2,
            label=f'最优ε={optimal_epsilon:.4f}')
# 标注斜率/曲率辅助信息
plt.scatter(inflection_idx, optimal_epsilon, color='red', s=100, zorder=5)
plt.xlabel('样本（按距离升序排序）', fontsize=12)
plt.ylabel(f'到第{min_pts}个近邻的距离', fontsize=12)
plt.title('k-距离图（自动识别拐点）', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()

# ===================== 5. 验证：用最优ε训练DBSCAN =====================
dbscan = DBSCAN(eps=optimal_epsilon, min_samples=min_pts)
labels = dbscan.fit_predict(data_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nDBSCAN聚类结果：")
print(f"识别簇数：{n_clusters}")
print(f"噪声点（异常）数量：{n_noise}")
print(f"轮廓系数：{silhouette_score(data_scaled, labels):.3f}")

# 可视化DBSCAN结果
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
    class_mask = (labels == k)
    xy = data[class_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'簇{k}' if k != -1 else '噪声（异常）', alpha=0.7)

plt.xlabel('设备转速 (r/min)', fontsize=12)
plt.ylabel('设备温度 (°C)', fontsize=12)
plt.title(f'DBSCAN聚类结果（ε={optimal_epsilon:.4f}）', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()