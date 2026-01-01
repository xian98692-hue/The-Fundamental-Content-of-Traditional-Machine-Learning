import os
# 避免多线程数据泄漏
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成模拟制造业设备数据（转速、温度 二维特征）
np.random.seed(42)
# 3类工况：稳定工况、轻度异常、重度异常
cluster_stable = np.random.normal(loc=[150, 40], scale=[8, 2], size=(120, 2))
cluster_mild = np.random.normal(loc=[220, 55], scale=[12, 4], size=(100, 2))
cluster_severe = np.random.normal(loc=[280, 75], scale=[15, 5], size=(80, 2))
data = np.vstack([cluster_stable, cluster_mild, cluster_severe])

# 2. 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. 肘部法则：计算不同k的WCSS
wcss_list = []  # 存储不同k的簇内平方和
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    wcss_list.append(kmeans.inertia_)  # inertia_ 对应WCSS


# 4. 自动找肘部法则曲线的拐点（核心函数）
def find_elbow_point(k_range, wcss_list):
    """
    自动识别肘部法则曲线的拐点
    参数：
        k_range: k的取值列表（如range(1,11)）
        wcss_list: 对应k的WCSS值列表
    返回：
        best_k: 最优簇数（拐点对应的k）
        elbow_idx: 拐点在k_range中的索引
    """
    # 转换为numpy数组便于计算
    k_array = np.array(list(k_range))
    wcss_array = np.array(wcss_list)

    # 步骤1：计算WCSS的一阶导数（下降速率）
    # 导数为负表示WCSS下降，取绝对值便于分析
    wcss_deriv = np.abs(np.diff(wcss_array))

    # 步骤2：计算导数的变化率（二阶导数），反映下降速率的衰减程度
    deriv_change = np.diff(wcss_deriv)

    # 步骤3：找导数变化率的突变点（拐点）
    # 突变点：导数变化率由"大幅下降"转为"小幅变化"，取最大值对应的点
    if len(deriv_change) == 0:
        elbow_idx = 0  # 兜底：无变化则取第一个点
    else:
        # 导数变化率最大值对应"下降速率突然变慢"的位置
        max_change_idx = np.argmax(deriv_change)
        elbow_idx = max_change_idx + 2  # 修正diff导致的索引偏移

    # 步骤4：边界处理（避免索引越界）
    elbow_idx = np.clip(elbow_idx, 0, len(k_array) - 1)
    best_k = k_array[elbow_idx]

    # 兜底验证：若拐点k=1，结合轮廓系数选次优k
    if best_k == 1 and len(k_range) > 2:
        # 计算k>=2的轮廓系数，选最优
        sil_scores = []
        for k in k_range:
            if k < 2:
                sil_scores.append(-1)
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            sil = silhouette_score(data_scaled, labels)
            sil_scores.append(sil)
        best_k = k_array[np.argmax(sil_scores)]

    return best_k, elbow_idx


# 执行自动找拐点
best_k, elbow_idx = find_elbow_point(k_range, wcss_list)
print(f"自动识别的最优簇数 k = {best_k}")

# 5. 绘制肘部法则曲线（标注拐点）
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 绘制WCSS曲线
plt.plot(k_range, wcss_list, marker='o', linestyle='-', color='blue', linewidth=2)
# 标注拐点
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'拐点 k={best_k}')
plt.scatter(best_k, wcss_list[elbow_idx], color='red', s=100, zorder=5)
# 图表配置
plt.xlabel('簇数 k', fontsize=12)
plt.ylabel('簇内平方和 (WCSS)', fontsize=12)
plt.title('肘部法则曲线（自动识别拐点）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(k_range)
plt.legend(fontsize=10)

# 5. 方法2：轮廓系数法（遍历 k=2 到 10，k=1无轮廓系数）
sil_score_list = []
k_sil_range = range(2, 11)
for k in k_sil_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    sil_score = silhouette_score(data_scaled, labels)
    sil_score_list.append(sil_score)

# 绘制轮廓系数曲线
plt.subplot(1, 2, 2)
plt.plot(k_sil_range, sil_score_list, marker='s', linestyle='-', color='orange')
plt.xlabel('簇数 k')
plt.ylabel('平均轮廓系数')
plt.title('轮廓系数曲线（找最大值）')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(k_sil_range)

plt.tight_layout()
plt.show()
'''
肘部法则逻辑：找 “WCSS 下降速率突变” 的点，但如果曲线衰减平缓（如 k=3 后仍缓慢下降），算法会误把后期的小突变当成拐点；
轮廓系数法逻辑：直接衡量簇的紧凑性 / 分离度，对 “真实簇数” 更敏感（你的数据是 3 簇，所以轮廓系数在 k=3 时最高）
优先选轮廓系数法
'''
# 6. 确定最优k并训练最终模型
# 从轮廓系数中找最大值对应的k
best_k = k_sil_range[np.argmax(sil_score_list)]
print(f"通过轮廓系数法确定最优簇数 k = {best_k}")

# 用最优k训练KMeans
kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_best = kmeans_best.fit_predict(data_scaled)
centers_best = kmeans_best.cluster_centers_
centers_best_original = scaler.inverse_transform(centers_best)  # 反标准化

# 7. 绘制最优k的聚类结果
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels_best, cmap='viridis', alpha=0.7)
plt.scatter(centers_best_original[:, 0], centers_best_original[:, 1],
            marker='*', s=300, c='red', label='聚类中心')
plt.xlabel('设备转速 (r/min)')
plt.ylabel('设备温度 (°C)')
plt.title(f'制造业设备工况聚类结果（最优k={best_k}）')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()