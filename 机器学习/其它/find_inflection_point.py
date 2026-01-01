import numpy as np
'''
k_dist_sorted 是升序排列的一维 k - 距离数组，
其本质是 “数据集所有样本的第 k 个最近邻距离按从小到大排序的结果”
自动找拐点
时间复杂度为 
O(n)
'''
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
        max_curvature_idx = np.argmax(curvature)
        # 修正索引（两次diff后长度减2，需还原）
        inflection_idx = max_curvature_idx + 2

        # 兜底逻辑：若曲率无明显峰值，找斜率超过均值2倍的第一个点
    slope_mean = np.mean(slope)#slope是斜率
    slope_threshold = slope_mean * 2
    over_threshold_idx = np.where(slope > slope_threshold)[0]#后缀 [0] 是因为 np.where 返回的是元组（即使只有一维），取第一个元素才能拿到索引数组
    if len(over_threshold_idx) > 0 and over_threshold_idx[0] < inflection_idx:
        inflection_idx = over_threshold_idx[0] + 1  # 还原索引，slope 是 k_dist_sorted 差分后的结果，长度减 1

    # 步骤4：计算最优ε
    optimal_epsilon = k_dist_sorted[inflection_idx]

    return inflection_idx, optimal_epsilon

