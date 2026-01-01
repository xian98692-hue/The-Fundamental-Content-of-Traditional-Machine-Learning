from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch  # 画树状图

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据准备（标准化，避免量纲影响距离计算）
data = load_iris()
X = data.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 初始化凝聚聚类模型
# 关键参数：n_clusters（目标簇数）、linkage（簇间距离方式）
agg_clust = AgglomerativeClustering(
    n_clusters=2,  # 目标簇数（和KMeans一致）
    linkage='ward'  # 默认ward，适合紧凑数据
)
'''
方法	            核心目标	                            输出结果	                            典型适用算法
fit_transform	先拟合数据规律，再将数据 “变换” 到新空间	变换后的特征矩阵（和输入同维度 / 降维）	标准化（Scaler）、PCA、TF-IDF
fit_predict	    先拟合数据规律，再为每个样本分配类别标签	一维标签数组（长度 = 样本数）	        所有聚类算法（KMeans、DBSCAN、凝聚聚类）
二者均只用于训练集
关键区别：为什么聚类不用fit_transform？
聚类的核心是 “给样本分簇”，而非 “改造特征”：
fit_transform的核心是特征空间的变换（比如 PCA 把高维数据降到低维，Scaler 把数据缩放到 0-1），输出是 “新特征”；
fit_predict的核心是样本的类别分配，输出是 “标签”，特征本身不会被改造（聚类算法的输入特征还是原始 / 预处理后的特征）。
'''
# 3. 训练+预测（凝聚聚类无fit_transform，只有fit_predict）
labels = agg_clust.fit_predict(X_scaled)

# 4. 评估聚类效果（整体轮廓系数）
sil_score = silhouette_score(X_scaled, labels)
print(f"凝聚聚类轮廓系数: {sil_score:.3f}")
# 5. 可视化：树状图（ dendrogram）——凝聚聚类的核心优势
plt.figure(figsize=(10, 6))
# 生成层次聚类的链接矩阵（用于画树状图）
linkage_matrix = sch.linkage(X_scaled, method='ward')
# 画树状图（截断显示前20个样本，避免过密）
sch.dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
plt.title('凝聚聚类树状图')
plt.xlabel('样本数')
plt.ylabel('簇间距离')
plt.show()