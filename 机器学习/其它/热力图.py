import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
plt.rcParams['font.sans-serif'] = ['SimHei']
# 1. 准备结构化数据
data = {
    "职业": ["学生", "教师", "程序员", "教师", "程序员", "学生"],
    "城市": ["北京", "上海", "广州", "北京", "上海", "广州"],
    "消费等级": [1, 2, 3, 2, 3, 1]
}
df = pd.DataFrame(data)
print("原始结构化数据:\n", df)

# 2. 对类别特征做独热编码，生成数值特征矩阵
# 提取类别特征
cat_features = df[["职业", "城市"]]
# 初始化独热编码器
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded_features = ohe.fit_transform(cat_features)
# 获取编码后的特征名称
encoded_feature_names = ohe.get_feature_names_out(["职业", "城市"])
# 合并数值特征（消费等级）
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoded_feature_names
)
encoded_df["消费等级"] = df["消费等级"].to_numpy()

# 最终特征矩阵
feature_matrix = encoded_df.to_numpy()
feature_names = encoded_df.columns.tolist()
sample_labels = [f"用户{i+1}" for i in range(len(df))]

# 3. 绘制特征矩阵热图
plt.figure(figsize=(10, 6))
sns.heatmap(
    data=feature_matrix,          # 核心：编码后的二维数值矩阵
    annot=True,                   # 显示单元格具体数值
    fmt=".0f",                    # 整数格式（编码值为0/1，消费等级为整数）
    cmap="RdYlGn",                # 红绿渐变配色
    xticklabels=feature_names,    # x轴：特征名称
    yticklabels=sample_labels,    # y轴：样本标签
    cbar_kws={"label": "特征值"},  # 颜色条标签
    linewidths=0.5                # 单元格边框，提升区分度
)

# 4. 图表美化
plt.title("结构化数据特征矩阵热图", fontsize=14, pad=20)
plt.xlabel("特征名称", fontsize=12)
plt.ylabel("样本", fontsize=12)
plt.xticks(rotation=30, ha="right")  # 旋转x轴标签防重叠
plt.tight_layout()
plt.show()