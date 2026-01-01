from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# ===================== 1. 自定义英文文本数据集 =====================
# 构建小型英文语料库（主题：不同科技产品评论）
corpus = [
    # 文档0：iPhone 相关（关键词：iPhone, camera, battery, screen）
    "The iPhone has a great camera and long battery life, the screen is clear",
    # 文档1：Samsung 相关（关键词：Samsung, display, battery, performance）
    "Samsung phone has amazing display and good battery performance",
    # 文档2：Laptop 相关（关键词：Laptop, keyboard, battery, performance）
    "This laptop has a comfortable keyboard and strong battery performance",
    # 文档3：iPhone 补充（关键词：iPhone, camera, screen, design）
    "iPhone camera quality is top notch, screen design is elegant",
    # 文档4：Laptop 补充（关键词：Laptop, keyboard, speed, performance）
    "Laptop keyboard is responsive and speed performance is excellent"
]

# 给每个文档命名（方便后续解析）
doc_names = ["iPhone_Review_1", "Samsung_Review", "Laptop_Review_1", "iPhone_Review_2", "Laptop_Review_2"]

# ===================== 2. 初始化 TF-IDF 向量化器 =====================
# 配置：
# - stop_words='english'：移除英文停用词（the/a/has 等无意义词汇）
# - lowercase=True：全部转为小写
# - max_features=20：只保留Top20高频关键词（简化输出）
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=20
)

# ===================== 3. 计算 TF-IDF 矩阵 =====================
# fit_transform：拟合语料库 + 转换为TF-IDF矩阵
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# 获取特征词列表（所有被保留的关键词）
feature_names = tfidf_vectorizer.get_feature_names_out()

# 转换为DataFrame（方便可视化）
tfidf_df = pd.DataFrame(
    data=tfidf_matrix.toarray(),  # 稀疏矩阵转密集数组
    index=doc_names,              # 行：文档名
    columns=feature_names         # 列：关键词
)

# ===================== 4. 输出结果 =====================
print("="*50)
print("1. 所有关键词（Feature Names）：")
print(feature_names)
print("\n" + "="*50)
print("2. TF-IDF 矩阵（数值越高，关键词对文档越重要）：")
print(tfidf_df.round(3))  # 保留3位小数，简化显示

# ===================== 5. 关键结果解析 =====================
print("\n" + "="*50)
print("3. 核心结论解析：")
# 示例1：查看 iPhone 相关文档的核心关键词
iphone_tfidf = tfidf_df.loc[["iPhone_Review_1", "iPhone_Review_2"]]
print("\n- iPhone 文档核心关键词（TF-IDF 最高）：")
print(iphone_tfidf.idxmax(axis=1))  # 每行（文档）TF-IDF最高的关键词

# 示例2：查看 battery 词的 TF-IDF 分布（跨文档重要性）
battery_tfidf = tfidf_df["battery"]
print("\n- 'battery' 在各文档的 TF-IDF 值（值越低说明越通用）：")
print(battery_tfidf.round(3))

# 示例3：查看 laptop 专属关键词
laptop_tfidf = tfidf_df.loc[["Laptop_Review_1", "Laptop_Review_2"]]
laptop_top_words = laptop_tfidf.max().sort_values(ascending=False).head(3)
print("\n- Laptop 文档Top3关键词：")
print(laptop_top_words.round(3))