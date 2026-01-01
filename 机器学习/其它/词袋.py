from sklearn.feature_extraction.text import CountVectorizer
texts=[
    'I love Python',
    'Python is a cool,I love Python'
]
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(texts)
print(vectorizer.vocabulary_)#{词汇：索引}
print('词汇',vectorizer.get_feature_names_out())#词汇表列表
print(X)
print(X.toarray())#转为密集矩阵