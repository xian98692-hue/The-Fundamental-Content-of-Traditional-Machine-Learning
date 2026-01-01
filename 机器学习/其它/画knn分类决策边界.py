import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
data=pd.read_csv('../../python数据分析活用pandas库/data/Iris.csv')
X=data.iloc[:,1:3].to_numpy()  #二维特征
y=data.iloc[:,-1].to_numpy()    #最后一列一维标签
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,4))
for i,ax in zip([1,2,3],axes):
    clf=KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(clf,X_train, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train,ax=ax)
    ax.set_title("KNN with k={}".format(i))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    axes[0].legend(loc=3)
plt.show()
