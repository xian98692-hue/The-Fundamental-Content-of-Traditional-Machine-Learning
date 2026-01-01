import xgboost as xgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=42)
# 修正特征名称获取方式
feature_names = cancer.feature_names.tolist()

xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
xgb_test = xgb.DMatrix(X_test, label=y_test,feature_names=feature_names)

params = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 5,
}
num_round = 50
evals = [(xgb_train, 'train'), (xgb_test, 'val')]
bst = xgb.train(params=params, dtrain=xgb_train, num_boost_round=num_round, evals=evals,verbose_eval=1)

importance = bst.get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

df = pd.DataFrame(importance, columns=['feature', 'gain'])
print(df[:10])
df['gain'] = df['gain'] / df['gain'].sum()
print(df[:10])
xgb.plot_importance(bst, height=0.5)
plt.show()