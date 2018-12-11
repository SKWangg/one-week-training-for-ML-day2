# 加载库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 读取数据
data_all = pd.read_csv('data_all.csv', encoding='gbk')

# 划分数据集
x = data_all.drop(columns=["status"]).as_matrix()
y = data_all[["status"]].as_matrix()
y = y.ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

# 随机森林
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=2018)
rf.fit(x_train, y_train)
rf_acc = rf.score(x_test, y_test)
print("RandomForestClassifier Acc: ", rf_acc)

# GBDT
gb = GradientBoostingClassifier(random_state=2018)
gb.fit(x_train, y_train)
gb_acc = gb.score(x_test, y_test)
print("GradientBoostingClassifier Acc: ", gb_acc)

# XGBoost
xgb = XGBClassifier(random_state=2018)
xgb.fit(x_train, y_train)
xgb_acc = xgb.score(x_test, y_test)
print("XGBClassifier Acc: ", xgb_acc)

# LightGBM
lg = LGBMClassifier(random_state=2018)
lg.fit(x_train, y_train)
lg_acc = lg.score(x_test, y_test)
print("LGBMClassifier Acc: ", lg_acc)
