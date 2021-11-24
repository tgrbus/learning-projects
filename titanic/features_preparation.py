import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.compose as cmp
import sklearn.feature_selection as fs
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as ms

#data = np.genfromtxt("titanic\\train.csv", delimiter=",", names=True)
pd_data = pd.read_csv("titanic\\train.csv", header=0)

missing_values = pd_data[pd_data.isnull().any(axis=1)]
missing_values_embarked = pd_data[pd_data.Embarked.isnull()].PassengerId.agg([len])

embarked_grouping = pd_data.groupby('Embarked').Embarked.count()

pd_data.Embarked.fillna('S', inplace=True)

missing_values_embarked = pd_data[pd_data.Age.isnull()].PassengerId.agg([len])

pd_data["Age"].fillna((pd_data["Age"].mean()), inplace=True)

ohe = pp.OneHotEncoder(sparse=False)
sex = ohe.fit_transform(pd_data.loc[:,["Sex"]])
names = ohe.get_feature_names()
embarked = ohe.fit_transform(pd_data.loc[:, ["Embarked"]])
names2 = ohe.get_feature_names()
names = np.concatenate((names, names2))

#pd_data.Sex = sex

column_transform = cmp.make_column_transformer((ohe, ["Sex", "Embarked"]), remainder='drop')

categorical = column_transform.fit_transform(pd_data)

categorical = pd.DataFrame(data=categorical, columns=names)

categorical["Pclass"] = pd_data["Pclass"]

scores = fs.chi2(sex, pd_data[["Survived"]])
scores2 = fs.chi2(embarked, pd_data[["Survived"]])

continous = pd_data.loc[:, ["Age", "SibSp", "Parch", "Fare"]]

relatives = pd_data["SibSp"] + pd_data["Parch"]
continous2 = pd_data.loc[:, ["Age", "Fare"]]
continous2["Relatives"] = relatives
missing_values = continous[continous.isnull().any(axis=1)]

scaler = pp.StandardScaler()

X_standard = scaler.fit_transform(continous)
x_mean, x_std = scaler.mean_, scaler.var_

X_standard2 = scaler.fit_transform(continous2)

continous = pd.DataFrame(data=X_standard, columns=["Age", "SibSp", "Parch", "Fare"])
continous2 = pd.DataFrame(data=X_standard2, columns=["Age", "Fare", "Relatives"])

x_mean = continous["Fare"].mean()
x_stdv = continous["Fare"].std()

X = pd.concat([categorical, continous], axis=1)
X2 = pd.concat([categorical, continous2], axis=1)
y = pd_data["Survived"]

tree = DecisionTreeClassifier()
tree.fit(X, y)
importances = tree.feature_importances_

print(X.columns)
print(importances)

'''
['x0_female', 'x0_male', 'x0_C', 'x0_Q',    'x0_S',     'Pclass', 'Age',        'SibSp', 'Parch',   'Fare']
[0.30933519 0.         0.0018837  0.0075518  0.01225673 0.10817857 0.24170013 0.05043061 0.02899193 0.23967133]
'''
tree = DecisionTreeClassifier()
tree.fit(X2, y)
importances = tree.feature_importances_

print(X2.columns)
print(importances)

'''
['x0_female', 'x0_male', 'x0_C',    'x0_Q',     'x0_S',     'Pclass',   'Age',      'Fare',     'Relatives']
[0.            0.30933519 0.00420406 0.00382267 0.0086093  0.10697478   0.23827272 0.24731184 0.08146945]
'''

'''
['x0_female', 'x0_male',    'x0_C',     'x0_Q', 'x0_S',     'Pclass', 'Age',        'Fare', 'Relatives', 'Cabin']
[0.30602149     0.         0.0051164  0.0004997  0.00655065 0.08455361 0.26422839 0.242042   0.0513947  0.03959306]
'''

X.drop(labels=['x0_male', 'x0_C', 'x0_Q', 'x0_S'], axis=1, inplace=True)
print(X.columns)

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=15)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)

errors = y_test[y_test != y_test_pred].count()

cabin = pd_data["Cabin"].isnull() == False
cabin = cabin.astype(int)

X2["Cabin"] = cabin

tree = DecisionTreeClassifier()
tree.fit(X2, y)
importances = tree.feature_importances_

print(X2.columns)
print(importances)

