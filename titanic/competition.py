import pandas as pd
import sklearn.preprocessing as pp
import sklearn.ensemble as ensmbl
from sklearn.svm import SVC

import helpers

data_train = pd.read_csv("titanic\\train.csv", header=0)
data_test = pd.read_csv("titanic\\test.csv", header=0)

y_train = data_train["Survived"]

encoder = pp.OneHotEncoder(sparse=False)
categorical_train = encoder.fit_transform(data_train.loc[:,["Sex"]])
categorical_train = pd.DataFrame(data=categorical_train[:,0], columns=["Female"])
categorical_test = encoder.fit_transform(data_test.loc[:, ["Sex"]])
categorical_test = pd.DataFrame(data=categorical_test[:,0], columns=["Female"])

categorical_train["Pclass"] = data_train["Pclass"]
categorical_test["Pclass"] = data_test["Pclass"]

categorical_train["Cabin"] = (data_train["Cabin"].isnull() == False).astype(int)
categorical_test["Cabin"] = (data_test["Cabin"].isnull() == False).astype(int)

mean_age = pd.concat([data_train["Age"], data_test["Age"]]).mean()
data_train["Age"].fillna(mean_age, inplace=True)
data_test["Age"].fillna(mean_age, inplace=True)

mean_fare = pd.concat([data_train["Fare"], data_test["Fare"]]).mean()
data_train["Fare"].fillna(mean_fare, inplace=True)
data_test["Fare"].fillna(mean_fare, inplace=True)

continous_train = data_train.loc[:,["Age", "Fare"]]
continous_test = data_test.loc[:,["Age", "Fare"]]

continous_train["Relatives"] = data_train["SibSp"] + data_train["Parch"]
continous_test["Relatives"] = data_test["SibSp"] + data_test["Parch"]

continous_all = pd.concat([continous_train, continous_test])

scaler = pp.StandardScaler()
scaler.fit(continous_all)

X_standard = scaler.transform(continous_train)
continous_train = pd.DataFrame(data=X_standard, columns=["Age", "Fare", "Relatives"])

X_standard = scaler.transform(continous_test)
continous_test = pd.DataFrame(data=X_standard, columns=["Age", "Fare", "Relatives"])

X_train = pd.concat([categorical_train, continous_train], axis=1)
X_test = pd.concat([categorical_test, continous_test], axis=1)

missing = X_test[X_test.isnull().any(axis=1)]

#estimator = ensmbl.GradientBoostingClassifier(learning_rate=0.05, n_estimators=700, max_depth=3)
estimator = SVC(C=10, kernel='rbf')
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)

print("Classifier: SVC")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')

y_test_pred = estimator.predict(X_test)

results = pd.DataFrame(data=data_test["PassengerId"], columns=["PassengerId"])
results["Survived"] = y_test_pred
results.set_index('PassengerId', inplace=True)

results.to_csv("titanic\\results.csv")

x = 1