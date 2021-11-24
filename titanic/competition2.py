import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.ensemble as ensmbl
from sklearn.svm import SVC

import helpers

data = pd.read_csv("titanic\\train.csv", header=0)
data_test = pd.read_csv("titanic\\test.csv", header=0)
y = data["Survived"]
encoder = pp.OneHotEncoder(sparse=False)
categorical = encoder.fit_transform(data.loc[:,["Sex"]])
categorical = pd.DataFrame(data=categorical[:, 0], columns=["Female"])
categorical["Pclass"] = data["Pclass"]
categorical["Cabin"] = (data["Cabin"].isnull() == False).astype(int)

mean_age = pd.concat([data["Age"], data_test["Age"]]).mean()
data_test["Age"].fillna(mean_age, inplace=True)
data["Age"].fillna(mean_age, inplace=True)

mean_fare = pd.concat([data["Fare"], data_test["Fare"]]).mean()
data_test["Fare"].fillna(mean_fare, inplace=True)
data["Fare"].fillna(mean_fare)

relatives = data["SibSp"] + data["Parch"]
continous = data.loc[:, ["Age", "Fare"]]
continous["Relatives"] = relatives
scaler = pp.StandardScaler()
X_standard = scaler.fit_transform(continous)
continous = pd.DataFrame(data=X_standard, columns=["Age", "Fare", 'Relatives'])
X = pd.concat([categorical, continous], axis=1)
#
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=15)
#
#X_train, y_train = X,y

#estimator = ensmbl.GradientBoostingClassifier(learning_rate=0.08, n_estimators=400, max_depth=2)
#estimator = SVC(C=10, kernel='rbf')
#estimator = SVC(C=5, kernel='rbf', probability=True, gamma='auto')
estimator = ensmbl.RandomForestClassifier(n_estimators=500, max_samples=0.4, max_depth=100)
estimator.fit(X_train, y_train)

y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: Gradient Boosting")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)


categorical_test = encoder.fit_transform(data_test.loc[:,["Sex"]])
categorical_test = pd.DataFrame(data=categorical_test[:, 0], columns=["Female"])
categorical_test["Pclass"] = data_test["Pclass"]
categorical_test["Cabin"] = (data_test["Cabin"].isnull() == False).astype(int)

relatives_test = data_test["SibSp"] + data_test["Parch"]
continous_test = data_test.loc[:, ["Age", "Fare"]]
continous_test["Relatives"] = relatives_test

continous_test = scaler.transform(continous_test)
continous_test = pd.DataFrame(data=continous_test, columns=["Age", "Fare", 'Relatives'])
X_test = pd.concat([categorical_test, continous_test], axis=1)

y_test_pred = estimator.predict(X_test)

results = pd.DataFrame(data=data_test["PassengerId"], columns=["PassengerId"])
results["Survived"] = y_test_pred
results.set_index('PassengerId', inplace=True)

results.to_csv("titanic\\results.csv")

x = 1