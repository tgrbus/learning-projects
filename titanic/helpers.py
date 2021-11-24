import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

def getDatasets():
    data = pd.read_csv("titanic\\train.csv", header=0)
    y = data["Survived"]
    encoder = pp.OneHotEncoder(sparse=False)
    categorical = encoder.fit_transform(data.loc[:,["Sex"]])
    categorical = pd.DataFrame(data=categorical[:, 0], columns=["Female"])
    categorical["Pclass"] = data["Pclass"]
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    continous = data.loc[:, ["Age", "Fare", 'SibSp', 'Parch']]
    scaler = pp.StandardScaler()
    X_standard = scaler.fit_transform(continous)
    continous = pd.DataFrame(data=X_standard, columns=["Age", "Fare", 'SibSp', 'Parch'])
    X = pd.concat([categorical, continous], axis=1)
    return ms.train_test_split(X, y, test_size=0.2, random_state=15)

def getDatasets2():
    data = pd.read_csv("titanic\\train.csv", header=0)
    data_test = pd.read_csv("titanic\\test.csv", header=0)
    y = data["Survived"]
    encoder = pp.OneHotEncoder(sparse=False)
    categorical = encoder.fit_transform(data.loc[:,["Sex"]])
    categorical = pd.DataFrame(data=categorical[:, 0], columns=["Female"])
    categorical["Pclass"] = data["Pclass"]
    categorical["Cabin"] = (data["Cabin"].isnull() == False).astype(int)

    #data["Age"].fillna(data["Age"].mean(), inplace=True)
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
    return ms.train_test_split(X, y, test_size=0.2, random_state=15)
    #return X, None, y, None

def getDatasets3():
    #whole test set
    data = pd.read_csv("titanic\\train.csv", header=0)
    y = data["Survived"]
    encoder = pp.OneHotEncoder(sparse=False)
    categorical = encoder.fit_transform(data.loc[:,["Sex"]])
    categorical = pd.DataFrame(data=categorical[:, 0], columns=["Female"])
    categorical["Pclass"] = data["Pclass"]
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    relatives = data["SibSp"] + data["Parch"]
    continous = data.loc[:, ["Age", "Fare"]]
    continous["Relatives"] = relatives
    scaler = pp.StandardScaler()
    X_standard = scaler.fit_transform(continous)
    continous = pd.DataFrame(data=X_standard, columns=["Age", "Fare", 'Relatives'])
    X = pd.concat([categorical, continous], axis=1)
    return X, None, y, None

def getDatasets4():
    data = pd.read_csv("titanic\\train.csv", header=0)
    y = data["Survived"]
    encoder = pp.OneHotEncoder(sparse=False)
    categorical = encoder.fit_transform(data.loc[:,["Sex"]])
    categorical = pd.DataFrame(data=categorical[:, 0], columns=["Female"])
    
    data.Embarked.fillna('S', inplace=True)
    embarked = encoder.fit_transform(data.loc[:,["Embarked"]])
    categorical["S"] = embarked[:,2]
    categorical["Q"] = embarked[:,1]
    
    categorical["Pclass"] = data["Pclass"]
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    relatives = data["SibSp"] + data["Parch"]
    continous = data.loc[:, ["Age", "Fare"]]
    continous["Relatives"] = relatives
    scaler = pp.StandardScaler()
    X_standard = scaler.fit_transform(continous)
    continous = pd.DataFrame(data=X_standard, columns=["Age", "Fare", "Relatives"])
    X = pd.concat([categorical, continous], axis=1)
    return ms.train_test_split(X, y, test_size=0.2, random_state=15)

def computeAccuracy(y, y_pred):
    correctNum = y[y == y_pred].count()
    acc = correctNum / y.shape[0]
    acc = round(acc, 5)
    return acc