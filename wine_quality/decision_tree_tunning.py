import sklearn.tree as tree

from feature_analysis import exportDatasets
from models import print_performace

X_train, X_test, y_train, y_test = exportDatasets()

for d in range(2, 7, 1):
    reg = tree.DecisionTreeRegressor(max_depth=d)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    print(f"Depth: {d}")
    print_performace(y_train, y_train_pred, y_test, y_test_pred)
    print("-"*30)