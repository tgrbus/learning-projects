import sklearn.tree as tree

import helpers

X_train, X_test, y_train, y_test = helpers.getDatasets()

for d in range(5, 15, 1):
    reg = tree.DecisionTreeClassifier(max_depth=d)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    print(f"Depth: {d}")
    print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
    print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
    print("-"*30)

# depth 11