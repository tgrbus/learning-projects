import helpers
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb
from sklearn.svm import SVC
import sklearn.tree
import sklearn.neighbors as nbs
import sklearn.ensemble as ensmbl

X_train, X_test, y_train, y_test = helpers.getDatasets2()

estimator = lm.LogisticRegression()
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("-"*25)
print("Classifier: LogisticRegression, C = 1")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = lm.LogisticRegression(C=100)
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: LogisticRegression, C = 100")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = nb.GaussianNB()
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: Gaussian Naive Bayes")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = SVC(C=10, kernel='rbf')
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: Support vector classifier")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = sklearn.tree.DecisionTreeClassifier(max_depth=11)
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: Decision Tree")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = nbs.KNeighborsClassifier()
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: KNN")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = ensmbl.RandomForestClassifier(n_estimators=400, max_samples=0.4)
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: Random Forest")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)

estimator = ensmbl.GradientBoostingClassifier(learning_rate=0.05, n_estimators=700, max_depth=2)
estimator.fit(X_train, y_train)
y_train_pred = estimator.predict(X_train)
y_test_pred = estimator.predict(X_test)

print("Classifier: Gradient Boosting")
print(f'Train accuracy = {helpers.computeAccuracy(y_train, y_train_pred)}')
print(f'Test accuracy = {helpers.computeAccuracy(y_test, y_test_pred)}')
print("-"*25)