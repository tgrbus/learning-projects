from sklearn import metrics as sm
from sklearn import linear_model as lm
from sklearn import pipeline
from sklearn import preprocessing
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble

from feature_analysis import exportDatasets

def print_performace(y_train, y_train_pred, y_test, y_test_pred):
    print("Train mean squared error =", round(sm.mean_squared_error(y_train, y_train_pred), 2))
    print("Test mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

X_train, X_test, y_train, y_test = exportDatasets()

reg = lm.LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

print("\nLinear regression")
print("-"*30)
print_performace(y_train, y_train_pred, y_test, y_test_pred)

degree = 3
reg = pipeline.make_pipeline(preprocessing.PolynomialFeatures(degree), lm.LinearRegression())
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

print(f"\nPolinomial regression with degree = {degree}")
print("-"*30)
print_performace(y_train, y_train_pred, y_test, y_test_pred)

reg = svm.SVR(C=0.5, epsilon=0.2)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

print("\nSVR")
print("-"*30)
print_performace(y_train, y_train_pred, y_test, y_test_pred)

max_depth = 4
reg = tree.DecisionTreeRegressor(max_depth=max_depth)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

print("\nDecision Tree")
print("-"*30)
print_performace(y_train, y_train_pred, y_test, y_test_pred)

reg = ensemble.GradientBoostingRegressor(n_estimators=600, max_depth=5, learning_rate=0.01)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

print("\nGradient Boosting")
print("-"*30)
print_performace(y_train, y_train_pred, y_test, y_test_pred)