import sklearn.svm as svm
import sklearn.model_selection as ms
from feature_analysis import exportDatasets

X_train, X_test, y_train, y_test = exportDatasets()

svr = svm.SVR(kernel="rbf")
parameters = {'kernel':['poly', 'rbf', 'linear'], 'epsilon':[0.1, 0.2, 0.3], 'C':[0.1, 0.25, 0.5, 1]}

gs = ms.GridSearchCV(svr, parameters)
gs.fit(X_train, y_train)
res = gs.cv_results_
print(res)
print(gs.best_estimator_)
print(gs.best_params_)
print(gs.best_score_)
