import sklearn.ensemble as ensemble
import sklearn.model_selection as ms
from feature_analysis import exportDatasets

X_train, X_test, y_train, y_test = exportDatasets()

gb = ensemble.GradientBoostingRegressor()

parameters = {'max_depth':[6, 7], 'n_estimators':[600, 700], 'learning_rate':[0.01, 0.1]}

gs = ms.GridSearchCV(gb, parameters)
gs.fit(X_train, y_train)
res = gs.cv_results_
print(res)
print(gs.best_estimator_)
print(gs.best_params_)
print(gs.best_score_)

print("")