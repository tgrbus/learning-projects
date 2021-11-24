import sklearn.ensemble as ensemble
import sklearn.model_selection as ms
import pprint
import helpers

X_train, X_test, y_train, y_test = helpers.getDatasets2()

gb = ensemble.GradientBoostingClassifier()

parameters = {'max_depth':[2, 3, 4], 'n_estimators':[200, 300, 400, 500, 700], 'learning_rate':[0.05, 0.06, 0.07, 0.08, 0.9, 1]}

gs = ms.GridSearchCV(gb, parameters)
gs.fit(X_train, y_train)
res = gs.cv_results_
pp = pprint.PrettyPrinter(indent=4, width=900)
#print(res)
pp.pprint(res)
print(gs.best_estimator_)
print(gs.best_params_)
print(gs.best_score_)