from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms
import pprint

import helpers


X_train, X_test, y_train, y_test = helpers.getDatasets2()

svc = RandomForestClassifier()

parameters = {'n_estimators': [200, 300, 400, 500], 'max_samples': [0.3, 0.4, 0.6, 0.8], 'max_depth': [10, 50, 100, None], 'max_features':[0.3, 0.4, None]}

gs = ms.GridSearchCV(svc, parameters)
gs.fit(X_train, y_train)
res = gs.cv_results_
pp = pprint.PrettyPrinter(indent=4, width=900)
#print(res)
pp.pprint(res)
#print(gs.best_estimator_)
print(gs.best_params_)
print(gs.best_score_)