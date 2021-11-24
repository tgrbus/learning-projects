import helpers
from sklearn.svm import SVC
import sklearn.model_selection as ms
import pprint

X_train, X_test, y_train, y_test = helpers.getDatasets2()

svc = SVC(class_weight=None)

parameters = {'kernel':['poly', 'rbf', 'sigmoid'], 'C':[1, 5, 10]}

gs = ms.GridSearchCV(svc, parameters, scoring='accuracy')
gs.fit(X_train, y_train)
res = gs.cv_results_
pp = pprint.PrettyPrinter(indent=4, width=900)
#print(res)
pp.pprint(res)
#print(gs.best_estimator_)
print(gs.best_params_)
print(gs.best_score_)

#kernel: rbf, C=10

ranks = gs.cv_results_["rank_test_score"]
combinations = gs.cv_results_["params"]
score = gs.cv_results_["mean_test_score"]

#first 3
yy = ranks[0]
xx = combinations[ranks[0] + 1]
'''
for i in range(0,3):
    print(combinations[ranks[i]+1] + "," + score[ranks[i] + 1]
'''
#pp.pprint(xx)