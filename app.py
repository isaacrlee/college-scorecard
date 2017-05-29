import numpy as np
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from data import samples, target, income

imp = Imputer(missing_values = -1, strategy='mean', axis=0)
imp.fit(samples)
samples = imp.transform(samples)

def calc_zeror():
    d = {}
    for i in target:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return float(max(i for i in d.values()))/len(target)

def classifiers():
    dtree_clf = tree.DecisionTreeClassifier().fit(samples, target)
    dtree_scores = cross_val_score(dtree_clf, samples, target, cv=10)
    
    knn_1 = KNeighborsClassifier(n_neighbors = 1).fit(samples, target)
    knn_3 = KNeighborsClassifier(n_neighbors = 3).fit(samples, target)
    knn_5 = KNeighborsClassifier(n_neighbors = 5).fit(samples, target)

    knn1_scores = cross_val_score(knn_1, samples, target, cv = 10)
    knn3_scores = cross_val_score(knn_3, samples, target, cv = 10)
    knn5_scores = cross_val_score(knn_5, samples, target, cv = 10)

    gnb   = GaussianNB().fit(samples,target)
    gnb_scores  = cross_val_score(gnb, samples, target, cv = 10)

    adaboost = AdaBoostClassifier().fit(samples,target)
    adaboost_scores = cross_val_score(adaboost, samples,target, cv = 10)

    rnn   = MLPClassifier(alpha = 1, max_iter=100).fit(samples, target)
    rnn_scores = cross_val_score(rnn, samples, target, cv =10)

    rand_forest = RandomForestClassifier().fit(samples,target)
    randforest_scores = cross_val_score(rand_forest, samples, target, cv =10)

    print("Classification Accuracy: ")
    print("ZeroR Accuracy: %0.2f" % (calc_zeror()))
    print("Tree Accuracy: %0.2f (+/- %0.2f)" % (dtree_scores.mean(), dtree_scores.std() * 2))
    print("1-NN Accuracy: %0.2f (+/- %0.2f)" % (knn1_scores.mean(), knn1_scores.std() * 2))
    print("3-NN Accuracy: %0.2f (+/- %0.2f)" % (knn3_scores.mean(), knn3_scores.std() * 2))
    print("5-NN Accuracy: %0.2f (+/- %0.2f)" % (knn5_scores.mean(), knn5_scores.std() * 2))
    print("Naive Bayes Accuracy: %0.2f (+/- %0.2f)" % (gnb_scores.mean(), gnb_scores.std() * 2))
    print("Adaboost Accuracy: %0.2f (+/- %0.2f)" % (adaboost_scores.mean(), adaboost_scores.std() * 2))
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)"% (randforest_scores.mean(), randforest_scores.std()*2))
    print("Recureent NN Accuracy:  %0.2f (+/- %0.2f)"%(rnn_scores.mean(), rnn_scores.std()*2))

def regressors():
    diff = [[],[],[],[],[],[],[],[]]
    regressors = [
        tree.DecisionTreeRegressor().fit(samples, income),
        KNeighborsRegressor(n_neighbors=1).fit(samples,income),
        KNeighborsRegressor(n_neighbors=3).fit(samples,income),
        KNeighborsRegressor(n_neighbors=5).fit(samples,income),
        AdaBoostRegressor().fit(samples,target),
        RandomForestRegressor().fit(samples,target),
        MLPRegressor().fit(samples, target)
    ]
    
    averages, diff = [], []
    for d in range(0,len(regressors)):
        averages.append(cross_val_score(regressors[d], samples, income, scoring = "neg_mean_squared_error", cv = 10))

    income_avg = np.average(income)
    for d in range(0, len(income)):
        diff.append(np.absolute(income[d]- income_avg))
    print("Estimation Errors: ")
    print("ZeroR: %0.2f" % (np.average(diff)))
    print("DTree: %0.2f" % (np.sqrt(np.absolute(averages[0].mean()))))
    print("1-NN: %0.2f" % (np.sqrt(np.absolute(averages[1].mean()))))
    print("3-NN: %0.2f" % (np.sqrt(np.absolute(averages[2].mean()))))
    print("5-NN: %0.2f" % (np.sqrt(np.absolute(averages[3].mean()))))
    print("AdaBoost: %0.2f" % (np.sqrt(np.absolute(averages[4].mean()))))
    print("Random Forest: %0.2f" % (np.sqrt(np.absolute(averages[5].mean()))))
    print("Recurrent NN: %0.2f" % (np.sqrt(np.absolute(averages[6].mean()))))

print np.average(income)   
#regressors()