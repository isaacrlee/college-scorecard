from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from data import samples as samples
from data import target as target

imp = Imputer(missing_values = -1, strategy='mean', axis=0)
imp.fit(samples)
samples = imp.transform(samples)

# Calculates ZeroR accuracy for Y
def calc_zeror():
    d = {}
    for i in target:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return float(max(i for i in d.values()))/len(target)

if __name__ == "__main__":
    dtree_clf = tree.DecisionTreeClassifier()
    dtree_clf = dtree_clf.fit(samples, target)
    dtree_scores = cross_val_score(dtree_clf, samples, target, cv=10)
    dtree_scores = cross_val_score(dtree_clf, samples, target, cv=10)
    
    knn_1 = KNeighborsClassifier(n_neighbors = 1)
    knn_3 = KNeighborsClassifier(n_neighbors = 3)
    knn_5 = KNeighborsClassifier(n_neighbors = 5)

    knn_1 = knn_1.fit(samples, target)
    knn_3 = knn_3.fit(samples, target)
    knn_5 = knn_5.fit(samples, target)

    knn1_scores = cross_val_score(knn_1, samples, target, cv = 10)
    knn3_scores = cross_val_score(knn_3, samples, target, cv = 10)
    knn5_scores = cross_val_score(knn_5, samples, target, cv = 10)

    gnb   = GaussianNB().fit(samples,target)
    gnb_scores  = cross_val_score(gnb, samples, target, cv = 10)

    adaboost = AdaBoostClassifier().fit(samples,target)
    adaboost_scores = cross_val_score(adaboost, samples,target, cv = 10)

    rnn   = MLPClassifier(alpha = 1, max_iter=500).fit(samples, target)
    rnn_scores = cross_val_score(rnn, samples, target, cv =10)

    # svm  = SVC().fit(samples,target)
    # svm_scores = cross_val_score(svm, samples,target, cv=10)

    print ("ZeroR Accuracy: %0.2f" % (calc_zeror()))
    print("Tree Accuracy: %0.2f (+/- %0.2f)" % (dtree_scores.mean(), dtree_scores.std() * 2))
    print("1-NN Accuracy: %0.2f (+/- %0.2f)" % (knn1_scores.mean(), knn1_scores.std() * 2))
    print("3-NN Accuracy: %0.2f (+/- %0.2f)" % (knn3_scores.mean(), knn3_scores.std() * 2))
    print("5-NN Accuracy: %0.2f (+/- %0.2f)" % (knn5_scores.mean(), knn5_scores.std() * 2))
    print("Gaussian Naive Bayes Accuracy: %0.2f (+/- %0.2f)" % (gnb_scores.mean(), gnb_scores.std() * 2))
    print("Adaboost Accuracy: %0.2f (+/- %0.2f)" % (adaboost_scores.mean(), adaboost_scores.std() * 2))
    print("Multilayer Perceptron Accuracy:  %0.2f (+/- %0.2f)"%(rnn_scores.mean(), rnn_scores.std()*2))
    # print("SVM Accuracy:  %0.2f (+/- %0.2f)"%(svm_scores.mean(), svm_scores.std()*2))

