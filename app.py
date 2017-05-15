import pydotplus
from sklearn import tree
from sklearn.model_selection import cross_val_score
from data import samples as X
from data import target as Y

def calc_zeror():
    """Calculates ZeroR accuracy for Y"""
    d = {}
    for i in Y:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return float(max(i for i in d.values()))/len(Y)

def print_tree():
    """Prints the Decision Tree (not that helpful)"""
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("iris.pdf")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

scores = cross_val_score(clf, X, Y, cv=4)

print ("ZeroR Accuracy: %0.2f" % (calc_zeror()))
print("Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
