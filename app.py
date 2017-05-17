import pydotplus, numpy, heapq
from sklearn, sklearn.neighbors import tree, NearestNeighbors
from sklearn.model_selection import cross_val_score
from data import samples as samples
from data import target as target

# Calculates ZeroR accuracy for Y
def calc_zeror():
    d = {}
    for i in target:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return float(masamples(i for i in d.values()))/len(target)

# Prints the Decision Tree (not that helpful)
def print_tree():
    dot_data = tree.esamplesport_graphviz(dtree_clf, out_file=None) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("iris.pdf")

# Hamming representations, Euclidean Distance, Normalized (Number of Standard Deviations/Div by Max)
for attr in range(len(samples[0])):
    attr_val = []
    for s in samples:   
        attr_val.append(s[attr])
    s_avgs.append(numpy.average(attr_val))
    s_stds.append(numpy.std(attr_val))

def hamming_kNN(example, k):
    distance = []
    for s in range(len(samples)):
        onehot_d = 0
        for attr in range(len(s)):
            if example[attr] != s[attr]:
                onehot_d += 1 
        distance.append(onehot_d)
    min_dsts = heapq.nlargest(k, enumerate(distance), key=lambda x:x[1])
    classes  = [target[i] for i in min_dsts ] 
    return max(classes, key = classes.count)

def L1_kNN(example, k):
    distance = []
    for s in range(len(samples)):
        manhattan_d = 0
        for attr in range(len(s)):
            if example[attr] != s[attr]:
                manhattan_d += np.abs(example[attr] - s[attr])
        distance.append(manhattan_d)
    min_dsts = heapq.nlargest(k, enumerate(distance), key=lambda x:x[1])
    classes  = [target[i] for i in min_dsts ]
    return max(classes, key = classes.count)

def L2_kNN(example, k):
    distance = []
    for s in range(len(samples)):
        euclid_d = 0
        for attr in range(len(s)):
            if example[attr] != s[attr]:
                euclid_d += (example[attr] - s[attr]) ** 2 
        distance.append(numpy.sqrt(euclid_d))
    min_dsts = heapq.nlargest(k, enumerate(distance), key=lambda x:x[1])
    classes  = [target[i] for i in min_dsts ]
    return max(classes, key = classes.count)

def normalized_L2(example, k):
    distance = []
    averages = []
    std_devs = []
    
    for i in range(len(samples[0])):
        for j in samples:

    for s in range(len(samples)):
        euclid_d = 0
        for attr in range(len(s)):
            if example[attr] != s[attr]:
                euclid_d += (example[attr] - s[attr]) ** 2 
        distance.append(numpy.sqrt(euclid_d))
    min_dsts = heapq.nlargest(k, enumerate(distance), key=lambda x:x[1])
    classes  = [target[i] for i in min_dsts ]
    return max(classes, key = classes.count)

def tester():


if __name__ = "__main__":
    dtree_clf = tree.DecisionTreeClassifier()
    dtree_clf = dtree_clf.fit(samples, target)
    dtree_scores = cross_val_score(dtree_clf, samples, target, cv=4)

    print ("ZeroR Accuracy: %0.2f" % (calc_zeror()))
    print("Tree Accuracy: %0.2f (+/- %0.2f)" % (dtree_scores.mean(), dtree_scores.std() * 2))
