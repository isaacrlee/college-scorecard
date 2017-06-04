import numpy as np
import pydotplus
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, \
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn import linear_model
from data import samples, target, income

imp = Imputer(missing_values=-1, strategy='mean', axis=0)
imp.fit(samples)
samples = imp.transform(samples)


def calc_zeror():
    d = {}
    for i in target:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return float(max(i for i in d.values())) / len(target)


def classifiers():
    clfs = [
        tree.DecisionTreeClassifier().fit(samples, target),
        KNeighborsClassifier(n_neighbors=1).fit(samples, target),
        KNeighborsClassifier(n_neighbors=3).fit(samples, target),
        KNeighborsClassifier(n_neighbors=5).fit(samples, target),
        GaussianNB().fit(samples, target),
        AdaBoostClassifier().fit(samples, target),
        MLPClassifier(alpha=1, max_iter=100).fit(samples, target),
        RandomForestClassifier().fit(samples, target),
        GradientBoostingClassifier().fit(samples, target),
        linear_model.LogisticRegressionCV().fit(samples, target)
    ]

    averages = []
    for d in range(0, len(clfs)):
        averages.append(cross_val_score(clfs[d], samples, target, cv=10))

    print("Classification Accuracy: ")
    print("ZeroR Accuracy: %0.2f" % (calc_zeror()))
    print("Tree Accuracy: %0.2f (+/- %0.2f)" %
          (averages[0].mean(), averages[0].std() * 2))
    print("1-NN Accuracy: %0.2f (+/- %0.2f)" %
          (averages[1].mean(), averages[1].std() * 2))
    print("3-NN Accuracy: %0.2f (+/- %0.2f)" %
          (averages[2].mean(), averages[2].std() * 2))
    print("5-NN Accuracy: %0.2f (+/- %0.2f)" %
          (averages[3].mean(), averages[3].std() * 2))
    print("Naive Bayes Accuracy: %0.2f (+/- %0.2f)" %
          (averages[4].mean(), averages[4].std() * 2))
    print("Adaboost Accuracy: %0.2f (+/- %0.2f)" %
          (averages[5].mean(), averages[5].std() * 2))
    print("Recurrent NN Accuracy:  %0.2f (+/- %0.2f)" %
          (averages[6].mean(), averages[6].std() * 2))
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" %
          (averages[7].mean(), averages[7].std() * 2))
    print("GradientBoost Accuracy:  %0.2f (+/- %0.2f)" %
          (averages[8].mean(), averages[8].std() * 2))
    print("Logistic Regression Accuracy:  %0.2f (+/- %0.2f)\n" %
          (averages[9].mean(), averages[9].std() * 2))


def regressors():
    regs = [
        tree.DecisionTreeRegressor().fit(samples, income),
        KNeighborsRegressor(n_neighbors=1).fit(samples, income),
        KNeighborsRegressor(n_neighbors=3).fit(samples, income),
        KNeighborsRegressor(n_neighbors=5).fit(samples, income),
        AdaBoostRegressor().fit(samples, income),
        RandomForestRegressor().fit(samples, income),
        MLPRegressor().fit(samples, income),
        GradientBoostingRegressor().fit(samples, income),
        linear_model.LinearRegression().fit(samples, income),
    ]

    averages, diff = [], []
    for d in range(0, len(regs)):
        averages.append(cross_val_score(
            regs[d], samples, income, scoring="neg_mean_squared_error", cv=10))

    income_avg = np.average(income)
    for d in range(0, len(income)):
        diff.append((income[d] - income_avg)**2)

    print("Estimation Errors: ")
    print("ZeroR: %0.2f" % (np.sqrt(np.sum(diff) / len(income))))
    print("DTree: %0.2f" % (np.sqrt(np.absolute(averages[0].mean()))))
    print("1-NN: %0.2f" % (np.sqrt(np.absolute(averages[1].mean()))))
    print("3-NN: %0.2f" % (np.sqrt(np.absolute(averages[2].mean()))))
    print("5-NN: %0.2f" % (np.sqrt(np.absolute(averages[3].mean()))))
    print("AdaBoost: %0.2f" % (np.sqrt(np.absolute(averages[4].mean()))))
    print("Random Forest: %0.2f" % (np.sqrt(np.absolute(averages[5].mean()))))
    print("Recurrent NN: %0.2f" % (np.sqrt(np.absolute(averages[6].mean()))))
    print("Gradient Boost: %0.2f" % (np.sqrt(np.absolute(averages[7].mean()))))
    print("Linear Regression: %0.2f" %
          (np.sqrt(np.absolute(averages[8].mean()))))


def print_tree():
    feature_names = ["LATITUDE", "LONGITUDE", "ADM_RATE", "ADM_RATE_ALL", "SAT_AVG", "UGDS", "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN", "UGDS_AIAN", "UGDS_NHPI", "UGDS_2MOR", "UGDS_NRA", "UGDS_UNKN", "UGDS_WHITENH", "UGDS_BLACKNH", "UGDS_API", "UGDS_AIANOLD", "UGDS_HISPOLD", "UG_NRA", "UG_UNKN", "UG_WHITENH", "UG_BLACKNH", "UG_API", "UG_AIANOLD", "UG_HISPOLD", "TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITFTE", "INEXPFTE", "AVGFACSAL", "PFTFAC", "PCTPELL",
                     "PCTFLOAN", "COUNT_ED", "LOAN_EVER", "PELL_EVER", "AGE_ENTRY", "AGEGE24", "FEMALE", "MARRIED", "DEPENDENT", "VETERAN", "FIRST_GEN", "FAMINC", "MD_FAMINC", "FAMINC_IND", "PCT_WHITE", "PCT_BLACK", "PCT_ASIAN", "PCT_HISPANIC", "PCT_BA", "PCT_GRAD_PROF", "PCT_BORN_US", "MEDIAN_HH_INC", "POVERTY_RATE", "UNEMP_RATE", "LN_MEDIAN_HH_INC", "DEBT_MDN_SUPP", "GRAD_DEBT_MDN_SUPP", "GRAD_DEBT_MDN10YR_SUPP", "UGDS_MEN", "UGDS_WOMEN", "MEDIAN_HH_INC2", "10 yr median"]
    # target_names = ["<=23800", "<=30600", "<=38000", ">38000"]
    clf = tree.DecisionTreeRegressor(max_depth=3).fit(samples, income)
    dot_data = tree.export_graphviz(
        clf, out_file=None, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("tree.pdf")

if __name__ == "__main__":
    classifiers()
    # regressors()
