# Predicting Postgraduate Income
### Northwestern University EECS 349 
##### By: Murphy Angelo, Jared Fernandez, Isaac Lee 
##### Contact: jared.fern@u.northwestern.edu,  isaaclee2019@u.northwestern.edu, mca@u.northwestern.edu

### ABSTRACT
We used machine learning to explore data on American colleges to predict post-graduate earnings. There are hundreds of different features that make each college unique, including: location, admission rates, tuition, demographics, average student loans, and average standardized test scores. We want to discover which of these features are most significant in predicting the median ten year post-graduate earnings for a given college.

We used the Scikit-learn Python package to test different learners. Additionally, we tested multiple methods of ensembling these learners together. For the classification task, we tested the accuracy of the following base classifiers: Decision Tree, 1/3/5-Nearest Neighbor, Gaussian Naive Bayes, Adaboost, Random Forest, Recurrent Neural Net, Gradient Boosting, and Logistic Regression. Additionally, we tested an ensemble method, combining all the previous classifiers and assigning class using majority vote.  For the regression task, the same base algorithms were used. However, the Naive Bayes classifier was substituted with a linear regression model. Our accuracies can be found in Table 1 below.

![Table 1](https://raw.githubusercontent.com/isaacrlee/college-scorecard/master/table1.png)

From our decision tree, it can be determined that the most important features are: percentage of students that took out loans, amount of university spending per student, and average student family income.

![Tree](https://raw.githubusercontent.com/isaacrlee/college-scorecard/master/tree.png)
_A decision tree that shows the most important features when predicting Md10yr._
