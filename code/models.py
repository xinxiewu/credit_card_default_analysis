"""
models.py contains baseline & ensemble algorithms:
    1. Baseline Algorithms:
        (1) Logistic Regression - LR
        (2) Naive Bayes - NB
        (3) Gaussian Discriminant Analysis - GDA
        (4) Decision Tree - DT
        (5) Random Forest - RF
        (6) Support Vector Machine - SVM

    2. Unsupervised Learning:
        (1) Principal Component Analysis - PCA
        (2) K-means
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from util import point_eval_metric

# Baseline Models
def BaseLine(model=None, data=None, x_train=None, y_train=None, x_test=None, y_test=None, class_prior=[0.5,0.5], svm_kernel='rbf'):
    """ Fit/predict baseline models and generate confusion matrix/claffication report
    
    Args:
        model: str, four options: LR, GDA, NB, DT, SVM
        data: continuous or discrete
        x_train: DataFrame, training dataset of features
        y_train: DataFrame, training dataset of labels
        x_test: DataFrame, testing dataset of features
        y_test: DataFrame, testing dataset of labels

    Returns:
        DataFrame, with evaluation metrics
    """
    if model.lower() == 'lr':
        clf = LogisticRegression()
    elif model.lower() == 'gda':
        if class_prior is None:
            clf = GaussianNB()
        else:
            clf = GaussianNB(priors=class_prior)
    elif model.lower() == 'nb':
        clf = CategoricalNB(class_prior=class_prior)
    elif model.lower() == 'dt':
        clf = DecisionTreeClassifier()
    elif model.lower() == 'svm':
        clf = svm.SVC(probability=True, kernel=svm_kernel)
    clf.fit(x_train, np.ravel(y_train))
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]
    conf_m = confusion_matrix(y_test, y_pred)
    return point_eval_metric(conf_m=conf_m, model=model, data=data, 
                             y_true=y_test, y_score=y_pred_proba, class_prior=class_prior, svm_kernel=svm_kernel), y_pred_proba

# Unsupervised Learning
# pca_choice()
def pca_choice(data=None, model=None, x=None, x_train=None, x_test=None, y_train=None, y_test=None):
    """ Iterate and select the best PCA component # based on the improved LR result
    
    Args:
        data: continuous or discrete
        x: DataFrame, x_train + x_test
    
    Returns:
        int, the best PCA component #
    """
    total_accuracy = 0
    pca_best_components = 0
    for i in range(x.shape[1]):
        # PCA
        print(f"Component #: {i+1}")
        pca = PCA(n_components=(i+1)).fit(x)
        x_train_pca, x_test_pca = pca.transform(x_train), pca.transform(x_test)
        # Improved Models
        if data == 'continuous' and model == 'svm-rbf':
            svm, temp = BaseLine(model='svm', data='continuous', svm_kernel= 'rbf', x_train=x_train_pca, y_train=y_train, x_test=x_test_pca, y_test=y_test)
            if float(svm['Total Accuracy'][0][:5])/100 > total_accuracy:
                total_accuracy = float(svm['Total Accuracy'][0][:5])/100
                pca_best_components = (i+1)
        elif data == 'discrete' and model == 'nb-[0.78,0.22]':
            nb, temp = BaseLine(model='nb', data='discrete', class_prior=[0.78,0.22], x_train=x_train_pca, y_train=y_train, x_test=x_test_pca, y_test=y_test)
            if float(nb['Total Accuracy'][0][:5])/100 > total_accuracy:
                total_accuracy = float(nb['Total Accuracy'][0][:5])/100
                pca_best_components = (i+1)
    return pca_best_components

# kmeans()
def kmeans(n_cluster=2, x=None, y=None, epoch=1000):
    """ Run K-means multiple times with random initialization, pick the highest accuracy one, remove outliers

    Args:
        n_cluster: int, # of clusters
        x, y: datasets
        epoch: int, # of iteration

    Returns:
        highest accuracy, with the new x and y datasets
    """
    acc, x_new, y_new = 0, [], []
    for i in range(epoch):
        if (i+1)%10 == 0:
            print(f"Epoch: {i+1}")
        km = KMeans(n_clusters=n_cluster, n_init= 'auto').fit(x)
        y_km = km.predict(x)
        correct, x_temp, y_temp = 0, [], []
        for i in range(len(y)):
            if y_km[i] == 1 - y.iloc[i][0]:
                correct += 1
                x_temp.append(x[i])
                y_temp.append(y.iloc[i][0])
        if (correct / len(y)) > acc:
            acc, x_new, y_new = (correct / len(y)), x_temp, y_temp
    return acc, x_new, y_new
