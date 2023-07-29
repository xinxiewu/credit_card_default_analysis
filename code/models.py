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