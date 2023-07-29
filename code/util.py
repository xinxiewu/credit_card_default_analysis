"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. delete_files: Delete files in the given folder, except README.md
    3. target_prob: Generate probability by category - imbalance identifier
    4. feature_discrete: Generate Violin & Pie for discrete features
    5. feature_continuous: Generate Histogram for continuous features
    6. data_standardize: Standardize X's and return DataFrame with Y
    7. data_discretize: Per rule, convert continuous variables to discrete
    8. data_split: Split dataset into training, validation and testing
    9. point_eval_metric: Given confusion matrix, generate point evaluation metrics
    10. roc_curves: Given list of models, return AUC-ROC Curves
"""

import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_curve, roc_auc_score

# download_file(url)
def download_file(url=None, output=r'../public/output'):
    """ Download the .csv file from the given link and read as dataframe

    Args: 
        url: str
        output: path to store downloaded files
    
    Returns:
        DataFrame
    """
    local_filename = os.path.join(output, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return pd.read_csv(local_filename)

# delete_files(path)
def delete_files(path=None, keep=['README.md']):
    """ Delete files in the given folder path, except README.md

    Args:
        path: path, starting with r''
        keep: files to keep, default value as README.md

    Returns:
        nothing to return
    """
    for fname in os.listdir(path):
        if fname not in (keep):
            os.remove(os.path.join(path, fname))
    return

# target_prob(output, fname, figsize, x_ft, rot, y_ft, title, title_ft, text_ft)
def target_prob(df=None, target=None, output=None, fname='target_prob.png', 
                figsize=(5,5), x_ft=10, rot=0, y_ft=10, title='Probability of Default Payment Next Month (%)', title_ft=12, text_ft=10):
    """ Generate and save as bar chart for target variable

    Args:
        df: DataFrame
        target: target variable name
        output: output path
        fname: file name ending with .pnd
        figsize: figure size, default (5,5)

    Returns:
        nothing to return
    """
    tar_prob = round(getattr(df, target).value_counts(normalize=True)*100).astype(int)
    fig = plt.figure(figsize=figsize)
    tar_prob.plot.bar()
    plt.xticks(fontsize=x_ft, rotation=rot)
    plt.yticks(fontsize=y_ft)
    plt.title(title, fontsize=title_ft)
    for x, y in zip([0, 1], tar_prob):
        plt.text(x, y, y, fontsize=text_ft)
    plt.savefig(os.path.join(output, fname))
    return

# feature_discrete(df, cols, labels, key, target, fname, output, subplts, figsize, col_excl)
def feature_discrete(df=None, cols=None, labels=None, key=None, target=None, fname='demographics.png', output=None,
                     subplts=[3,2], figsize=(20,30), col_excl=None, autopct='%0.1f%%', radius=1.5):
    """ Generate and save Violin & Pie for discrete features

    Args:
        df: DataFrame
        target: target variable name
        output: output path
        fname: file name ending with .pnd
        figsize: figure size
        cols: discrete features' name
        labels: label mapping
        key: used for Violin
        subplts: list, # of sub plots
        col_excl: columns to exclude

    Returns:
        nothing to return
    """
    cols_res = [col for col in cols if col not in col_excl]
    fig, axes = plt.subplots(subplts[0], subplts[1], figsize=figsize)
    for i, col in enumerate(cols_res):
        inter = df.groupby(col)[target].count()
        sns.violinplot(x=col, y=key, hue=target, split=True, data=df, ax=axes[i,0])
        axes[i,0].set_xticklabels(labels=labels[col].values())
        axes[i,1].pie(inter, labels=labels[col].values(), autopct=autopct, radius=radius)
    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    return

# feature_continuous(df, cols)
def feature_continuous(df=None, cols=None, target=None, fname='financial_status.png', output=None,
                       figuresize=(10,10), subplts=[3,2]):
    """ Generate and save Histogram for continuous features

    Args:
        df: DataFrame
        target: target variable name
        output: output path
        fname: file name ending with .pnd
        figsize: figure size
        cols: continuous features' name
        subplts: list, # of sub plots

    Returns:
        nothing to return
    """
    fig, axes = plt.subplots(subplts[0], subplts[1], figsize=figuresize)
    ax = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(x=col,hue=target,data=df,kde=True,ax=ax[i])
    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    return

def data_standardize(df=None,target=None):
    """ Standardize X's and return DataFrame with Y

    Args:
        df: DataFrame
        target: target variable name

    Returns:
        DataFrame
    """
    df_res = df.copy()
    for col in df_res.columns[df_res.columns != target]:
        df_res[col] = (df_res[col] - df_res[col].mean()) / df_res[col].std()
    return df_res

def data_discretize(df=None, feat_continous=None, target=None, num_cat=None):
    """ Standardize X's and return DataFrame with Y

    Args:
        df: DataFrame
        feat_continous: continuous variables
        num_cat: int

    Returns:
        DataFrame
    """
    df_disct = df.copy()
    for col in feat_continous:
        gap = (getattr(df_disct, col).max() - getattr(df_disct, col).min())/num_cat
        df_disct[col] = ((getattr(df_disct, col) - getattr(df_disct, col).min()) / gap).round(decimals=0).astype(int)
    for col in df_disct.columns:
        temp = getattr(df_disct, col).min()
        if temp < 0 and col != target:
            df_disct[col] = getattr(df_disct, col) - temp
    return df_disct

# data_split()
def data_split(df=None, label=None, validation=False, train_size=0.8, random_state=42, tensor=False):
    """ Split dataset into training, validation & 
    
    Args:
        df: DataFrame
        label: str, label column name
        validation: boolean, True if a validation set is needed, otherwise False
        train_size: float, size of training dataset, <= 1
        random_state: int, random state, default value as 42
        tensor: boolean, True if need to convert to Tensor, otherwise False

    Returns:
        DataFrames, split
    """
    if validation == False and tensor == False:
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != label], df.iloc[:,df.columns == label], 
                                                            test_size=(1-train_size), random_state=random_state)
        return x_train, x_test, y_train, y_test
    elif validation == True and tensor == True:
        x_train, x_val_te, y_train, y_val_te = train_test_split(df.iloc[:,df.columns != label], df.iloc[:,df.columns == label], 
                                                            test_size=(1-train_size), random_state=random_state)
        x_val, x_test, y_val, y_test = train_test_split(x_val_te, y_val_te, 
                                                            test_size=0.5, random_state=random_state)
        X_train = torch.Tensor(x_train.values)
        X_val = torch.Tensor(x_val.values)
        X_test = torch.Tensor(x_test.values)
        Y_train = torch.Tensor(y_train.values)
        Y_val = torch.Tensor(y_val.values)
        Y_test = torch.Tensor(y_test.values)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

# point_eval_metric()
def point_eval_metric(conf_m=None, data=None, model=None, y_true=None, y_score=None, class_prior=None, svm_kernel=None):
    """ Given confusion matrix, generate point evaluation metrics

    Args:
        conf_m: confusion matrix
        data: continous or discrete
        model: str
        y_true: true target values
        y_score: probability of predicted target
    
    Returns:
        DataFrame with info: 
            - model, test_size, prevalence, acc_tot, acc_pos, acc_neg, prec, recall, f1, auc-roc
    """
    if model.lower() == 'lr':
        model = 'LogisticReg' + '-' + data
    elif model.lower() == 'nb':
        model = 'NaiveBayes' + '-' + data + '-' + str(class_prior)
    elif model.lower() == 'svm':
        model = 'SVM' + '-' + data + '-' + svm_kernel
    elif model.lower() == 'rf':
        model = 'Random Forest'
    elif model.lower() == 'dt':
        model = 'Decision Tree' + '-' + data
    elif model.lower() == 'gda':
        model = 'Gaussian Analysis' + '-' + data + '-' + str(class_prior)
    else:
        model = model

    tn, fp, fn, tp = conf_m[0][0], conf_m[0][1], conf_m[1][0], conf_m[1][1]
    data =  {'Model': [model],
             'Test Size': [tn + fn + fp + tp],
             'Prevalence': [format((tp + fn) / (tn + fn + fp + tp), '.2%')],
             'Total Accuracy': [format((tp + tn) / (tn + fn + fp + tp), '.2%')],
             'Positive Accuracy': [format(tp / (tp + fn), '.2%')],
             'Negative Accuracy': [format(tn / (tn + fp), '.2%')],
             'Precision': [format(tp / (tp+fp), '.2%')],
             'Recall': [format(tp / (tp+fn), '.2%')],
             'F1-Score': [format(2*((tp / (tp+fp)) * (tp / (tp+fn))) / ((tp / (tp+fp)) + (tp / (tp+fn))), '.2%')],
             'AUC-ROC': [format(roc_auc_score(y_true, y_score), '.4')]
            }
    
    return pd.DataFrame.from_dict(data)

# roc_curves
def roc_curves(y_true=None, y_score=None, pos_label=1, models=None, output=None, fname=None, figuresize=(10,10)):
    """ Given confusion matrix, generate point evaluation metrics

    Args:
        models: list of str
        y_true: true target values
        y_score: probability of predicted target
        output: output path
        fname: output file name
        figuresize: size of the output graph
    
    Returns:
        Nothing to return
    """
    fig = plt.figure(figsize=figuresize)
    for i, model in enumerate(models):
        fpr, tpr, _ = roc_curve(y_true, y_score[i], pos_label=pos_label)
        plt.plot(fpr, tpr, label=model)
    plt.legend(loc='lower right')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(output, fname))

    return