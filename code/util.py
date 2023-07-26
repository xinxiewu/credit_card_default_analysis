"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. delete_files: Delete files in the given folder, except README.md
    3. target_prob: Generate probability by category - imbalance identifier
    4. feature_discrete: Generate Violin & Pie for discrete features
    5. feature_continuous: Generate Histogram for continuous features
    6. data_standardize: Standardize X's and return DataFrame with Y
    7. data_discretize: Per rule, convert continuous variables to discrete
"""

import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

def data_discretize(df=None):
    return