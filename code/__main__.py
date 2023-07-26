"""
__maim__.py contains the workflow to run all sub-programs
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
from base_models import *
from deep_learning import *
from imbalanced_strategy import *

def main(fielurl=None, output=r'../public/output', drop_col=None, target=None, feat_disc_threshold=None, gpu_yn=False):
    """
    Step 1: Data Preparation & EDA
    """
    # 1.1 Download dataset from Github & read as DataFrame
    df = download_file(fielurl)
    if drop_col is not None:
        df.drop(columns=drop_col, inplace=True)
    # 1.2 EDA
    # (a) Basic Statistic, Missing & Outlier/Extreme Values
    df.describe().to_csv(os.path.join(output, 'basic_statistic.csv'))
    ## Distribution of Features (by threshold) & Targets
    target_prob(df=df, target=target, output=output)    # 78% vs 22%
    ## Features by discrete/continuous
    feat_discrete = [col for col in df.columns if getattr(df, col).nunique() <= feat_disc_threshold and col != target]
    feat_continous = [col for col in df.columns if col not in feat_discrete and col != target]
    ## Features of demographic 

    ## Features of payments & status


    # (b) Data Normalization

    # (c) Correlation Matrix

    # (d) Feature Engineering/Selection

    """
    Step 2: Base Models - Confusion Matrix & AUC-ROC
    """

    """
    Step 3: Improvement by Feature Engineering
    """

    """
    Step 4: Imbalance Discussion
    """

    """
    Step 5: Deep Learning
    """

    return

if __name__ == '__main__':
    """
    Step 1: Clean output folder
    """
    delete_files(path=r'../public/output')
    """
    Step 2: Call the main program
    """
    # main(fielurl = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/credit_card_default/default_of_credit_card_clients.csv',
    #      drop_col= ['ID'],
    #      target  = 'default_payment_next_month',
    #      feat_disc_threshold = 11,
    #      gpu_yn = True)