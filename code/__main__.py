"""
__maim__.py contains the workflow to run all sub-programs
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
from models import *
from deep_learning import *
from imbalanced_strategy import *

def main(fielurl=None, output=r'../public/output', drop_col=None, target=None, target_rename=None, feat_disc_threshold=None, cont_num_cat=None, gpu_yn=False):
    """
    Step 1: Data Preparation & EDA
    """
    # 1.1 Download dataset from Github & read as DataFrame
    df = download_file(fielurl)
    if drop_col is not None:
        df.drop(columns=drop_col, inplace=True)
    if target_rename is not None:
        df.rename(columns={target:target_rename},inplace=True)
        target=target_rename
        
    # 1.2 EDA
    # (a) Basic Statistic, Missing & Outlier/Extreme Values
    # df.describe().to_csv(os.path.join(output, 'basic_statistic.csv'))
    ## Distribution of Features (by threshold) & Targets
    target_prob(df=df, target=target, output=output)    # 78% vs 22%
    ## Features by discrete/continuous
    feat_discrete = [col for col in df.columns if getattr(df, col).nunique() <= feat_disc_threshold and col != target]
    feat_continous = [col for col in df.columns if col not in feat_discrete and col != target]
    ## Features of demographic 
    # feature_discrete(df = df, cols = feat_discrete, target = target, key = 'AGE', output = output,
    #                  col_excl = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
    #                  labels = {'SEX': {1:'Male', 2:'Female'},
    #                            'EDUCATION': {0:'Unk1', 1:'Graduate', 2:'University', 3:'High Sch', 
    #                                          4:'Others', 5:'Unk2', 6:'Unk3'},
    #                            'MARRIAGE': {0:'Unknown', 1:'Married', 2:'Single', 3:'Others'}}
    #                 )
    ## Features of payments & status
    # feature_continuous  (df = df, target = target, fname='bill_amount.png', output = output,
    #                     cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],
    #                     )
    # feature_continuous  (df = df, target = target, fname='pay_amount.png', output = output,
    #                     cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'],
    #                     )
    # feature_continuous  (df = df, target = target, fname='pay_status.png', output = output,
    #                     cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
    #                     )
    # feature_continuous  (df = df, target = target, fname='limit_age.png', subplts= [1,2], figuresize=(5,5), output = output,
    #                     cols = ['LIMIT_BAL', 'AGE'],
    #                     )

    # (b) Data Normalization & Discretization
    ## Normalization
    df_std = data_standardize(df, target=target)
    ## Discretization
    df_disct = data_discretize(df, feat_continous, target, cont_num_cat)

    # (c) Correlation Matrix
    # plt.subplots(figsize=(30,30))
    # sns.heatmap(df_std.corr(), annot=True)
    # plt.savefig(os.path.join(output, 'corr_heat_map.png'))

    # (d) Feature Engineering/Selection - PCA + K-means

    # (e) Data Split
    x_train_std, x_test_std, y_train_std, y_test_std = data_split(df=df_std, label=target)
    x_train_disct, x_test_disct, y_train_disct, y_test_disct = data_split(df=df_disct, label=target)

    """
    Step 2: Base Models - Confusion Matrix & AUC-ROC
    """
    print(f"BaseLine Models Start")
    # 2.1 Logistic Regression
    print(f"Logistic Regression - Continuous")
    lr_cont, lr_cont_proba = BaseLine(model='lr', data='continuous', 
                                      x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    # 2.2 Support Vector Machine
    print(f"SVM - Continuous - rbf")
    svm_cont_rbf, svm_cont_rbf_proba = BaseLine(model='svm', data='continuous', svm_kernel= 'rbf',
                                        x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    print(f"SVM - Continuous - sigmoid")
    svm_cont_sig, svm_cont_sig_proba = BaseLine(model='svm', data='continuous', svm_kernel= 'sigmoid',
                                        x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    print(f"SVM - Continuous - poly")
    svm_cont_pol, svm_cont_pol_proba = BaseLine(model='svm', data='continuous', svm_kernel= 'poly',
                                        x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    print(f"SVM - Discrete - rbf")
    svm_disct_rbf, svm_disct_rbf_proba = BaseLine(model='svm', data='discrete', svm_kernel= 'rbf',
                                          x_train=x_train_disct, y_train=y_train_disct, x_test=x_test_disct, y_test=y_test_disct)
    print(f"SVM - Discrete - sigmoid")
    svm_disct_sig, svm_disct_sig_proba = BaseLine(model='svm', data='discrete', svm_kernel= 'sigmoid',
                                          x_train=x_train_disct, y_train=y_train_disct, x_test=x_test_disct, y_test=y_test_disct)
    print(f"SVM - Discrete - poly")
    svm_disct_pol, svm_disct_pol_proba = BaseLine(model='svm', data='discrete', svm_kernel= 'poly',
                                          x_train=x_train_disct, y_train=y_train_disct, x_test=x_test_disct, y_test=y_test_disct)
    # 2.3 Naive Bayes
    print(f"Naive Bayes - Discrete - [0.5,0.5]")
    nb_disct_1, nb_disct_1_proba = BaseLine(model='nb', data='discrete', class_prior=[0.5,0.5],
                                        x_train=x_train_disct, y_train=y_train_disct, x_test=x_test_disct, y_test=y_test_disct)
    print(f"Naive Bayes - Discrete - [0.78,0.22]")
    nb_disct_2, nb_disct_2_proba = BaseLine(model='nb', data='discrete', class_prior=[0.78,0.22],
                                        x_train=x_train_disct, y_train=y_train_disct, x_test=x_test_disct, y_test=y_test_disct)
    # 2.4 Decision Treee
    dt_cont, dt_cont_proba = BaseLine(model='dt', data='continuous',
                                      x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    dt_disct, dt_disct_proba = BaseLine(model='dt', data='discrete',
                                        x_train=x_train_disct, y_train=y_train_disct, x_test=x_test_disct, y_test=y_test_disct)
    # 2.5 Gaussian Discriminant Analysis
    print(f"GDA - Continuous - No Prior")
    gda_cont_0, gda_cont_0_proba = BaseLine(model='gda', data='continuous', class_prior=None,
                                        x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    print(f"GDA - Continuous - [0.5,0.5]")
    gda_cont_1, gda_cont_1_proba = BaseLine(model='gda', data='continuous', class_prior=[0.5,0.5],
                                        x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    print(f"GDA - Continuous - [0.78,0.22]")
    gda_cont_2, gda_cont_2_proba = BaseLine(model='gda', data='continuous', class_prior=[0.78,0.22],
                                        x_train=x_train_std, y_train=y_train_std, x_test=x_test_std, y_test=y_test_std)
    print(f"Results")
    res_df = pd.concat([lr_cont, 
                        svm_cont_rbf, svm_cont_sig, svm_cont_pol, svm_disct_rbf, svm_disct_sig, svm_disct_pol,
                        nb_disct_1, nb_disct_2, 
                        gda_cont_0, gda_cont_1, gda_cont_2, 
                        dt_cont, dt_disct])
    res_df.to_csv(os.path.join(output, 'model_results.csv'), index=False)
    roc_curves(y_true=y_test_std, 
               y_score=[lr_cont_proba,  
                        svm_cont_rbf_proba, svm_cont_sig_proba, svm_cont_pol_proba, svm_disct_rbf_proba, svm_disct_sig_proba, svm_disct_pol_proba,
                        nb_disct_1_proba, nb_disct_2_proba, 
                        gda_cont_0_proba, gda_cont_1_proba, gda_cont_2_proba, 
                        dt_cont_proba, dt_disct_proba], 
               models=['lr-cont', 
                       'svm-cont-rbf', 'svm-cont-sig', 'svm-cont-poly', 'svm-disct-rbf', 'svm-disct-sig', 'svm-disct_poly', 
                       'nb-[0.5,0.5]', 'nb-[0.78,0.22]', 
                       'gda', 'gda-[0.5,0.5]', 'gda-[0.78,0.22]', 
                       'dt-cont', 'dt-disct'], 
               output=output, fname='base_rocs.png')

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
    main(fielurl = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/credit_card_default/default_of_credit_card_clients.csv',
         drop_col= ['ID'],
         target  = 'default_payment_next_month',
         target_rename = 'default',
         feat_disc_threshold = 11,
         cont_num_cat = 9,
         gpu_yn = True)