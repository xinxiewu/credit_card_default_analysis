"""
__maim__.py contains the workflow to run all sub-programs
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader
from util import *
from models import *
from deep_learning import *
from imbalanced_strategy import *

def main(fielurl=None, output=r'../public/output', drop_col=None, target=None, target_rename=None, 
         feat_disc_threshold=None, cont_num_cat=None, km_epoch=None, 
         nn_epoch=None, nn_batch_size=None, learning_rate=0.01, gpu_yn=False):
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
    df.describe().to_csv(os.path.join(output, 'basic_statistic.csv'))
    ## Distribution of Features (by threshold) & Targets
    target_prob(df=df, target=target, output=output)    # 78% vs 22%
    ## Features by discrete/continuous
    feat_discrete = [col for col in df.columns if getattr(df, col).nunique() <= feat_disc_threshold and col != target]
    feat_continous = [col for col in df.columns if col not in feat_discrete and col != target]
    ## Features of demographic 
    feature_discrete(df = df, cols = feat_discrete, target = target, key = 'AGE', output = output,
                     col_excl = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
                     labels = {'SEX': {1:'Male', 2:'Female'},
                               'EDUCATION': {0:'Unk1', 1:'Graduate', 2:'University', 3:'High Sch', 
                                             4:'Others', 5:'Unk2', 6:'Unk3'},
                               'MARRIAGE': {0:'Unknown', 1:'Married', 2:'Single', 3:'Others'}}
                    )
    ## Features of payments & status
    feature_continuous  (df = df, target = target, fname='bill_amount.png', output = output,
                        cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],
                        )
    feature_continuous  (df = df, target = target, fname='pay_amount.png', output = output,
                        cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'],
                        )
    feature_continuous  (df = df, target = target, fname='pay_status.png', output = output,
                        cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
                        )
    feature_continuous  (df = df, target = target, fname='limit_age.png', subplts= [1,2], figuresize=(5,5), output = output,
                        cols = ['LIMIT_BAL', 'AGE'],
                        )

    # (b) Data Normalization & Discretization
    ## Normalization
    df_std = data_standardize(df, target=target)
    ## Discretization
    df_disct = data_discretize(df, feat_continous, target, cont_num_cat)

    # (c) Correlation Matrix
    plt.subplots(figsize=(30,30))
    sns.heatmap(df_std.corr(), annot=True)
    plt.savefig(os.path.join(output, 'corr_heat_map.png'))

    # (d) Data Split
    x_train_std, x_test_std, y_train_std, y_test_std = data_split(df=df_std, label=target)
    x_train_disct, x_test_disct, y_train_disct, y_test_disct = data_split(df=df_disct, label=target)
    x_train_nn, x_val_nn, x_test_nn, y_train_nn, y_val_nn, y_test_nn = data_split(df=df_std, label=target
                                                                                  ,validation=True, train_size=0.8
                                                                                  ,tensor=True)

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
    print(f"BaseLine Results")
    res_df = pd.concat([lr_cont, 
                        svm_cont_rbf, svm_cont_sig, svm_cont_pol, svm_disct_rbf, svm_disct_sig, svm_disct_pol,
                        nb_disct_1, nb_disct_2, 
                        gda_cont_0, gda_cont_1, gda_cont_2, 
                        dt_cont, dt_disct])
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
    Step 3: Improvement by Feature Engineering - PCA + K-means
    """
    print(f"PCA & K-means Starts")
    x_std, y_std, x_disct, y_disct = pd.concat([x_train_std, x_test_std]), pd.concat([y_train_std, y_test_std]), pd.concat([x_train_disct, x_test_disct]), pd.concat([y_train_disct, y_test_disct])
    # 3.1 PCA
    ## Continuous
    print(f"PCA - Continuous")
    pca_n_cont = pca_choice(data='continuous', model='svm-rbf', x=x_std, x_train=x_train_std, x_test=x_test_std, y_train=y_train_std, y_test=y_test_std)
    # pca_n_cont = 23
    pca_cont = PCA(n_components=pca_n_cont).fit(x_std)
    x_pca_std, x_train_pca_std, x_test_pca_std = pca_cont.transform(x_std), pca_cont.transform(x_train_std), pca_cont.transform(x_test_std)
    print(f"pca_n_cont: {pca_n_cont}") #23
    # 3.2 K-means
    print(f"K-means - Continuous")
    acc_std, x_km_std, y_km_std = kmeans(n_cluster=2, x=x_pca_std, y=y_std, epoch=km_epoch)
    print(f"y_km_std: {len(y_km_std)}") # 20729
    # 3.3 Improved Results
    print(f"PCA & K-means")
    x_train_k, x_test_k, y_train_k, y_test_k = train_test_split(x_km_std, y_km_std, test_size=0.2, random_state=42)
    lr_pca_km, lr_pca_km_proba = BaseLine(model='svm', data='continuous', svm_kernel= 'rbf', x_train=x_train_k, y_train=y_train_k, x_test=x_test_k, y_test=y_test_k)
    lr_pca_km['Model'] = f"PCA({pca_n_cont})+KM({len(y_km_std)}, {format(acc_std, '.2%')}) SVM-rbf"
    print(f"PCA & K-means Results")
    res_df = pd.concat([res_df, lr_pca_km])

    """
    Step 4: Imbalance Discussion - SMOTE
    """
    print(f"SMOTE Start")
    pos_arr = (x_train_std[y_train_std['default']==1]).values
    new_pos_arr = SMOTE(pos_arr, 3, 5).oversampling()
    new_pos_df = pd.DataFrame(new_pos_arr)
    delete = new_pos_df.shape[0]+(x_train_std[y_train_std['default']==1]).shape[0] - (x_train_std[y_train_std['default']==0]).shape[0]
    new_pos_df_final = new_pos_df.iloc[:len(new_pos_df)-delete,:]
    new_pos_df_final.columns = x_train_std.columns
    new_pos_df_final.insert(new_pos_df_final.shape[1], 'default', 1)
    x_train_std_new = pd.concat([x_train_std, new_pos_df_final.iloc[:,:-1]])
    y_train_std_new = pd.concat([y_train_std, pd.DataFrame(new_pos_df_final.iloc[:,-1])])
    print(f"Logistic Regression - Continuous")
    lr_cont, lr_cont_proba = BaseLine(model='lr', data='continuous', 
                                      x_train=x_train_std_new, y_train=y_train_std_new, x_test=x_test_std, y_test=y_test_std)
    res_df = pd.concat([res_df, lr_cont])
    res_df.to_csv(os.path.join(output, 'model_results.csv'), index=False)

    """
    Step 5: Deep Learning
    """
    print(f"Deep Learning Start")
    if gpu_yn and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_loader = DataLoader(TensorDataset(x_train_nn, y_train_nn), batch_size=nn_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_nn, y_val_nn), batch_size=nn_batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_nn, y_test_nn), batch_size=nn_batch_size, shuffle=True)
    training_nn(epochs=nn_epoch, learning_rate=learning_rate, loss_func='CE', optimizer_para='SGD',
                input_feature=23, output_feature=2, dim1=36, dim2=108, dropout=0.8, val_size = 3000,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, output=output)

    print(f"Project Ends")
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
         km_epoch = 100,
         nn_epoch = 1000,
         nn_batch_size = 64,
         learning_rate = 0.01,
         gpu_yn = True)