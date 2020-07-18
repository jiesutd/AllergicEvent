# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-28 11:40:37
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-03 09:43:54
# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,6)
font = {'size'   : 12}

matplotlib.rc('font', **font)



def calculate_clf_p_r_f_acc_kappa(gold_label, pred_label, positive_id=1):
    ## gold_label: numpy array, binary. size: (instance_num,)
    ## pred_label: numpy array, binary, predict result. size: (instance_num,)
    assert(gold_label.ndim==1)
    if pred_label.ndim == 2:
        pred_label = np.argmax(pred_label, axis=1)
    elif pred_label.ndim > 2:
        print("PRF calculation error: dimension of pred_label should <= 2")
    gold_true = gold_label == positive_id
    pred_true = pred_label == positive_id
    gold_true_num = np.count_nonzero(gold_true)
    pred_true_num = np.count_nonzero(pred_true)
    right_pred_num = np.count_nonzero(gold_true == pred_true)
    p = (right_pred_num+0.)/pred_true_num
    r = (right_pred_num+0.)/gold_true_num
    f = 2*p*r/(p+r)
    acc = (np.sum(gold_label==pred_label)+0.)/gold_label.size
    print("Gold: %s; Pred: %s; Right: %s"%(gold_true_num, pred_true_num, right_pred_num))
    all_num = gold_label.size
    gold_false_num = all_num-gold_true_num 
    pred_false_num = all_num - pred_true_num
    pe = (gold_true_num*pred_true_num + gold_false_num*pred_false_num+0.)/(all_num*all_num)
    kappa = (acc-pe)/(1-pe)
    return p,r,f, acc, kappa



def calculate_p_r_f_acc_kappa(gold_label, pred_label):
    ## gold_label: numpy array, binary. size: (instance_numfe
    ## pred_label: numpy array, binary, predict result. size: (instance_num,)
    # print(gold_label)
    # exit(0)
    gold_true = np.sum(gold_label)
    pred_true = np.sum(pred_label)
    right_pred = np.sum(gold_label*pred_label)
    p = (right_pred+0.)/pred_true
    r = (right_pred+0.)/gold_true
    f = 2*p*r/(p+r)
    acc = (np.sum(gold_label==pred_label)+0.)/gold_label.shape[0]
    print("Gold: %s; Pred: %s; Right: %s"%(gold_true, pred_true, right_pred))
    all_num = gold_label.size
    gold_false = all_num-gold_true 
    pred_false = all_num - pred_true
    pe = (gold_true*pred_true + gold_false*pred_false+0.)/(all_num*all_num)
    kappa = (acc-pe)/(1-pe)
    return p,r,f, acc, kappa


def calculate_roc_list(gold_label, pred_label_prob, positive_id):
    ## gold_label: numpy array, binary. size: (instance_num,)
    ## pred_label: numpy array, float probability, predict result. size: (instance_num, label_num)
    
    pred = pred_label_prob[:, positive_id]
    fpr, tpr, roc_thresholds = metrics.roc_curve(gold_label, pred, pos_label=positive_id)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_thresholds, auc


def calculate_precision_recall_list(gold_label, pred_label_prob, positive_id):
    ## gold_label: numpy array, binary. size: (instance_num,)
    ## pred_label: numpy array, float probability, predict result. size: (instance_num, label_num)
    pred = pred_label_prob[:, positive_id]
    precision, recall, pr_thresholds = metrics.precision_recall_curve(gold_label, pred, pos_label=positive_id)
    return precision, recall, pr_thresholds


def plot_roc(fpr, tpr, save_dir=None):
    r''' plot roc curve for one model, based on different cut-off probabilities
        Args:
            fpr (numpy array): fpr value array
            tpr (numpy array): tpr value array
            auc (float): auc value
            model_name (string): name of the model
            save_dir (string): file directoy to be saved
    '''
    # plt.title('ROC Curve')
    plt.plot(fpr, tpr, '#1f77b4', lw=2)
    plt.plot([0, 1], [0, 1],color='gray', marker='.', linestyle='dashed',alpha=0.5)
    # plt.legend(loc='best')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    xtick = [(x+0.)/10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    # plt.grid()
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close()

def plot_multi_roc(fpr_list, tpr_list, auc_list, model_name_list, save_dir=None):
    r''' plot roc curves for multiple models, based on different cut-off probabilities
        Args:
            fpr_list (list of numpy array): fpr value array
            tpr_list (list of numpy array): tpr value array
            auc_list (list of float): auc value
            model_name_list (list of string): name of the model
            save_dir (list of string): file directoy to be saved
    '''
    plt.title('ROC Curve')
    model_num = len(fpr_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
    for idx in range(model_num):
        plt.plot(fpr_list[idx], tpr_list[idx], color_list[idx], label = '%s: AUC=%0.4f' % (model_name_list[idx], auc_list[idx]))
    plt.plot([0, 1], [0, 1],color='gray', marker='.', linestyle='dashed', alpha=0.5)
    plt.legend(loc='best')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    xtick = [(x+0.)/10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()


def plot_precision_recall(precision, recall, save_dir=None):
    r''' plot precision-recall for one model, based on different cut-off probabilities
        Args:
            precision (numpy array): precision value array
            recall (numpy array): recall value array
            model_name (string): name of the model
            save_dir (string): file directoy to be saved
    '''
    # plt.title('Precision-Recall Curve')
    plt.plot(recall, precision, 'orange', lw=2)
    plt.plot([0, 1], [1, 0], 'grey',ls='--')
    
    # f_list = [0.2, 0.4, 0.6, 0.8]
    # for f in f_list:
    #     all_rec = [f/(2-f)+idx*0.01 for idx in range(100)]
    #     rec = [a for a in all_rec if a <=1]
    #     pre_f = [f*r/(2*r-f) for r in rec]
    #     plt.plot(rec, pre_f,color='gray', linestyle='dashed')
    #     plt.text(0.9,(f/(2-f)+0.01+f/30),"F1=%s"%(f))
        

    # plt.legend(loc='best')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    xtick = [(x+0.)/10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)

    # plt.grid()
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close()


def plot_multi_precision_recall(precision_list, recall_list, model_name_list, save_dir=None):
    r''' plot precision-recall for multiple models, based on different cut-off probabilities
        Args:
            precision_list (list of numpy array): precision value array
            recall_list (list of numpy array): recall value array
            model_name_list (list of string): name of the model
            save_dir (string): file directoy to be saved
    '''
    plt.title('Precision-Recall Curve')
    model_num = len(precision_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
    for idx in range(model_num):
        plt.plot(recall_list[idx], precision_list[idx], color_list[idx], label = '%s' % (model_name_list[idx]))
    f_list = [0.2, 0.4, 0.6, 0.8]
    for f in f_list:
        all_rec = [f/(2-f)+idx*0.01 for idx in range(100)]
        rec = [a for a in all_rec if a <=1]
        pre_f = [f*r/(2*r-f) for r in rec]
        plt.plot(rec, pre_f,color='gray', linestyle='dashed')
        plt.text(0.9,(f/(2-f)+0.01+f/30),"F1=%s"%(f))
    plt.legend(loc='lower left')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    xtick = [(x+0.)/10 for x in range(0, 11)]
    ytick = xtick
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()


def plot_multi_curve(x_list, y_list, model_name_list, x_name="X_Name", y_name="Y_name", scale_one=True, log_y=False, save_dir=None):
    r''' plot precision-recall for multiple models, based on different cut-off probabilities
        Args:
            x_list (list of numpy array): list of x label array
            y_list (list of numpy array): list of y label array
            model_name_list (list of string): name of the model
            x_name (string): name of x axis
            y_name (string): name of y axis
            save_dir (string): file directoy to be saved
    '''
    plt.title('%s-%s Curve'%(y_name, x_name))
    model_num = len(x_list)
    color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
    for idx in range(model_num):
        plt.plot(x_list[idx], y_list[idx], color_list[idx], label = '%s' % (model_name_list[idx]))
    
    plt.legend(loc='best')
    if scale_one:
        plt.xlim([0, 1.001])
        plt.ylim([0, 1.001])
        plt.plot([0, 1], [1, 0],color='gray', marker='.', linestyle='dashed',alpha=0.5)
        xtick = [(x+0.)/10 for x in range(0, 11)]
        ytick = xtick
        plt.xticks(xtick)
        plt.yticks(ytick)
    if log_y:
        plt.yscale('log')
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.grid()
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

 



def read_result_file(result_file):
    flines = open(result_file,'r').readlines()
    precision_list = []
    recall_list = []
    FPR_list = []
    TPR_list = []
    for line in flines:
        if "Precision" in line:
            precision_list =[float(a) for a in line.split(' ')[1:]]
        elif "Recall" in line:
            recall_list =[float(a) for a in line.split(' ')[1:]]
        elif "FPR" in line:
            FPR_list =[float(a) for a in line.split(' ')[1:]]
        elif "TPR" in line:
            TPR_list =[float(a) for a in line.split(' ')[1:]]
    return precision_list, recall_list, FPR_list, TPR_list






if __name__ == '__main__':
    precision_list, recall_list, FPR_list, TPR_list = read_result_file("../../Data/HSR.LSTM.ATTENTION.TrueCNN.MGH.lower.emb.optSGD.wcut2.lr0.1.h200SGD.result.txt")
    # prec = [1, 0.9,0.8,0.4,0.2]
    # rec = [0,0.3,0.7,0.9,1]
    plot_precision_recall(precision_list, recall_list,"../../Data/prc.pdf")
    plot_roc(FPR_list, TPR_list,"../../Data/roc.pdf")
