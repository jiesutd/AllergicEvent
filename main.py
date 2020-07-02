# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-18 16:28:20

from __future__ import division
import numpy as np
import csv
import sys
import random
import time
from file_io import *
from model import Classifier
from utils import *
from metric_visual import *

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)

def examine_model(input_file, model_name='svm'):
    descriptions, HSR = load_text_classification_data_xlsx(input_file,"E",'D', True)
    decode_prob = True
    train_x, dev_x, test_x = descriptions[:6000], descriptions[6000:7000], descriptions[7000:]
    train_y, dev_y, test_y = HSR[:6000], HSR[6000:7000], HSR[7000:]
    the_model = train_model(train_x, train_y, model_name)
    dev_preds = the_model.decode_raw(dev_x, decode_prob)
    test_preds = the_model.decode_raw(test_x, decode_prob) 
    print("Test:", test_preds)
    precision, recall, pr_thresholds = calculate_precision_recall_list(test_y, test_preds, 1)
    fpr, tpr, roc_thresholds, auc = calculate_roc_list(test_y, test_preds, 1)
    # plot_roc(fpr, tpr, auc, "svm")
    prob_list, num_list = count_num_threshold(test_preds[:, the_model.positive_id])
    plot_multi_curve([prob_list], [num_list], ["svm"], "probability", "Number", False, True)
    # plot_multi_curve([recall], [precision], ["svm"],  "Recall","Precision")
    # plot_multi_curve([fpr], [tpr], ["svm"], "FPR", "TPR")
    # plot_multi_precision_recall([precision], [recall], ["svm"])

def examine_models(input_file, model_list):
    descriptions, HSR = load_text_classification_data_xlsx(input_file,"E",'D', True)
    train_x, dev_x, test_x = descriptions[:6000], descriptions[6000:7000], descriptions[7000:]
    train_y, dev_y, test_y = HSR[:6000], HSR[6000:7000], HSR[7000:]
    decode_prob = True
    fpr_list = []
    tpr_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    for each_model in model_list:
        the_model = train_model(train_x, train_y, each_model)
        dev_preds = the_model.decode_raw(dev_x, decode_prob)
        test_preds = the_model.decode_raw(test_x, decode_prob) 
        precision, recall, pr_thresholds = calculate_precision_recall_list(test_y, test_preds, 1)
        fpr, tpr, roc_thresholds, auc = calculate_roc_list(test_y, test_preds, 1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
        precision_list.append(precision)
        recall_list.append(recall)
    # plot_roc(fpr, tpr, auc, "svm")
    plot_multi_roc(fpr_list, tpr_list, auc_list, model_list)
    # plot_multi_curve([recall], [precision], model_list,  "Recall","Precision")
    plot_multi_precision_recall([precision],[recall],model_list)


def examine_nfold_models(input_file, model_list, nfold=10):
    X, Y = load_text_classification_data_xlsx(input_file,"E","D", True)

    print('\tLabel distribution:')
    show_distribution(Y)
    fpr_list = []
    tpr_list = []
    auc_list = []
    precision_list = []
    recall_list = [] 
    for each_model in model_list:
        dev_probs, dev_y, test_probs,  test_y = nfold_model(X, Y, each_model, nfold, True)
        precision, recall, pr_thresholds = calculate_precision_recall_list(test_y, test_probs, 1)
        fpr, tpr, roc_thresholds, auc = calculate_roc_list(test_y, test_probs, 1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
        precision_list.append(precision)
        recall_list.append(recall)
    # plot_roc(fpr, tpr, auc, "svm")
    plot_multi_roc(fpr_list, tpr_list, auc_list, model_list)
    # plot_multi_curve(recall_list, precision_list, model_list,  "Recall", "Precision")
    plot_multi_precision_recall(precision_list,recall_list,model_list)
    fout = open("HSR.statistical.result_prft.txt",'w')
    model_num = len(model_list)
    for p, r, fpr, tpr, auc, name in zip(precision_list,recall_list, fpr_list, tpr_list,auc_list, model_list):
        fout.write(name+": " + str(auc)+"\n")
        p_str = [str(a) for a in p]
        r_str = [str(a) for a in r]
        fpr_str = [str(a) for a in fpr]
        tpr_str = [str(a) for a in tpr]
        fout.write("Precision: "+ " ".join(p_str)+"\n")
        fout.write("Recall: " + " ".join(r_str)+"\n")
        fout.write("FPR: " + " ".join(fpr_str)+"\n")
        fout.write("TPR: " + " ".join(tpr_str)+"\n")






def train_model(x, y, model_name, use_bigram=False, clean_text=True):
    r''' train random forest model based on train data
        Args:
            x (list): training inputs, a list of strings (sentences/documents)
            y (list ): training labels, a list of labels
            clean_text: if clean clinic text
            parameters (tuple): input parameters (n_estimators, random_state)
        Return:
            the_model (model): the classifier model 
            dev_y/test_y (list) gold label list.
            dev_probs/test_probs (list of list) are decoded list of probability. [numberOfInstances, labelNum]
    '''
    ## build label alphabet
    
    ## extract feature
    positive_label = 1
    the_model = Classifier(model_name, use_bigram, positive_label, clean_text)
    train_features = the_model.generate_train_features(x)

    ## prepare label -> id 
    the_model.build_label_alphabet(y)
    train_label_array = the_model.get_label_id(y)
    ## train model
    the_model.train(train_features, train_label_array)
    return the_model




def nfold_model(x, y, model_name, nfold=10, decode_prob=False):
    r''' train random forest model based on train data
        Args:
            x (list): size[num_instance], list of strings (sentences/documents)
            y (list): size[num_instance], list of labels
            nfold (int): number of nfold
        Return:
    '''
    print("Run nfold experiments, nfold=%s"%nfold)
    dev_decode_list = []
    dev_gold_list = []
    test_decode_list = []
    test_gold_list = []
    for idx in range(nfold):
        start_time = time.time()
        print ("Proceesing: %s/%s.............."%(idx, nfold))
        ## notice the positive label id may vary in different fold, need fix the positive label id.
        train_x, dev_x, test_x = data_split_train_dev_test(x, nfold, idx)
        train_y, dev_y, test_y = data_split_train_dev_test(y, nfold, idx)
        the_model = train_model(train_x, train_y, model_name)
        dev_preds = the_model.decode_raw(dev_x, decode_prob)
        test_preds = the_model.decode_raw(test_x, decode_prob)
        dev_decode_list.append(dev_preds)
        dev_gold_list.append(dev_y)
        test_decode_list.append(test_preds)
        test_gold_list.append(test_y)
        cost_time = time.time() - start_time
        print("     Time cost: %.2f s"% cost_time)
    dev_all_decode = np.concatenate(dev_decode_list, axis = 0)
    dev_all_gold = np.concatenate(dev_gold_list, axis = 0)
    test_all_decode = np.concatenate(test_decode_list, axis = 0)
    test_all_gold = np.concatenate(test_gold_list, axis = 0)    
    return dev_all_decode, dev_all_gold, test_all_decode, test_all_gold


def multi_train_decode(train_file, decode_file,  output_file, model_name=['svm'], model_file=None):
    positive_id = 1
    x, y = load_text_classification_data_xlsx(train_file,"C","B", True)
    train_dict = {}
    for a in x:
        if a not in train_dict:
            train_dict[a] = 1
    model_list = []
    for each_model in model_name:
        the_model = train_model(x, y, each_model, False, False)
        if model_file:
            the_model.save(model_file+"_"+each_model+".model")
        model_list.append(the_model)
    record_id_list, input_x = load_pair_txt_file(decode_file)
    # [input_x] = load_text_classification_data_csv(decode_file, [8], False, True)
    print("Decode instance number:", len(input_x))

    decode_list = []
    for each_model in model_list:
        decode_probs = each_model.decode_raw(input_x, True)
        decode_list.append(decode_probs[:, positive_id])
    ## filter decode instance overlap in training data
    x_number = len(input_x)
    model_num = len(decode_list)
    new_input_x = []
    new_decode_list = [[] for a in range(model_num)]
    for idx in range(x_number):
        if input_x[idx] not in train_dict:
            new_input_x.append(input_x[idx])
            for idy in range(model_num):
                new_decode_list[idy].append(decode_list[idy][idx])
    print("Decode instance: %s, filter training data: %s"%(x_number, len(new_input_x)))

    write_multi_decode_to_csv(output_file, new_input_x, new_decode_list, model_name)





if __name__ == '__main__':

    input_file = r"../../9K_reports_20181228.xlsx"
    # input_file = r"../extended_annotated_reports_20190305.xlsx"
    examine_nfold_models(input_file,['svm','logistic','randomforest','xgboost'], 5)
    exit(0)
    # decode_file = r"../report_from_elgkeywords-annotated.txt")
    extend_lexicon_data = r"../keyvalue_desc_174799cases_0118019_mtermlexicon_remove_annotated_cases_4.csv"
    all_data = r'../report_from_final-annotated.txt'
    multi_train_decode(input_file, all_data,"mtermslexicon_extendtraining.csv", ['svm','logistic','randomforest', 'xgboost'], 'ext_training_model' )
    # examine_models(input_file,['svm','logistic'])
    exit(0)
    # multi_train_decode(input_file, decode_file, "test1.csv", ['svm','logistic'])
    # another_file = 'Data_For_Machine_Learing.xlsx'
    # examine_nfold_models(another_file,['svm','logistic','randomforest','xgboost'], 5)
    # exit(0)  
    # validation_file = "annotation_evaluation_150cases_NP.csv"
    # extend_lexicon_data = "keyvalue_desc_174799cases_0118019_mtermlexicon_remove_annotated_cases_4.csv"
    # file_after_spelling_check = "9k_HSR_reports_20181228_filterduplicate.txt"
    









