# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-05-05 14:05:49
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-05 11:52:26

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
from ml_data_io import *

reload(sys)  
sys.setdefaultencoding('utf8')

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




def examine_nfold_models(input_file, model_list, nfold=10):
    X, Y_old = load_text_classification_data_csv(train_file, [1,0], True)
    Y = []
    for y in Y_old:
        Y.append(int(y))
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
    plot_multi_curve(recall_list, precision_list, model_list,  "Recall","Precision")
    plot_multi_precision_recall(precision_list,recall_list,model_list)



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


def ml_data_train_multiple_model(train_file, decode_file, output_file, model_name=['svm'], model_file=None):
    positive_id = 1
    X, Y_old = load_text_classification_data_csv(train_file, [1,0], True)
    Y = []
    for y in Y_old:
        Y.append(int(y))
    model_list = []
    for each_model in model_name:
        the_model = train_model(X, Y, each_model, False, False)
        if model_file:
            the_model.save(model_file+"_"+each_model+".model")
        model_list.append(the_model)
    # record_id_list, record_list = load_pair_txt_file(decode_file)
    input_x, other, other_name_List = load_ml_data_txt(decode_file)
    print("Decode instance number:", len(input_x))

    decode_list = []
    for each_model in model_list:
        decode_probs = each_model.decode_raw(input_x, True)
        decode_list.append(decode_probs[:, positive_id])
    write_ml_multi_decode_to_csv(output_file, input_x, decode_list, model_name, other_name_List, other )


def combine_annotated(origin_file, new_file, conflict_file,  output_file):
    x, y = load_text_classification_data_xlsx(origin_file, "E","J", True)
    new_x, new_y = load_text_classification_data_xlsx(new_file, "L","P", True)
    con_x, con_y = load_text_classification_data_xlsx(conflict_file, "D","A", True)
    combine_dict = {}
    for a, b in zip(x,y):
        if a not in combine_dict:
            combine_dict[a] = b
    print("First dict size:", len(combine_dict))
    conflict_dict = {}
    for a, b in zip(con_x, con_y):
        if a not in conflict_dict:
            conflict_dict[a] = b
    print("conflict:", len(conflict_dict))
    

    for a, b in zip(new_x,new_y):
        if a not in combine_dict:
            combine_dict[a] = b
        else:
            if b != combine_dict[a]:
                if a not in conflict_dict:
                    print(a)
                else:   
                    combine_dict[a] = conflict_dict[a]
    print("Combine dict size:", len(combine_dict))
    with open(output_file,'w') as cfile:
        fwriter = csv.writer(cfile)
        fwriter.writerow(["Label", "Text"])
        for k, v in combine_dict.iteritems():
            fwriter.writerow([v, k])
        




if __name__ == '__main__':
    # old_an = 'Data_For_Machine_Learing.xlsx'
    # new_an = "Data_Combine_For_ML.xlsx"
    # conflict_file = "Resolved_Conflict.xlsx"
    # combine_annotated(old_an, new_an, conflict_file, "Data_merged.csv")
    # exit(0)
    model_list = ['svm','logistic','randomforest','xgboost']
    train_file = 'Data_merged.csv'
    # examine_nfold_models(train_file, model_list, 5)
    # exit(0)
    # decode_file = '/Volumes/ORADE/NotesFromJulie/ORADE NLP NoteTextData_Grp_5.txt'
    decode_file = "/Users/Jie/NoteTextData_Grp_all.txt"
    output_file = "Decode_MergeTraining636_NoteTextData_Grp_all.csv"
    
    # model_list = ['logistic']
    ml_data_train_multiple_model(train_file, decode_file, output_file, model_list)
    exit(0)
    validation_file = "annotation_evaluation_150cases_NP.csv"
    extend_lexicon_data = "keyvalue_desc_174799cases_0118019_mtermlexicon_remove_annotated_cases_4.csv"
    file_after_spelling_check = "9k_HSR_reports_20181228_filterduplicate.txt"
    









