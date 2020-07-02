# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-22 11:35:27



import time
import numpy as np
np.set_printoptions(threshold=np.nan)
from NCRFpp_multitask.ncrf import NCRF
from file_io import *
from utils import  clinic_text_list_processing, data_split_train_dev_test
from sklearn import metrics
import nltk
from metric_visual import *
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random

def write_results(out_file, input_list, name_list, auc=None):
    print("Write results:...")
    fout = open(out_file,'w')
    if auc != None:
        fout.write("AUC: " + str(auc)+"\n")
    assert(len(input_list)==len(name_list))
    for name, the_list in zip(name_list, input_list):
        fout.write(name+": "+ " ".join([str(a) for a in the_list.tolist()])+"\n")
    print("Results are written into file:", out_file)


def nfold_hsr_multi(input_file, nfold=5, word_feature="FF", use_char = False, lr= 0.2, cutoff=2, iteration=10):
    r''' train random forest model based on train data
        Args:
            x (list): size[num_instance], list of strings (sentences/documents)
            y (list): size[num_instance], list of labels
            nfold (int): number of nfold
        Return:
    '''
    hsr_x, hsr_y = load_text_classification_data_xlsx(input_file,"E",'D', False)
    adr_x, adr_y = load_text_classification_data_xlsx(input_file,"E",'C', False)
    hsr_dict = {}
    adr_dict = {}
    for x, y in zip(hsr_x, hsr_y):
        hsr_dict[x] = y
    for x, y in zip(adr_x, adr_y):
        adr_dict[x] = y

    new_x = []
    hsr_y = []
    adr_y = []
    for x in adr_x:
        if x in hsr_dict:
            new_x.append(x)
            hsr_y.append(hsr_dict[x])
            adr_y.append(adr_dict[x])
    combined = list(zip(new_x, hsr_y, adr_y))
    random.shuffle(combined)
    new_x[:], hsr_y[:], adr_y[:] = zip(*combined)
    hsr_x = new_x 
    adr_x = new_x

    print("hsr:",len(hsr_y))
    print("adr:",len(adr_y))
    # x = x[0:5000]
    # y = y[0:5000]
    
    print("Run nfold experiments, nfold=%s"%nfold)

    test_decode_list = []
    test_gold_list = []
    new_test_decode_list = []
    new_test_gold_list = []
    for idx in range(nfold):
        start_time = time.time()
        print ("Proceesing: %s/%s.............."%(idx, nfold))
        ## notice the positive label id may vary in different fold, need fix the positive label id.
        train_hsr_x, dev_hsr_x, test_hsr_x = data_split_train_dev_test(hsr_x, nfold, idx)
        train_hsr_y, dev_hsr_y, test_hsr_y = data_split_train_dev_test(hsr_y, nfold, idx)
        train_adr_x, dev_adr_x, test_adr_x = data_split_train_dev_test(adr_x, nfold, idx)
        train_adr_y, dev_adr_y, test_adr_y = data_split_train_dev_test(adr_y, nfold, idx)

        train_hsr_list = convert_to_ncrf_list(train_hsr_x, train_hsr_y)
        dev_hsr_list = convert_to_ncrf_list(dev_hsr_x, dev_hsr_y)
        test_hsr_list = convert_to_ncrf_list(test_hsr_x, test_hsr_y)

        train_adr_list = convert_to_ncrf_list(train_adr_x, train_adr_y)
        dev_adr_list = convert_to_ncrf_list(dev_adr_x, dev_adr_y)
        test_adr_list = convert_to_ncrf_list(test_adr_x, test_adr_y)

        config_file = "NCRFpp/demo.clf.simple.config"
        ncrf = NCRF()
        ncrf.data.multitask = True
        ncrf.read_data_config_file(config_file)
        ncrf.data.word_feature_extractor = "LSTM"
        ncrf.data.word_feature_extractor = word_feature
        ncrf.data.use_char = use_char
        ncrf.data.word_emb_dir= "MGH.lower.emb"
        ncrf.data.word_cutoff = cutoff
        ncrf.data.HP_lr = lr
        ncrf.data.HP_iteration = iteration

        # exit(0)
        # ncrf.data.optimizer = "Adam"
        # ncrf.data.norm_word_emb = True
        # ncrf.data.model_dir = "multitask/debug"
        ncrf.data.model_dir = "multitask/"+ncrf.data.word_feature_extractor+"."+str(ncrf.data.use_char)+"."+ncrf.data.word_emb_dir+".opt"+ncrf.data.optimizer+".wcut"+str(ncrf.data.word_cutoff) + ".model"
        ncrf.initialization_multi([train_hsr_list, dev_hsr_list, test_hsr_list], [train_adr_list, dev_adr_list, test_adr_list])
        ncrf.data.show_data_summary()

        train_Ids = ncrf.generate_instance_from_list(train_hsr_list,'train')
        ncrf.generate_instance_from_list(dev_hsr_list,'dev')
        ncrf.generate_instance_from_list(test_hsr_list,'test')

        new_train_Ids = ncrf.generate_instance_from_list(train_adr_list,'train',"new")
        ncrf.generate_instance_from_list(dev_adr_list,'dev',"new")
        ncrf.generate_instance_from_list(test_adr_list,'test',"new")
        ncrf.data.show_data_summary()
        # ncrf.data.HP_iteration = 3

        best_model, new_best_model = ncrf.train(train_Ids, new_train_Ids)
        print("Best model:", best_model)
        print("New best model:", new_best_model)
        ncrf.load(best_model)
        target_prob = ncrf.decode_prob(ncrf.data.test_Ids, "new")
        test_decode_list.append(np.asarray(target_prob))
        test_gold_list.append(np.asarray(test_hsr_y))
        ncrf.load(new_best_model)
        new_target_prob = ncrf.decode_prob(ncrf.data.new_test_Ids, "new")
        new_test_decode_list.append(np.asarray(new_target_prob))
        new_test_gold_list.append(np.asarray(test_adr_y))
        
    test_all_decode = np.concatenate(test_decode_list, axis = 0)
    test_all_gold = np.concatenate(test_gold_list, axis = 0)  
    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = test_all_decode[:,target_id]
    target_label = 1
    precision, recall, pr_thresholds = metrics.precision_recall_curve(test_all_gold, test_all_target_decode,target_label)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_all_gold, test_all_target_decode, target_label)
    auc = metrics.auc(fpr, tpr)
    if ncrf.data.use_char:
        model_name = ncrf.data.word_feature_extractor+"+"+ncrf.data.char_feature_extractor+"."+str(ncrf.data.word_emb_dir)+".iter"+str(ncrf.data.HP_iteration)+".opt"+ncrf.data.optimizer+".lr"+str(ncrf.data.HP_lr)
    else:
        model_name = ncrf.data.word_feature_extractor+"nochar."+str(ncrf.data.word_emb_dir)+"+.iter"+str(ncrf.data.HP_iteration)+".opt"+ncrf.data.optimizer+".lr"+str(ncrf.data.HP_lr)
    model_name = "hsr."+model_name
    plot_precision_recall(precision, recall, model_name, "multitask/"+model_name+".prc.jpg")
    plot_roc(fpr, tpr, auc, model_name, "multitask/"+model_name+".roc.jpg")
    write_results("multitask/"+model_name+".result.txt", [precision, recall, fpr, tpr], ["Precision","Recall", "FPR", "TPR"], auc)
    print("HSR Auc:", auc)
    fout = open("multitask/all_results.txt", 'a')
    fout.write(model_name+":"+str(auc)+"\n")
    fout.close()

    new_test_all_decode = np.concatenate(new_test_decode_list, axis = 0)
    new_test_all_gold = np.concatenate(new_test_gold_list, axis = 0)  
    target_id = ncrf.data.new_label_alphabet.get_index('1')
    test_all_target_decode = new_test_all_decode[:,target_id]
    target_label = 1
    precision, recall, pr_thresholds = metrics.precision_recall_curve(new_test_all_gold, test_all_target_decode,target_label)
    fpr, tpr, roc_thresholds = metrics.roc_curve(new_test_all_gold, test_all_target_decode, target_label)
    auc = metrics.auc(fpr, tpr)
    if ncrf.data.use_char:
        model_name = ncrf.data.word_feature_extractor+"+"+ncrf.data.char_feature_extractor+"."+str(ncrf.data.word_emb_dir)+".iter"+str(ncrf.data.HP_iteration)+".opt"+ncrf.data.optimizer+".lr"+str(ncrf.data.HP_lr)
    else:
        model_name = ncrf.data.word_feature_extractor+"nochar."+str(ncrf.data.word_emb_dir)+"+.iter"+str(ncrf.data.HP_iteration)+".opt"+ncrf.data.optimizer+".lr"+str(ncrf.data.HP_lr)
    model_name = "adr."+model_name
    plot_precision_recall(precision, recall, model_name, "multitask/"+model_name+".prc.jpg")
    plot_roc(fpr, tpr, auc, model_name, "multitask/"+model_name+".roc.jpg")
    write_results("multitask/"+model_name+".result.txt", [precision, recall, fpr, tpr], ["Precision","Recall", "FPR", "TPR"], auc)
    print("ADR Auc:", auc)
    fout = open("multitask/all_results.txt", 'a')
    fout.write(model_name+":"+str(auc)+"\n")
    fout.close()

    return test_all_target_decode, test_all_gold



    

def convert_to_ncrf_list(sent_list, label_list):
    word_list = []
    feature_list = []
    strlabel_list = []
    for sent, label in zip(sent_list, label_list):
        words = nltk.word_tokenize(sent)
        word_list.append(words)
        strlabel_list.append(str(label))
        feature_list.append([])
    return [word_list, strlabel_list, feature_list]


def convert_multi_to_ncrf_list(sent_list, label_list, new_label_list):
    word_list = []
    feature_list = []
    strlabel_list = []
    newlabel_list = []
    for sent, label, new_label in zip(sent_list, label_list, new_label_list):
        words = nltk.word_tokenize(sent)
        word_list.append(words)
        strlabel_list.append(str(label))
        newlabel_list.append(str(new_label))
        feature_list.append([])
    return [word_list, [strlabel_list,newlabel_list], feature_list]


if __name__ == '__main__':
    input_file = r"../../9K_reports_20181228.xlsx"
    nfold_hsr_multi(input_file, 5, "FF")
    exit(0)
    word_feature=sys.argv[1] 
    if sys.argv[2]=="T":
        use_char = True
    else:
        use_char = False 
    lr = float(sys.argv[3]) 
    cutoff = int(sys.argv[4])
    iteration = int(sys.argv[5])
    nfold_hsr_multi(input_file, 5, word_feature, use_char, lr, cutoff, iteration)
    
    