# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-09-26 09:27:03



import time
import numpy as np
np.set_printoptions(threshold=np.nan)
from NCRFpp.ncrf import NCRF
from file_io import *
from utils import  clinic_text_list_processing, data_split_train_dev_test
from sklearn import metrics
import nltk
from metric_visual import *
import sys

def write_results(out_file, input_list, name_list, auc=None):
    print("Write results:...")
    fout = open(out_file,'w')
    if auc != None:
        fout.write("AUC: " + str(auc)+"\n")
    assert(len(input_list)==len(name_list))
    for name, the_list in zip(name_list, input_list):
        fout.write(name+": "+ " ".join([str(a) for a in the_list.tolist()])+"\n")
    print("Results are written into file:", out_file)


def nfold_hsr_arg(input_file, nfold=5, word_feature="FF", use_char = False, lr= 0.2, cutoff=5, pred="HSR"):
    r''' train random forest model based on train data
        Args:
            x (list): size[num_instance], list of strings (sentences/documents)
            y (list): size[num_instance], list of labels
            nfold (int): number of nfold
        Return:
    '''
    if pred == "ADR":
        x, y = load_text_classification_data_xlsx(input_file,"E",'C', True)
    else:
        x, y = load_text_classification_data_xlsx(input_file,"E",'D', True)
    # x = x[0:5000]
    # y = y[0:5000]
    
    print("Run nfold experiments for %s, nfold=%s"%(pred, nfold))

    test_decode_list = []
    test_gold_list = []
    for idx in range(nfold):
        start_time = time.time()
        print ("Proceesing: %s/%s.............."%(idx, nfold))
        ## notice the positive label id may vary in different fold, need fix the positive label id.
        train_x, dev_x, test_x = data_split_train_dev_test(x, nfold, idx)
        train_y, dev_y, test_y = data_split_train_dev_test(y, nfold, idx)
        train_list = convert_to_ncrf_list(train_x, train_y)
        dev_list = convert_to_ncrf_list(dev_x, dev_y)
        test_list = convert_to_ncrf_list(test_x, test_y)
        config_file = "NCRFpp/demo.clf.simple.config"
        ncrf = NCRF()
        ncrf.read_data_config_file(config_file)
        ncrf.data.word_feature_extractor = word_feature
        ncrf.data.use_char = use_char
        ncrf.data.word_emb_dir= "MGH.lower.emb"
        ncrf.data.word_cutoff = cutoff
        ncrf.data.optimizer = "SGD"
        ncrf.data.HP_lr = lr
        ncrf.data.char_feature_extractor = "CNN"
        ncrf.data.words2sent_representation = "ATTENTION"
        if pred == "ADR":
            ncrf.data.HP_iteration = 20
            if ncrf.data.word_feature_extractor == "CNN":
                ncrf.data.HP_iteration = 30
        else:
            if ncrf.data.word_feature_extractor == "CNN":
                ncrf.data.HP_iteration = 20
        model_name = ncrf.data.word_feature_extractor+"."+ncrf.data.words2sent_representation+"."+str(ncrf.data.use_char)+ncrf.data.char_feature_extractor+"."+ncrf.data.word_emb_dir+".opt"+ncrf.data.optimizer+".wcut"+str(ncrf.data.word_cutoff) +".lr"+str(ncrf.data.HP_lr)+".h"+str(ncrf.data.HP_hidden_dim)
        if pred == "ADR":
            model_name = "ADR."+model_name
        else:
            model_name = "HSR."+model_name
        model_name = model_name + ncrf.data.optimizer
        print("Model Name:", model_name)
        ncrf.data.model_dir = "mask_log/"+model_name+ ".model"
        ncrf.initialization([train_list, dev_list, test_list])
        ncrf.generate_instances_from_list(train_list,'train')
        ncrf.generate_instances_from_list(dev_list,'dev')
        test_Ids = ncrf.generate_instances_from_list(test_list,'test')
        ncrf.data.show_data_summary()
        ncrf.train()
        target_prob = ncrf.decode_prob(ncrf.data.test_Ids)
        test_decode_list.append(np.asarray(target_prob))
        test_gold_list.append(np.asarray(test_y))
        
    test_all_decode = np.concatenate(test_decode_list, axis = 0)
    test_all_gold = np.concatenate(test_gold_list, axis = 0)   

    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = test_all_decode[:,target_id]
    target_label = 1
    precision, recall, pr_thresholds = metrics.precision_recall_curve(test_all_gold, test_all_target_decode,target_label)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_all_gold, test_all_target_decode, target_label)
    auc = metrics.auc(fpr, tpr)

    plot_precision_recall(precision, recall, model_name, "mask_log/"+model_name+".prc.jpg")
    plot_roc(fpr, tpr, auc, model_name, "mask_log/"+model_name+".roc.jpg")
    write_results("mask_log/"+model_name+".result.txt", [precision, recall, fpr, tpr], ["Precision","Recall", "FPR", "TPR"], auc)
    print("Auc:", auc)
    fout = open("mask_log/all_results.txt", 'a')
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


if __name__ == '__main__':
    input_file = r"../../9K_reports_20181228.xlsx"
    # nfold_hsr_arg(input_file,5)
    # exit(0)
    word_feature=sys.argv[1] 
    if sys.argv[2]=="T":
    	use_char = True
    else:
    	use_char = False 
    lr= float(sys.argv[3]) 
    cutoff = int(sys.argv[4])
    task = sys.argv[5]
    nfold_hsr_arg(input_file, 5, word_feature, use_char, lr, cutoff,task)
    
    