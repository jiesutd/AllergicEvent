# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-07-08 12:07:56



import time
import numpy as np
np.set_printoptions(threshold=2**31)
from NCRFpp.ncrf import NCRF
from file_io import *
from utils import  clinic_text_list_processing, data_split_train_dev_test
from sklearn import metrics
import nltk
from metric_visual import *
import sys
import xlsxwriter

def write_results(out_file, input_list, name_list, auc=None):
    print("Write results:...")
    fout = open(out_file,'w')
    if auc != None:
        fout.write("AUC: " + str(auc)+"\n")
    assert(len(input_list)==len(name_list))
    for name, the_list in zip(name_list, input_list):
        fout.write(name+": "+ " ".join([str(a) for a in the_list.tolist()])+"\n")
    print("Results are written into file:", out_file)


def adr_hsr_train_all(input_file, pred="HSR"):
    pred = pred.upper()
    print("Pred:", pred)
    if pred == "ADR":
        x, y = load_text_classification_data_xlsx(input_file,"E",'C', True)
    else:
        x, y = load_text_classification_data_xlsx(input_file,"E",'D', True)
    print("Train full model ...", pred)
    total_instance_num = len(x)
    train_num = int(0.8*total_instance_num)
    train_x = x
    train_y = y
    dev_x = test_x = x[train_num:]
    dev_y = test_y = y[train_num:]
    decode_num = len(test_x)
    print("original decode num:", decode_num)

    train_list = convert_to_ncrf_list(train_x, train_y)
    dev_list = convert_to_ncrf_list(dev_x, dev_y)
    test_list = convert_to_ncrf_list(test_x, test_y)

    config_file = "NCRFpp/adr_hsr_final.config"
    ncrf = NCRF()
    ncrf.read_data_config_file(config_file)
    ncrf.data.word_feature_extractor = "LSTM"
    ncrf.data.use_char = "CNN"
    ncrf.data.word_emb_dir= "MGH.lower.emb"
    ncrf.data.word_cutoff = 2
    ncrf.data.optimizer = "SGD"
    ncrf.data.HP_lr = 0.1
    ncrf.data.words2sent_representation = "ATTENTION"
    model_name = pred+".att.all_model.trainall."+ncrf.data.word_feature_extractor+"."+str(ncrf.data.use_char)+"."+ncrf.data.word_emb_dir+".opt"+ncrf.data.optimizer+".wcut"+str(ncrf.data.word_cutoff) +".lr"+str(ncrf.data.HP_lr)+".h"+str(ncrf.data.HP_hidden_dim)
    if pred == "ADR":
        ncrf.data.HP_iteration = 20
    print("Model Name:", model_name)
    ncrf.data.model_dir = "decode/"+model_name+ ".model"
    ncrf.initialization([train_list, dev_list, test_list])
    ncrf.generate_instances_from_list(train_list,'train')
    ncrf.generate_instances_from_list(dev_list,'dev')
    test_Ids = ncrf.generate_instances_from_list(test_list,'test')
    ncrf.data.show_data_summary()
    ncrf.train()
    # ncrf.save(ncrf.data.model_dir)
    target_prob = ncrf.decode_prob(ncrf.data.test_Ids)
    test_all_decode = np.asarray(target_prob)
    test_all_gold = np.asarray(test_y) 

    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = test_all_decode[:,target_id]
    target_label = 1
    precision, recall, pr_thresholds = metrics.precision_recall_curve(test_all_gold, test_all_target_decode,target_label)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_all_gold, test_all_target_decode, target_label)
    auc = metrics.auc(fpr, tpr)
    print("Auc:", auc)
    decode_prob = test_all_target_decode.tolist()
    print("decode x: %s, decode y: %s"%(len(test_x), len(decode_prob)))
    
    return 0



def adr_hsr_train_decode(input_file, decode_file, output_file, pred="HSR"):
    pred = pred.upper()
    print("Pred:", pred)
    if pred == "ADR":
        x, y = load_text_classification_data_xlsx(input_file,"E",'C', True)
    else:
        x, y = load_text_classification_data_xlsx(input_file,"E",'D', True)
    print("Train full model ...", pred)
    total_instance_num = len(x)
    train_num = int(0.8*total_instance_num)
    train_x = x
    train_y = y
    dev_x = x[train_num:]
    dev_y = y[train_num:]
    record_id_list, test_x = load_pair_txt_file(decode_file)
    decode_num = len(test_x)
    print("original decode num:", decode_num)
    test_y = [1 for idx in range(decode_num)]

    train_list = convert_to_ncrf_list(train_x, train_y)
    dev_list = convert_to_ncrf_list(dev_x, dev_y)
    test_list = convert_to_ncrf_list(test_x, test_y)

    config_file = "NCRFpp/adr_hsr_final.config"
    ncrf = NCRF()
    ncrf.read_data_config_file(config_file)
    ncrf.data.word_feature_extractor = "LSTM"
    ncrf.data.use_char = "CNN"
    ncrf.data.word_emb_dir= "MGH.lower.emb"
    ncrf.data.word_cutoff = 2
    ncrf.data.optimizer = "SGD"
    ncrf.data.HP_lr = 0.1
    ncrf.data.words2sent_representation = "ATTENTION"
    model_name = "att.all_model.trainall."+ncrf.data.word_feature_extractor+"."+str(ncrf.data.use_char)+"."+ncrf.data.word_emb_dir+".opt"+ncrf.data.optimizer+".wcut"+str(ncrf.data.word_cutoff) +".lr"+str(ncrf.data.HP_lr)+".h"+str(ncrf.data.HP_hidden_dim)
    if pred == "ADR":
        model_name = "ADR."+model_name
        ncrf.data.HP_iteration = 20
    print("Model Name:", model_name)
    ncrf.data.model_dir = "decode/"+model_name+ ".model"
    ncrf.initialization([train_list, dev_list, test_list])
    ncrf.generate_instances_from_list(train_list,'train')
    ncrf.generate_instances_from_list(dev_list,'dev')
    test_Ids = ncrf.generate_instances_from_list(test_list,'test')
    ncrf.data.show_data_summary()
    ncrf.train()
    ncrf.save(ncrf.data.model_dir)
    target_prob = ncrf.decode_prob(ncrf.data.test_Ids)
    test_all_decode = np.asarray(target_prob)
    test_all_gold = np.asarray(test_y) 

    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = test_all_decode[:,target_id]
    target_label = 1
    precision, recall, pr_thresholds = metrics.precision_recall_curve(test_all_gold, test_all_target_decode,target_label)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_all_gold, test_all_target_decode, target_label)
    auc = metrics.auc(fpr, tpr)

    # plot_precision_recall(precision, recall, model_name)
    # plot_roc(fpr, tpr, auc, model_name)
    print("Auc:", auc)
    decode_prob = test_all_target_decode.tolist()
    print("decode x: %s, decode y: %s"%(len(test_x), len(decode_prob)))

    output_file = "decode/"+pred + "." + output_file
    with open(output_file,'w') as cfile:
        wt = csv.writer(cfile)
        wt.writerow(["ID", "Gold", "Pred", "Des"])
        for idx in range(decode_num):
            wt.writerow([record_id_list, test_y[idx], decode_prob[idx], test_x[idx]])
    
    return test_all_target_decode, test_all_gold
    

def adr_hsr_decode(model_file, decode_file, output_file):
    print("Load mode from  full model ...", model_file)

    record_id_list, test_x = load_pair_txt_file(decode_file)
    x_id_dict = {}
    # for the_id, the_x in zip(record_id_list, test_x):
    decode_num = len(test_x)
    print("original decode num:", decode_num)
    test_y = [1 for idx in range(decode_num)]
    
    test_list = convert_to_ncrf_list(test_x, test_y)

    ncrf = NCRF()
    ncrf.load(model_file)
    print("Model Loaded.")
    ncrf.data.MAX_SENTENCE_LENGTH = -1
    # ncrf.data.HP_batch_size = 100
    test_Ids = ncrf.generate_instances_from_list(test_list,'test')
    ncrf.data.show_data_summary()
    target_prob = ncrf.decode_prob(test_Ids)
    test_all_decode = np.asarray(target_prob)

    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = test_all_decode[:,target_id]
    decode_prob = test_all_target_decode.tolist()
    print("decode x: %s, decode y: %s"%(len(test_x), len(decode_prob)))
    with open(output_file,'w') as cfile:
        wt = csv.writer(cfile)
        wt.writerow(["ID", "Gold", "Pred", "Des"])
        for idx in range(decode_num):
            wt.writerow([record_id_list[idx], '--', decode_prob[idx], test_x[idx]])

def adr_hsr_decode_attention(model_file, decode_file, output_file):
    print("Load mode from  full model ...", model_file)
    record_id_list, test_x = load_pair_txt_file(decode_file)
    x_id_dict = {}
    # for the_id, the_x in zip(record_id_list, test_x):
    decode_num = len(test_x)
    print("original decode num:", decode_num)
    test_y = [1 for idx in range(decode_num)]
    test_list = convert_to_ncrf_list(test_x, test_y)
    ncrf = NCRF()
    ncrf.load(model_file)
    print("Model Loaded.")
    ncrf.data.MAX_SENTENCE_LENGTH = -1
    # ncrf.data.HP_batch_size = 100
    test_Ids = ncrf.generate_instances_from_list(test_list,'test')
    ncrf.data.show_data_summary()
    target_prob, weights = ncrf.decode_prob_and_attention_weights(test_Ids)
    test_all_decode = np.asarray(target_prob)
    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = test_all_decode[:,target_id]
    decode_prob = test_all_target_decode.tolist()
    print("decode x: %s, decode y: %s"%(len(test_x), len(decode_prob)))

    workbook = xlsxwriter.Workbook(output_file)
    worksheet = workbook.add_worksheet()

    title = ["ID", "Gold", "Pred", "Des", "Des_attention", ">4Sigma Found"]
    title_num = len(title)
    for idx in range(title_num):
        worksheet.write_string(0, idx, title[idx])
    for idx in range(decode_num):
        sent_len = len(ncrf.data.test_texts[idx][0])
        the_weights = weights[idx][:sent_len]
        std_weights, over_2_sigma = standard_vector(the_weights)
        out_string = ""
        for idy in range(sent_len):
            out_string += ncrf.data.test_texts[idx][0][idy] +"|"+'%.3f'%float(std_weights[idy])+" "
        out_string = out_string.strip(", ").strip('\n')
        worksheet.write_string(idx+1, 0, str(record_id_list[idx]))
        worksheet.write_string(idx+1, 1, '--')
        worksheet.write_number(idx+1, 2, decode_prob[idx])
        worksheet.write_string(idx+1, 3, test_x[idx])
        worksheet.write_string(idx+1, 4, out_string)
        worksheet.write_string(idx+1, 5, str(over_2_sigma))
    workbook.close()


    # with open(output_file,'w',newline='') as cfile:
    #     wt = csv.writer(cfile)
    #     wt.writerow(["ID", "Gold", "Pred", "Des", "Des_attention", ">4Sigma Found"])
    #     for idx in range(decode_num):
    #         sent_len = len(ncrf.data.test_texts[idx][0])
    #         the_weights = weights[idx][:sent_len]
    #         std_weights, over_2_sigma = standard_vector(the_weights)
    #         out_string = ""
    #         for idy in range(sent_len):
    #             out_string += ncrf.data.test_texts[idx][0][idy] +"|"+'%.3f'%float(std_weights[idy])+" "
    #         out_string = out_string.strip(", ").strip('\n')
    #         wt.writerow([record_id_list[idx], '--', decode_prob[idx], test_x[idx], out_string, over_2_sigma])
    print("Finished!")



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


def standard_vector(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_avg = np.mean(the_array)
    the_std = np.std(the_array)
    rescale = (the_array - the_avg)/the_std
    over_4_sigma = False 
    if the_max-the_avg > 4*the_std:
        over_4_sigma = True
    return rescale, over_4_sigma


if __name__ == '__main__':

    input_file = r"../../9K_reports_20181228.xlsx"
    MGH_all_data = r'../Data/MGH_report_from_final-annotated.txt'
    MGH_full_data = r'../Data/MGH_keyvalue_desc_174799cases_04012019.txt'
    BWH_all_data = r'../Data/BWH_keyvalue_desc_124230_Cases_04012019.txt'
    test_decode = r'../Data/test.txt'
    HSR_model = "decode/HSR.att.all_model.trainall.LSTM.CNN.MGH.lower.emb.optSGD.wcut2.lr0.1.h200.model.model"
    ADR_model = "decode/ADR.att.all_model.trainall.LSTM.CNN.MGH.lower.emb.optSGD.wcut2.lr0.1.h200.model.model"
    # adr_hsr_decode_attention(HSR_model, test_decode, "decode/test.xlsx")
    # exit(0)
    # adr_hsr_train_decode(input_file, all_data, "att.decode.all.csv", sys.argv[1])
    # adr_hsr_train_all(input_file, sys.argv[1])
    if sys.argv[1] == "HSR":
        the_model = HSR_model
    else:
        the_model = ADR_model

    if sys.argv[2] == "MGH":
        decode = MGH_all_data
    elif sys.argv[2] == "MGHFULL":
        decode = MGH_full_data
    else:
        decode = BWH_all_data

    output = "decode/"+sys.argv[1] +"."+ sys.argv[2]+".att.trainall.decode.xlsx"

    adr_hsr_decode_attention(the_model, decode, output)
    