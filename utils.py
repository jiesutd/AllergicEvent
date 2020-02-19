# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-28 10:50:18
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2020-02-19 11:33:58
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import re


def show_distribution(input_list):
    the_dict = {}
    for a in input_list:
        if a in the_dict:
            the_dict[a] +=1
        else:
            the_dict[a] = 1 
    for key, value in sorted(the_dict.iteritems(), key=lambda kv: (kv[1],kv[0])):
        print("\t\t%s: %s" % (key, value))

def filter_duplicacte(input_list, indent='\t', filter_id = 0):
    ## remove duplicate instance based on input_list[0]
    item_num = len(input_list)
    original_size = len(input_list[filter_id])
    input_dict = {}
    output_list = [[] for idx in range(item_num)]
    for idx in range(original_size):
        if input_list[filter_id][idx] not in input_dict:
            input_dict[input_list[filter_id][idx]] = 1 
            for idy in range(item_num):
                output_list[idy].append(input_list[idy][idx])
        else:
            continue
    new_size = len(output_list[filter_id])
    print(indent+"Filter duplicate data size: %s -> %s" %(original_size, new_size))
    return output_list


def data_split(input_list, partition_num = 10, partition_id = 0):
    ## partition_id: where the training data start
    instance_num = len(input_list)
    partition_size = int(instance_num/partition_num)
    partition_list = []
    for idx in range(partition_num):
        start_id = idx*partition_size
        end_id = (idx+1) * partition_size
        if idx == partition_num -1:
            the_part = input_list[start_id:]
        else:
            the_part = input_list[start_id:end_id]
        partition_list.append(the_part)
    takeout_id =  (partition_num-2+partition_id)%partition_num
    dev = partition_list.pop(takeout_id)
    ## Notice: the original list changes after the dev pop.
    if takeout_id == partition_num -1:
        test = partition_list.pop(0)
    else:
        test = partition_list.pop(takeout_id)
    train = []
    for each_part in partition_list:
        train = train + each_part
    return train, dev, test


def data_split_train_dev_test(input_list, partition_num = 10, split_id = 0):
    r''' split a list into partition_num partitions, and split them as train/dev/test based on the split_id.
        The train/dev/test size ratio is : (partition_num-2)/1/1
        Args:
            input_list (list): list to be splitted
            partition_num (int): partition num to be splited
            split_id (int): the split boundary id of the partition, 0<=split_id<=partition_num
        Return:
            train (list): train instance 
            dev (list): dev instance
            test (list): test instance

    '''
    instance_num = len(input_list)
    partition_size = int(instance_num/partition_num)
    partition_list = []
    for idx in range(partition_num):
        start_id = idx*partition_size
        end_id = (idx+1) * partition_size
        if idx == partition_num -1:
            the_part = input_list[start_id:]
        else:
            the_part = input_list[start_id:end_id]
        partition_list.append(the_part)
    takeout_id =  (partition_num-2+split_id)%partition_num
    dev = partition_list.pop(takeout_id)
    ## Notice: the original list changes after the dev pop.
    if takeout_id == partition_num -1:
        test = partition_list.pop(0)
    else:
        test = partition_list.pop(takeout_id)
    train = []
    for each_part in partition_list:
        train = train + each_part
    return train, dev, test



def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def string_clean(input_string):
    return input_string.strip().replace("\n",' ').replace("  "," ")

def unigram2bigram(unigram_list):
    bigram_list = []
    for sent in unigram_list:
        words = sent.split()
        word_num = len(words)
        bisent = ""
        for idx in range(word_num):
            if idx == word_num -1:
                bisent += words[idx]+"END"
            else:
                bisent += words[idx]+words[idx+1]+" "
        bigram_list.append(bisent)
    return bigram_list


def build_alphabet(label_set, binary=False):
    if binary:
        label2id = {1:1, 0:0}
        id2label = {1:1, 0:0}
        return label2id, id2label
    label2id = {}
    id2label = {}
    label_id = 0 
    for label in label_set:
        if label not in label2id:
            label2id[label] = label_id
            id2label[label_id] = label 
            label_id +=1
    return label2id, id2label


def count_num_threshold(probability_list):
    threshold_list = [(x+0.)/100 for x in range(101)]
    num_list = []
    for thres in threshold_list:
        the_num = sum(float(i) > thres for i in probability_list)
        num_list.append(the_num)
    return threshold_list, num_list


def clinic_text_processing(clinic_text):
    # clinic_text.encode('utf-8').replace("&nbsp;", " ").replace("<P>","").replace("</P>", "").strip()
    # print(clinic_text.encode('utf-8'))
    clinic_text = re.sub(r'\<.*?\>', '', clinic_text)
    clinic_text = clinic_text.replace("\\xc2\\xa0", " ").replace("0xb0", " ").replace("\\xa0", " ").replace("&nbsp;", " ").replace("<P>","").replace("</P>", "").strip()
    clinic_text = clinic_text.replace("\n", "").replace("\r", "")
    return clinic_text
    if sys.version_info[0] < 3:
        return clinic_text.encode('utf-8',errors='ignore')
    else:
        return clinic_text

def clinic_text_list_processing(clinic_text_list):
    return [clinic_text_processing(a) for a in clinic_text_list]

def extract_report_from_csv(input_csv, report_file, out_txt):
    print("Start load csv file from %s."%input_csv)
    
    record_id_dict = {}
    with open(input_csv) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        idx = 0
        for row in spamreader:
            if idx == 0:
                idx+= 1
                continue
            if row[0] not in record_id_dict:
                record_id_dict[row[0]] = 1
    print("Entry num: %s"%(len(record_id_dict)))
    id2report = {}
    fins = open(report_file,'r').readlines()
    for line in fins:
        pair = line.split('\t',1)
        if len(pair) != 2:
            continue
        else:
            id2report[pair[0]] = pair[-1].replace('\n', ' ').replace("\r", ' ')

    print("Report map: %s"% (len(id2report)))
    fout = open(out_txt,'w')
    print("Write output: line:%s,  file:%s"%(len(record_id_dict),out_txt) )
    for each_id in record_id_dict.keys():
        if each_id in id2report:
            fout.write(each_id +"\t" + id2report[each_id]+"\n")
        else:
            print(id2report.keys())
            print(each_id)
            exit(0)

def load_all_decode_result(csv_file):
    xg_test = []
    rf_test = []
    svm_test = []
    log_test = []
    with open(csv_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        idx = 0
        for row in spamreader:
            if idx == 0:
                idx+= 1
                continue
            else:
                xg_test.append(row[3])
                rf_test.append(row[2])
                svm_test.append(row[0])
                log_test.append(row[1])
    input_list = []
    xg_distribution = count_num_threshold(xg_test)
    print(xg_distribution)
    
    input_list.append(list(xg_distribution)+["XGBOOST"])
    rf_distribution = count_num_threshold(rf_test)
    input_list.append(list(rf_distribution)+["RANDFOREST"])
    svm_distribution = count_num_threshold(svm_test)
    input_list.append(list(svm_distribution)+["SVM"])
    log_distribution = count_num_threshold(log_test)
    input_list.append(list(log_distribution)+["LOGISTIC"])
    plot_multiple_classification_results_pp(input_list)


def plot_multiple_classification_results_pp(input_list):
    ## input_list: list of [[threshold, num,  model],[threshold, num,  model],...]
    result_num = len(input_list)
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    plt.figure(1)
    plt.title('HSR Classification Results')
    ## Sentivisity-Specifitivity curve
    plt.title('PositiveNum-ProbabilityThreshold Curve ')

    for idx in range(result_num):
        print(input_list[idx])
        plt.plot(input_list[idx][0], input_list[idx][1], color_list[idx], label = '%s' % (input_list[idx][2]))
    # plt.plot([0, 1], [0, 1],'r--')
    plt.legend(loc='best')
    plt.xlim([0, 1.01])
    # plt.ylim([0, 1.01])
    plt.ylabel('#Predicted Positive Reports')
    plt.xlabel('Probability Threshold')
    plt.yscale('log')
    xtick = [(x+0.)/10 for x in range(0, 11)]
    # ytick = [y*10000 for y in range(0, 18)]
    plt.xticks(xtick)
    # plt.yticks(ytick)
    plt.grid()
    plt.show() 




if __name__ == '__main__':
    print("TCBox-utils.py")
    load_all_decode_result("all.csv")
