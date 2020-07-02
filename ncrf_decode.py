# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-29 15:46:54



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



def train_decode(input_file):
    x, y = load_text_classification_data_xlsx(input_file,"E",'D', True)
    train_x, dev_x, test_x = x[:5000], x[5000:6500], x[6500:]
    train_y, dev_y, test_y = y[:5000], y[5000:6500], y[6500:]
    train_list = convert_to_ncrf_list(train_x, train_y)
    dev_list = convert_to_ncrf_list(dev_x, dev_y)
    test_list = convert_to_ncrf_list(test_x, test_y)
    config_file = "NCRFpp/demo.clf.simple.config"
    ncrf = NCRF()
    ncrf.read_data_config_file(config_file)
    ncrf.data.word_feature_extractor = "LSTM"
    ncrf.data.use_char = "CNN"
    ncrf.data.word_emb_dir= "MGH.lower.emb"
    ncrf.data.word_cutoff = 2
    ncrf.data.optimizer = "SGD"
    ncrf.data.HP_lr = 0.1
    ncrf.data.words2sent_representation = "ATTENTION"
    ncrf.data.HP_iteration = 2
    model_name = ncrf.data.word_feature_extractor+"."+ncrf.data.words2sent_representation+"."+str(ncrf.data.use_char)+"."+ncrf.data.word_emb_dir+".opt"+ncrf.data.optimizer+".wcut"+str(ncrf.data.word_cutoff) +".lr"+str(ncrf.data.HP_lr)+".h"+str(ncrf.data.HP_hidden_dim)
    # ncrf.data.model_dir = "mask_log/"+model_name+ ".model"
    ncrf.initialization([train_list, dev_list, test_list])
    ncrf.generate_instances_from_list(train_list,'train')
    ncrf.generate_instances_from_list(dev_list,'dev')
    test_Ids = ncrf.generate_instances_from_list(test_list,'test')
    ncrf.data.show_data_summary()
    ncrf.train(None, "model/att.model")
    ncrf.decode_prob_and_attention_weights(test_Ids)


    

def adr_hsr_decode(model_file, decode_file, output_file):
    print("Load mode from  full model ...", model_file)

    test_x, test_y = load_text_classification_data_xlsx(input_file,"E",'D', True)
    test_x = test_x[:1000]
    testy = test_y[:1000]
    # for the_id, the_x in zip(record_id_list, test_x):
    decode_num = len(test_x)
    print("original decode num:", decode_num)
    
    test_list = convert_to_ncrf_list(test_x, test_y)

    ncrf = NCRF()
    ncrf.load(model_file)
    print("Model Loaded.")
    ncrf.data.MAX_SENTENCE_LENGTH = -1
    # ncrf.data.HP_batch_size = 100
    test_Ids = ncrf.generate_instances_from_list(test_list,'test')
    ncrf.data.show_data_summary()
    target_prob, weights = ncrf.decode_prob_and_attention_weights(test_Ids)
    decode_num = len(weights)
    target_id = ncrf.data.label_alphabet.get_index('1')
    test_all_target_decode = target_prob[:,target_id]
    print(type(test_all_target_decode))
    for idx in range(decode_num):
        if test_all_target_decode[idx] > 0.5:
            sent_len = len(ncrf.data.test_texts[idx][0])
            out_string = ""
            the_weights = weights[idx][:sent_len]
            the_weights, strong_index = standard_vector(the_weights)
            for idy in range(sent_len):
                out_string += ncrf.data.test_texts[idx][0][idy] +"|"+'%.3f'%float(the_weights[idy])+", "
            out_string += str(strong_index)
            if sent_len < 400:
                if "rash" not in ncrf.data.test_texts[idx][0] and "rashs" not in ncrf.data.test_texts[idx][0] and "hive" not in ncrf.data.test_texts[idx][0] and "hives" not in ncrf.data.test_texts[idx][0]:
                    print(out_string+"\n")

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)
    return rescale

def standard_vector(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_avg = np.mean(the_array)
    the_std = np.std(the_array)
    rescale = (the_array - the_avg)/the_std
    strong_index = False
    if the_max-the_avg > 4*the_std:
        strong_index = True
    return rescale, strong_index


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
    the_model = "model/att.model.model"
    adr_hsr_decode(the_model,input_file,None)
    # train_decode(input_file)
    