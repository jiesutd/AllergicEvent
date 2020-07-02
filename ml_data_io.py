# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-24 13:03:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-12 15:50:24
# -*- coding: utf-8 -*-
import csv
from utils import clinic_text_processing

def load_ml_data_txt(input_file, split_token="`"):
    r''' load text classification data in normal text format (not limited in .txt file)
        Args:
            input_file (string): input text file directory
            split_token (string): split the text based on split_token
        Return:
            x (list): the text list of all documents/sentences
            y (list): the label list of the corresponding documents/sentences. = [] when no split_token found in text.
    '''
    print("Start loading txt file from %s."%input_file)
    x = []
    y = []
    other = []
    other_name_list = []
    with open(input_file,'rb') as infile:
        fins = infile.readlines()
        other_name_list = fins[0].strip('\r\n').split(split_token, 5)[:-1]
        for line in fins[1:]:
            pair = line.strip('\r\n').split(split_token, 5)
            assert(len(pair)==6)

            x.append(clinic_text_processing(pair[-1].decode('windows-1252')))
            other.append(pair[:-1])
    instance_num = len(x)
    print("\tInstance Num: %s"%instance_num)
    return x, other, other_name_list

def write_ml_multi_decode_to_csv(csv_file, X, Y_list, title_list=[], other_title_list= [],other_name_list=[]):
    r''' write the decoded label with input to csv file
        Args:
            csv_file (string): output file directory
            X (list): list of input X
            Y_list (list): list of labels for different models
            title_list (list of string): name of differnet models
    '''
    model_num = len(Y_list)
    instance_num  = len(X)
    print(instance_num,len(Y_list[0]) )
    assert(instance_num==len(Y_list[0]))
    with open(csv_file, 'w') as cfile:
        csvwriter = csv.writer(cfile) 
        csvwriter.writerow(other_title_list + title_list+['Text'])
        # csvwriter.writerow(other_title_list + title_list)
        decode_list = other_name_list
        for idx in range(instance_num):
            for idy in range(model_num):
                decode_list[idx].append(Y_list[idy][idx])
            decode_list[idx].append(X[idx])
            # print(decode_list[idx])
            csvwriter.writerow(decode_list[idx])
    print("Decoded results has been saved into file %s"%csv_file)



    
