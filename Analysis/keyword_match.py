# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-24 13:03:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-11-15 13:58:46
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

random.seed(42)

lower_flag = True
print("Lower case:", lower_flag)

def clinic_text_processing(clinic_text):
    clinic_text = clinic_text.replace("\\xc2\\xa0", " ").replace("&nbsp;", " ").replace("<P>","").replace("</P>", "").strip()
    return clinic_text

def sent_token(input_sent, lower=False):
    replace_list = []#[",","?",".","!", ";", "-"]
    for comma in replace_list:
        input_sent = input_sent.replace(comma, " "+comma+" ")
    if lower:
        input_sent = input_sent.lower()
    return input_sent.split()


def load_origin_keyword_list(keyword_file):
    fins = open(keyword_file,'r').readlines()
    pre_list = []
    suf_list = []
    full_list = []
    for lin in fins:
        if lower_flag:
            lin = lin.lower()
        lin = lin.strip().strip(',').split(", ")
        for a in lin:
            a = a.strip()
            # print(a)
            if a[0] == '-':
                suf_list.append(a[1:])
            elif a[-1] == '-':
                pre_list.append(a[:-1])
            else:
                full_list.append(a)
    print("Prefix list num:",len(pre_list))
    print("Suffix list num:", len(suf_list))
    print("Full   list num:", len(full_list))
    return pre_list, suf_list, full_list


def keyword_match(sentence, pre_list, suf_list, full_list):
    sentence = clinic_text_processing(sentence)
    words = sent_token(sentence, lower_flag)
    for word in words:
        # print(word, type(word), full_list[0], type(full_list[0]))
        if word in full_list:
            return True 
        else:
            for pre in pre_list:
                if word.startswith(pre):
                    return True 
            for suf in suf_list:
                if word.endswith(suf):
                    return True 
    return False


def read_key2date(input_csv):
    df = pd.read_csv(input_csv)
    print(df.columns)
    key2date_dict = pd.Series(df.EVENTDATE.values,index=df.KeyValue).to_dict()
    return key2date_dict
    



def MGH_file_load_match(original_file, keyword_file):
    key2date = read_key2date("../../Data/MGH_key2date.csv")
    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_excel(original_file)
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    
    new_data = ori_data[:8700]
    selected_data = new_data.loc[new_data["Gold"]==1]
    print("Original data size:", ori_data.shape[0])
    print("True data size:", selected_data.shape[0])
    last_selected_id = selected_data.iloc[-1]["ID"]
    print(ori_data.loc[ori_data["ID"]==last_selected_id].index)
    keyword_true_id_list = []
    all_true_id_list = []
    all_matched_id_list = []

    ## match keywords in true cases
    all_true_matched_key_in_2016_count = 0
    all_true_matched_key_over_2016_count = 0
    for a in range(selected_data.shape[0]):
        sent = str(selected_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        the_id = str(selected_data.iloc[[a]]['ID']).split("    ", 1)[1]
        the_id = int(the_id.split('\n')[0])
        the_date = int(key2date[the_id].split()[0].replace("-",""))
        if match:
            keyword_true_id_list.append(the_id)
            if the_date < 20160301:
                all_true_matched_key_in_2016_count += 1
            else:
                all_true_matched_key_over_2016_count += 1
        all_true_id_list.append(the_id)
    ## match keywords in all cases
    all_matched_key_in_2016_count = 0
    all_matched_key_over_2016_count = 0
    for a in range(ori_data.shape[0]):
        sent = str(ori_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        the_id = str(ori_data.iloc[[a]]['ID']).split("    ", 1)[1]
        the_id = int(the_id.split('\n')[0])
        the_date = int(key2date[the_id].split()[0].replace("-",""))
        if match:
            all_matched_id_list.append(the_id)
            if the_date < 20160301:
                all_matched_key_in_2016_count += 1
            else:
                all_matched_key_over_2016_count += 1
    ## count number in all manual reviewed
    all_manual_in_2016_count = 0
    all_manual_over_2016_count = 0
    for a in range(new_data.shape[0]):
        sent = str(new_data.iloc[a]['Des'])
        the_id = str(new_data.iloc[[a]]['ID']).split("    ", 1)[1]
        the_id = int(the_id.split('\n')[0])
        the_date = int(key2date[the_id].split()[0].replace("-",""))
        if the_date < 20160301:
            all_manual_in_2016_count += 1
        else:
            all_manual_over_2016_count += 1
    print("MGH Manual review size: old(%s) + new(%s) = %s"%(all_manual_in_2016_count,all_manual_over_2016_count, all_manual_in_2016_count+all_manual_over_2016_count ))
    print("MGH Original data match keywords size: old(%s) + new(%s) = %s"%(all_matched_key_in_2016_count, all_matched_key_over_2016_count, len(all_matched_id_list)))
    print("MGH True data match keywords size: old(%s) + new(%s) = %s"%(all_true_matched_key_in_2016_count, all_true_matched_key_over_2016_count, len(keyword_true_id_list)))
    matched_true_in_2016_count = 0
    matched_true_over_2016_count = 0
    for each_id in keyword_true_id_list:
        # each_id = int(each_id.split('\n')[0])
        the_date = int(key2date[each_id].split()[0].replace("-",""))
        if the_date < 20160301:
            matched_true_in_2016_count += 1 
        else:
            matched_true_over_2016_count +=1 
    print("MGH matched true: old(%s) + new(%s) = %s"%(matched_true_in_2016_count,matched_true_over_2016_count ,matched_true_in_2016_count+matched_true_over_2016_count) )
    true_in_2016_count = 0
    true_over_2016_count = 0
    for each_id in all_true_id_list:
        # each_id = int(each_id.split('\n')[0])
        the_date = int(key2date[each_id].split()[0].replace("-",""))
        if the_date < 20160301:
            true_in_2016_count += 1 
        else:
            true_over_2016_count += 1
    print("MGH all true: old(%s) + new(%s) = %s"% (true_in_2016_count,true_over_2016_count,true_in_2016_count+true_over_2016_count ))
    return all_true_id_list, keyword_true_id_list
 

def MGH_file_all_unmatched(original_file, keyword_file):
    key2date = read_key2date("../../Data/MGH_key2date.csv")
    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_excel(original_file)
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    
    print("Original data size:", ori_data.shape[0])
    the_id_list = []
    the_date_list = []
    for a in range(ori_data.shape[0]):
        sent = str(ori_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        the_id = str(ori_data.iloc[[a]]['ID']).split("    ", 1)[1]
        the_id = int(the_id.split('\n')[0])
        the_date = int(key2date[the_id].split()[0].replace("-",""))
        if not match:
            the_id_list.append(the_id)
            the_date_list.append(the_date)
    print("All unmatched num:", len(the_id_list))
    the_data = pd.DataFrame(list(zip(the_id_list, the_date_list)), columns = ['ID', "Des"])
    the_data.to_excel("MGH_all_matched_id.xlsx", index=False)
    return 


def BWH_file_load_match(original_file, keyword_file):
    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_excel(original_file)
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    
    new_data = ori_data[:5800]
    selected_data = new_data.loc[new_data["Gold"]==1]
    all_true = selected_data.shape[0]
    print("Original data size:", ori_data.shape[0])
    print("True data size:", all_true)
    last_selected_id = selected_data.iloc[-1]["ID"]
    keyword_true_id_list = []
    all_true_id_list = []
    all_matched_id_list = []

    ## match keywords in true cases
    all_true_matched_count = 0
    for a in range(selected_data.shape[0]):
        sent = str(selected_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        if match:
            all_true_matched_count += 1
    ## match keywords in all cases
    all_matched_count = 0
    for a in range(ori_data.shape[0]):
        sent = str(ori_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        if match:
            all_matched_count += 1
    print("BWH all true:", all_true)
    print("BWH all manual:", 5800)
    print("BWH all matched:", all_matched_count)
    print("BWH true matched:", all_true_matched_count)
    return 0, 1


def select_BWH_keyword_not_minus_true(BWH_reviewed_file, keyword_file, output_file):
    ## select cased not in model prediction but has keyword
    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_excel(BWH_reviewed_file)
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    
    selected_data = ori_data.loc[ori_data["Gold"]!=1]
    non_true = selected_data.shape[0]
    print("Original data size:", ori_data.shape[0])
    print("True data size:", non_true)
    ## match keywords in true cases
    matched_tuple_list = []
    for a in range(selected_data.shape[0]):
        sent = str(selected_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        if match:
            ID = str(selected_data.iloc[a]['ID'])
            reviewed = selected_data.iloc[a]['Gold']
            matched_tuple_list.append((ID,reviewed,sent))
    random.shuffle(matched_tuple_list)
    print("Matched_num:", len(matched_tuple_list))
    out_df = pd.DataFrame(matched_tuple_list[:1000], columns=["ID", "Gold", "Des"])
    out_df.to_excel(output_file, index=False)
    return 0, 1


def select_MGH_keyword_not_minus_true(MGH_reviewed_file, keyword_file, output_file):
    key2date = read_key2date("../../Data/MGH_key2date.csv")
    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_excel(MGH_reviewed_file)
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    
    selected_data = ori_data.loc[ori_data["Gold"]!=1]
    non_true = selected_data.shape[0]
    print("Original data size:", ori_data.shape[0])
    print("True data size:", non_true)
    ## match keywords in true cases
    matched_tuple_list = []
    for a in range(selected_data.shape[0]):
        sent = str(selected_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        if match:
            the_id = str(selected_data.iloc[[a]]['ID']).split("    ", 1)[1]
            the_id = int(the_id.split('\n')[0])
            the_date = int(key2date[the_id].split()[0].replace("-",""))
            if the_date >=20160301:
                ID = str(selected_data.iloc[a]['ID'])
                reviewed = selected_data.iloc[a]['Gold']
                matched_tuple_list.append((ID,reviewed,sent, str(the_date)))
    random.shuffle(matched_tuple_list)
    print("Matched_num:", len(matched_tuple_list))
    out_df = pd.DataFrame(matched_tuple_list[:1000], columns=["ID", "Gold", "Des", "Date"])
    out_df.to_excel(output_file, index=False)
    return 0, 1







def merge_MGH_together(MGH_decode_all, MGH_manual_review_all, DatasetI_file, old_matched_ID_file, keyword_file):
    key2date = read_key2date("../../Data/MGH_key2date.csv")
    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_csv(MGH_decode_all,encoding = "ISO-8859-1")
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)

    ori_data.insert(1, "Date", "")
    ori_data.insert(1, "Keyword_Match", "")
    ori_data.insert(1, "Dataset", "")
    ori_data.insert(1, "Over201603", "")
    ori_data.insert(1, "old_match", "") ## matched based on file MGH_all_matched_id.xlsx

    ## review to dict: key2gold
    review_data = pd.read_excel(MGH_manual_review_all)
    if "Des" not in review_data.columns:
        review_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in review_data.columns:
        review_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    key2gold = dict(zip(review_data.ID, review_data.Gold))


    ## old matched_ID dict
    old_matched = pd.read_excel(old_matched_ID_file)
    if "ID" not in old_matched.columns:
        old_matched.rename(columns={'KeyValue':'ID'}, inplace=True)
    old_set =set(old_matched['ID'].unique())
    print("Old set:", len(old_set))

    ## Dataset I to dict: key2datesetI
    datasetI_data = pd.read_csv(DatasetI_file,encoding = "ISO-8859-1")
    print("Dataset I columns:",datasetI_data.columns,datasetI_data.shape[0] )
    if "Des" not in datasetI_data.columns:
        datasetI_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in datasetI_data.columns:
        datasetI_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    key2datasetI = dict(zip(datasetI_data.ID, datasetI_data.HSR))

    dataset_I_size = 0
    dataset_1p5_size = 0
    dataset_II_size = 0
    dataset_III_size = 0

    dataset_II_keyword_match_num = 0
    dataset_III_keyword_match_num = 0
    dataset_II_deeplearning_match_num = 0
    dataset_III_deeplearning_match_num = 0

    dataset_II_keyword_match_true_num = 0 
    dataset_III_keyword_match_true_num = 0
    dataset_II_deeplearning_match_true_num = 0 
    dataset_III_deeplearning_match_true_num = 0
    from tqdm import tqdm
    for a in tqdm(range(ori_data.shape[0])):
        sent = str(ori_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        the_id = ori_data.iloc[a]['ID']
        # the_id = str(ori_data.iloc[[a]]['ID']).split("    ", 1)[1]
        
        # the_id = int(the_id.split('\n')[0])
        # if old_id != the_id:
        #     print(old_id, the_id)
        #     exit(0)
        the_date = int(key2date[the_id].split()[0].replace("-",""))
        
        
        ori_data.at[a,"Date"] = the_date
        ori_data.at[a,"Keyword_Match"] = match

        
        old_matched_flag = ""
        if ori_data.iloc[a]['ID'] in old_set:
            old_matched_flag = True 
        else:
            old_matched_flag = False 
        ori_data.at[a,"old_match"] = old_matched_flag
        new_date = ""
        if the_date <20160301:
            new_date = False 
        else:
            new_date = True 
        ori_data.at[a,"Over201603"] = new_date

        gold_value = ""
        if ori_data.iloc[a]['ID'] in key2gold:
            gold_value = key2gold[ori_data.iloc[a]['ID']]
        dataset = ""
        if ori_data.iloc[a]['ID'] in key2datasetI:
            dataset = "Dataset I"
            gold_value = key2datasetI[ori_data.iloc[a]['ID']]
            dataset_I_size += 1
        elif the_date < 20160301 and match:
            dataset = "Dataset 1.5"
            dataset_1p5_size += 1
        elif the_date < 20160301 and not match:
            dataset = "Dataset II"
            dataset_II_size += 1
            if gold_value != "": ## means case is manually reviewed
                dataset_II_deeplearning_match_num += 1
                if str(gold_value) == '1':
                    dataset_II_deeplearning_match_true_num += 1
        elif the_date >= 20160301:
            dataset = "Dataset III"
            dataset_III_size += 1
            if match:
                dataset_III_keyword_match_num += 1 
                if str(gold_value) == '1':
                    dataset_III_keyword_match_true_num += 1
            if gold_value != "": ## means case is manually reviewed
                dataset_III_deeplearning_match_num += 1
                if str(gold_value) == '1':
                    dataset_III_deeplearning_match_true_num += 1
        ori_data.at[a,"Dataset"] = dataset
        ori_data.at[a,"Gold"] = gold_value
    print("Dataset 1.5 size:", dataset_1p5_size)
    print("Dataset II size:", dataset_II_size)
    print("Dataset II keyword match num:", 0)
    print("Dataset II keyword match true num:", 0)
    print("Dataset II deep learning num:", dataset_II_deeplearning_match_num)
    print("Dataset II deep learning true num:", dataset_II_deeplearning_match_true_num)

    print("Dataset III size:", dataset_III_size)
    print("Dataset III keyword match num:", dataset_III_keyword_match_num)
    print("Dataset III keyword match true num:", dataset_III_keyword_match_true_num)
    print("Dataset III deep learning num:", dataset_III_deeplearning_match_num)
    print("Dataset III deep learning true num:", dataset_III_deeplearning_match_true_num)
    # exit(0)
    ori_data.to_excel('../../Data/MGH_merge_all.xlsx', encoding ='utf-8')



def merge_BWH_together(BWH_decode_all, BWH_manual_review_all, keyword_file):
    RL = pd.read_csv("../../Data/BWH_RL_id2date.csv")
    RL['key'] = 'RL-' + RL['File_ID'].astype(str)
    key2date = dict(zip(RL['key'], RL['SUBMITTEDDATE']))
    legacy = pd.read_csv("../../Data/BWH_legacy_id2date.csv")
    legacy['key'] = 'Legacy-' + legacy['FILE_ID'].astype(str)
    key2date2 = dict(zip(legacy['key'], legacy['EVENTDATE']))
    key2date.update(key2date2)
    

    pre_list, suf_list, full_list = load_origin_keyword_list(keyword_file)
    ori_data = pd.read_csv(BWH_decode_all,encoding = "ISO-8859-1")
    print(ori_data.columns)
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in ori_data.columns:
        ori_data.rename(columns={'KeyValue':'ID'}, inplace=True)


    ori_data.insert(1, "Keyword_Match", "")
    ori_data.insert(1, "Dataset", "")
    ori_data.insert(1, "Date", "")

    ## review to dict: key2gold
    review_data = pd.read_excel(BWH_manual_review_all)
    if "Des" not in review_data.columns:
        review_data.rename(columns={'Description':'Des'}, inplace=True)
    if "ID" not in review_data.columns:
        review_data.rename(columns={'KeyValue':'ID'}, inplace=True)
    key2gold = dict(zip(review_data.ID, review_data.Gold))
    
    dataset_IV_size = ori_data.shape[0]
    dataset_IV_keyword_match_num = 0
    dataset_IV_keyword_match_true_num = 0 
    dataset_IV_deeplearning_match_num = 0
    dataset_IV_deeplearning_match_true_num = 0 


    for a in tqdm(range(ori_data.shape[0])):
        sent = str(ori_data.iloc[a]['Des'])
        match = keyword_match(sent, pre_list, suf_list, full_list)
        the_id = ori_data.iloc[a]['ID']
        date = ""
        if the_id in key2date:
            date = key2date[the_id]
        ori_data.at[a,"Date"] = date
        ori_data.at[a,"Keyword_Match"] = match
        gold_value = ""
        if ori_data.iloc[a]['ID'] in key2gold:
            gold_value = key2gold[ori_data.iloc[a]['ID']]
        dataset = "Dataset IV"
        ori_data.at[a,"Dataset"] = dataset
        ori_data.at[a,"Gold"] = gold_value
        if match:
            dataset_IV_keyword_match_num += 1 
            if str(gold_value) == '1':
                dataset_IV_keyword_match_true_num += 1
        if gold_value != "": ## means case is manually reviewed
            dataset_IV_deeplearning_match_num += 1
            if str(gold_value) == '1':
                dataset_IV_deeplearning_match_true_num += 1

    print("Dataset IV size:", dataset_IV_size)
    print("Dataset IV keyword match num:", dataset_IV_keyword_match_num)
    print("Dataset IV keyword match true num:", dataset_IV_keyword_match_true_num)
    print("Dataset IV deep learning num:", dataset_IV_deeplearning_match_num)
    print("Dataset IV deep learning true num:", dataset_IV_deeplearning_match_true_num)
    # exit(0)
    ori_data.to_excel('../../Data/BWH_merge_all.xlsx', encoding ='utf-8')


def failure_analysis_sample_100_for_keyword_and_deep_learning(MGH_merge_all):
    ori_data = pd.read_excel(MGH_merge_all)
    ori_data = ori_data[ori_data['Dataset']=="Dataset III"]
    ori_data.fillna({'Gold':'0'})
    ori_data['Gold'] = ori_data['Gold'].map({0:"0", 1:"1", "":'0'})
    
    keyword_failure1 = ori_data[ori_data['Keyword_Match'] == True][ori_data['Gold'] == '0'] 
    keyword_failure2 = ori_data[ori_data['Keyword_Match'] == False][ori_data['Gold'] == '1'] 
    keyword = pd.concat([keyword_failure1, keyword_failure2])
    print(keyword_failure1.shape, keyword_failure2.shape)
    deep_learning1 = ori_data[ori_data['Pred'] > 0.8][ori_data['Gold'] == '0'] 
    deep_learning2 = ori_data[ori_data['Pred'] < 0.05][ori_data['Gold'] == '1'] 
    print(deep_learning1.shape, deep_learning2.shape)
    deep = pd.concat([deep_learning1, deep_learning2])
    import random
    random.seed(42)
    keyword = keyword.sample(frac=1)
    deep = deep.sample(frac= 1)
    keyword.head(200).to_excel("../../Data/keyword_failure_random200.xlsx", index=False)
    deep.head(200).to_excel("../../Data/deep_failure_random200.xlsx", index =False)




def word_count_frequency(data_file, word_list):
    word_list = [word.lower() for word in word_list]
    ori_data = pd.read_csv(data_file,encoding = "ISO-8859-1")
    if "Des" not in ori_data.columns:
        ori_data.rename(columns={'Description':'Des'}, inplace=True)
    case_word_dict_list = []
    true_count_list = [0]*len(word_list)
    word_count_list = [0]*len(word_list)
    for a in tqdm(range(ori_data.shape[0])):
        sent = clinic_text_processing( str(ori_data.iloc[a]['Des']))
        word_dict = set(sent_token(sent, True))
        for idx, word in enumerate(word_list):
            if word in word_dict:
                # print(sent)
                word_count_list[idx] += 1 
                # print(ori_data.iloc[a]['HSR'])
                # if str(ori_data.iloc[a]['HSR']) == '1.0':
                #     true_count_list[idx] += 1

    case_num = ori_data.shape[0]
    for idx, word in enumerate(word_list):
        print("Word: %s, count: %s, true count: %s,  freq: %s"%(word, word_count_list[idx],true_count_list[idx], word_count_list[idx]/case_num))
    




# def plot_bar():
#     BWH: AI 1569/5800=27.1%, keyword(lowercase): 1344/15896=8.5%, keyword(normal): 1206/11572=10.4%
#     MGH new: AI 622/1971=31.6%, keyword(lowercase): 570/10131=5.6%  keyword(normal): 519/6047=8.6%
#     ytick = [(x+0.)/10 for x in range(0, 11,2)]
#     xtick = [idx*100 for idx in range(0,11,2)]
#     plt.xticks(xtick, fontsize=font_size)
#     plt.yticks(ytick, fontsize=font_size)
#     plt.ylabel(y_name ,fontname = "Arial", fontsize=font_size)
#     plt.xlabel(x_name,fontname = "Arial",fontsize=font_size)
#     plt.tight_layout()
#     ax = plt.gca()
#     ax.xaxis.set_minor_locator(ticker.IndexLocator(base=100, offset=0))
#     # plt.grid()
#     legends = []
#     for idx in range(model_num):
#         legends.append(mlines.Line2D([], [], color=color_list[idx],  markersize=5, linestyle=linestyle_list[idx], label=name_list[idx], lw=3))##marker=marker[idx],
#     plt.legend(handles=legends, loc='best',prop={'size':font_size})
#     if save_dir:
#         plt.savefig(save_dir)
#     plt.show()






if __name__ == '__main__':
    # word_count_frequency("../../Data/HSR.BWH.att.trainall.decode.sparateID.csv",['anaphylaxis', 'anaphylactic','angioedema','swelling','edema']) ##9K_reports_20181228.xlsx
    # exit(0)
    # failure_analysis_sample_100_for_keyword_and_deep_learning('../../Data/MGH_merge_all.xlsx')
    # exit(0)
    # merge_MGH_together("../../Data/HSR.MGH.att.trainall.decode.csv","../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx", "../../Data/DatasetI.csv","../../Data/MGH_all_matched_id.xlsx","auto_keyword_jps.txt")
    merge_BWH_together("../../Data/HSR.BWH.att.trainall.decode.sparateID.csv", "../../Data/HSR.BWH.att.all.annotated_top5800.sparateID.xlsx",  "auto_keyword_jps.txt")

    # plot_bars()

    # pre_list, suf_list, full_list = load_origin_keyword_list("auto_keyword_jps.txt")
    # pre_list, suf_list, full_list = load_keyword_list("../keywords.txt")
    # sentence = "I got heavy allergy"
    # print(keyword_match(sentence,pre_list, suf_list, full_list))
    # "HSR.BWH.att.all.annotated_top5800" "HSR.MGH.tobe_reviewed_20191015"
    ## Dataset III efficiency 
    # all_true_id_list, keyword_true_id_list = MGH_file_load_match("../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx", "auto_keyword_jps.txt")

    ## Dataset IV efficiency
    # bwh_data = "../../Data/HSR.BWH.att.all.annotated_top5800.sparateID.xlsx" #"../../Data/HSR.BWH.att.trainall.decode.sparateID.xlsx"
    # all_true_id_list, keyword_true_id_list = BWH_file_load_match(bwh_data, "auto_keyword_jps.txt")

    # select_BWH_keyword_not_minus_true("HSR.BWH.all_with_top5800_annotated.xlsx","auto_keyword_jps.txt", "BWH_random_select1000_keyword_match_minus_true.xlsx")
    # select_MGH_keyword_not_minus_true("../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx","auto_keyword_jps.txt", "../../Data/MGH_new_data_random_select1000_keyword_match_minus_true.xlsx")
    # MGH_file_all_unmatched("../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx","auto_keyword_jps.txt")

    
