# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-28 11:40:37
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2020-02-13 00:29:58
# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import pandas as pd
from keyword_match import *


def extract_results():
    key2date = pd.read_csv("../../Data/MGH_key2date.csv")
    keys = key2date["KeyValue"].tolist()
    dates = key2date["EVENTDATE"].tolist()
    date_dict = dict(zip(keys, dates))
    MGH_all_data = pd.read_excel("../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx")
    new_mgh_result = []
    for idx in range(MGH_all_data.shape[0]):
        the_date = date_dict[MGH_all_data.iloc[idx]["KeyValue"]].split()[0].replace("-","")
        if int(the_date) >= 20160301:
            new_mgh_result.append(MGH_all_data.iloc[idx]["Gold"])
    # print(new_mgh_result, len(new_mgh_result))
    BWH_all_data = pd.read_excel("../../Data/HSR.BWH.att.all.annotated_top5800.sparateID.xlsx")
    BWH_result = BWH_all_data['Gold'].tolist()
    # print(BWH_result, len(BWH_result))

    return new_mgh_result[:1026], BWH_result 


def plot_multi_in_one(x_lists, y_lists, name_list, x_name="X_Name", y_name="Y_name",title="Title",  save_dir=None):
    r''' plot precision-recall for multiple models, based on different cut-off probabilities
        Args:
            x_lists (list of numpy array): list of x label array
            y_lists (list of numpy array): list of y label array
            name_list (list of string): name of the model
            x_name (string): name of x axis
            y_name (string): name of y axis
            save_dir (string): file directoy to be saved
    '''
    font_size = 14
    plt.figure(figsize=(5,5))
    plt.title(title,fontname = "Arial", fontsize= font_size)
    model_num = len(x_lists)
    color_list = ['black','#ff7f0e', '#1f77b4', 'blue', 'tab:orange',  'tab:green', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']
    marker = ['o', "*", '^']
    linestyle_list = ['dotted','-','--']

    new_length_node = []
    for idx in range(model_num):
        new_length_node.append(np.linspace(x_lists[idx][0], x_lists[idx][-1], 100))

    fit_function_poly = 1
    for idx in range(model_num):
        # plt.plot(x_list[idx], y_list[idx], color_list[idx], label = '%s' % (model_name_list[idx]))
        interplt1 = interp1d(x_lists[idx], y_lists[idx], kind=fit_function_poly)
        # plt.plot(x_lists[idx], y_lists[idx],  new_length_node[idx], interplt1(new_length_node[idx]),'-',color=color_list[idx], lw = 2.5)#, markersize=5)
        plt.plot(x_lists[idx], y_lists[idx], linestyle=linestyle_list[idx],color=color_list[idx], lw = 3)#, markersize=5)
    
    plt.ylim([0, 1.005])
    plt.xlim([0, 1005])
        
    ytick = [(x+0.)/10 for x in range(0, 11,2)]
    xtick = [idx*100 for idx in range(0,11,2)]
    plt.xticks(xtick, fontsize=font_size)
    plt.yticks(ytick, fontsize=font_size)
    plt.ylabel(y_name ,fontname = "Arial", fontsize=font_size)
    plt.xlabel(x_name,fontname = "Arial",fontsize=font_size)
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.IndexLocator(base=100, offset=0))
    # plt.grid()
    legends = []
    for idx in range(model_num):
        legends.append(mlines.Line2D([], [], color=color_list[idx],  markersize=5, linestyle=linestyle_list[idx], label=name_list[idx], lw=3))##marker=marker[idx],
    plt.legend(handles=legends, loc='best',prop={'size':font_size})
    if save_dir:
        plt.savefig(save_dir)
    plt.show()





def turn_sequence_binary_to_topk_value(binary_list,interval=1):
    print(binary_list)
    list_length = len(binary_list)
    x_list = []
    y_list = []
    for idx in range(100, list_length, interval):
        rate = (sum(binary_list[:idx])+0.)/idx
        x_list.append(idx)
        y_list.append(rate)
    print(x_list)
    print(y_list)
    return x_list, y_list


    
def extract_nonkeyword_reports(save_dir=None):
    pre_list, suf_list, full_list = load_origin_keyword_list("auto_keyword_jps.txt")
    key2date = read_key2date("../../Data/MGH_key2date.csv")
    MGH_all_data = pd.read_excel("../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx")
    nokeyword_mgh_result = []
    nokeyword_mgh_key_list = []
    for idx in range(8750):
        the_key = MGH_all_data.iloc[idx]["KeyValue"]
        the_date = int(key2date[the_key].split()[0].replace("-",""))
        if the_date < 20160301:
            sent = str(MGH_all_data.iloc[idx]["Description"])
            match_result = keyword_match(sent, pre_list, suf_list, full_list)
            if  not match_result:
                result = MGH_all_data.iloc[idx]["Gold"] 
                if str(result) == "nan":
                    result = 0.0
                nokeyword_mgh_result.append(result)
                nokeyword_mgh_key_list.append(the_key)
    print(nokeyword_mgh_result, len(nokeyword_mgh_result))
    print(sum(nokeyword_mgh_result), len(nokeyword_mgh_result))
    return nokeyword_mgh_result
    x_list = []
    y_list = []
    for idx in range(100, 1050, 50):
        x_list.append(idx)
        y_list.append((sum(nokeyword_mgh_result[:idx])+0.)/idx)
    print(x_list)
    print(y_list)
    font_size = 14
    plt.figure(figsize=(5,5))
    plt.title("Precision at top k",fontname = "Arial", fontsize= font_size)
    plt.bar(x_list, y_list, color="#2ca02c", width=25)
    ytick = [(x+0.)/10 for x in range(0, 11,2)]
    xtick = [idx*100 for idx in range(1,11)]
    plt.xticks(xtick, fontsize=font_size)
    plt.yticks(ytick, fontsize=font_size)
    plt.ylabel("Precision" ,fontname = "Arial", fontsize=font_size)
    plt.xlabel("Top-k",fontname = "Arial",fontsize=font_size)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir)
    plt.show()


def nokeyword_annotated_result():
    pre_list, suf_list, full_list = load_origin_keyword_list("auto_keyword_jps.txt")
    key2date = read_key2date("../../Data/MGH_key2date.csv")
    MGH_all_data = pd.read_excel("../../Data/HSR.MGH.tobe_reviewed_top8750.xlsx")
    annotated_true = 0
    annotated_false = 0
    for idx in range(8750):
        the_key = MGH_all_data.iloc[idx]["KeyValue"]
        the_date = int(key2date[the_key].split()[0].replace("-",""))
        if the_date < 20160301:
            sent = str(MGH_all_data.iloc[idx]["Description"])
            match_result = keyword_match(sent, pre_list, suf_list, full_list)
            if  not match_result:
                result = MGH_all_data.iloc[idx]["Gold"] 
                if str(result) == "nan":
                    result = 0.0
                if result > 0:
                    annotated_true += 1 
                else:
                    annotated_false += 1 
    print("nokeyword true: %s; False: %s; total: %s"%(annotated_true, annotated_false, annotated_false+annotated_true))
    return 0






if __name__ == '__main__':
    # nokeyword_annotated_result()
    # exit(0)
    steps = 20
    nonkeyword_bar = "top_k_bar.pdf"
    nokeyword_mgh_result = extract_nonkeyword_reports(nonkeyword_bar)
    print(nokeyword_mgh_result)
    # exit(0)
    new_mgh, bwh = extract_results()
    mgh_x, mgh_y = turn_sequence_binary_to_topk_value(new_mgh, steps)
    bwh_x, bwh_y = turn_sequence_binary_to_topk_value(bwh, steps)
    nokey_mgh_x, nokey_mgh_y = turn_sequence_binary_to_topk_value(nokeyword_mgh_result, steps)
    for a, b in zip(bwh_x, bwh_y):
        print("BWH: ", a, b)
    for a, b in zip(mgh_x, mgh_y):
        print("MGH: ",a, b)
    for a, b in zip(nokey_mgh_x, nokey_mgh_y):
        print("nokey MGH: ",a, b)
    name_list = ["Dataset II (MGH)", "Dataset III (MGH)", "Dataset IV (BWH)"]
    plot_multi_in_one([nokey_mgh_x, mgh_x, bwh_x], [nokey_mgh_y, mgh_y, bwh_y], name_list, x_name="k", y_name="Precision",title="Precision at Top k", save_dir= "../../Data/top_k_curvex3.pdf")
