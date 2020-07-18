# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-06-10 10:00:09
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-11-25 22:24:42


import json
import os
import csv
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

font_size = 14

def load_csv(csv_file, column_list):
    output_list = [[] for x in range(len(column_list))]
    with open(csv_file,'r') as fin:
        csv_reader = csv.reader(fin, delimiter=',')
        line_count = 0
        for line in csv_reader:
            if line_count == 0:
                line_count += 1 
                continue
            if len(line) > 3:
                print("longer than 3:",len(line))
                exit(0)
            for id in range(len(column_list)):
                focus_id = column_list[id]
                if focus_id == 1:
                    line[focus_id] = datetime.strptime(line[focus_id].split()[0], '%Y-%m-%d').date()
                output_list[id].append(line[focus_id])
    return output_list


def extract_length_distribution(word_list):
    length_list = [len(a.split()) for a in word_list]
    distance = 20
    length_interval = [[a*distance, a*distance+distance, 0] for a in range(21)]
    length_interval[-1][1] = 10000000
    for each_length in length_list:
        found = False 
        for idx in range(len(length_interval)):
            if found: 
                continue 
            if (length_interval[idx][0] <= each_length) and (each_length < length_interval[idx][1]):
                length_interval[idx][2] += 1
                found = True
    return length_interval




def extract_date_distribution(date_list):
    time_interval = []
    start_date = date(2004, 1, 1)
    for idx in range(30):
        end_date = start_date + relativedelta(years=+1)
        if end_date < date(2019, 1, 2):
            time_interval.append([start_date, end_date, 0])
            start_date = end_date
    for each_date in date_list:
        found = False 
        for idx in range(len(time_interval)):
            if found: 
                continue 
            if (time_interval[idx][0] <= each_date) and (each_date < time_interval[idx][1]):
                time_interval[idx][2] += 1
                found = True
    return time_interval


def draw_date_distribution(BWH_file, MGH_file, savefig=None):
    MGH_date, MGH_text = load_csv(MGH_file, [1,2])
    BWH_date, BWH_text = load_csv(BWH_file, [1,2])
    MGH_date_list = extract_date_distribution(MGH_date)
    BWH_date_list = extract_date_distribution(BWH_date)
    # print(BWH_date_list)
    draw_date_figure([BWH_date_list, MGH_date_list], ['MGH',"BWH"], savefig)


def draw_length_distribution(BWH_file, MGH_file, savefig=None):
    MGH_date, MGH_text = load_csv(MGH_file, [1,2])
    BWH_date, BWH_text = load_csv(BWH_file, [1,2])
    MGH_length_list = extract_length_distribution(MGH_text)
    BWH_length_list = extract_length_distribution(BWH_text)
    # print(BWH_date_list)
    draw_length_figure([BWH_length_list, MGH_length_list], ['MGH',"BWH"], savefig)


def draw_length_figure(input_lists, input_names, save_fig=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    curve_num = len(input_names)
    x = range(len(input_lists[0]))
    x_labels = list(range(0, 401, 40))
    x_labels[-1] = ">400"
    color_list = ['cornflowerblue','darkorange','g','y']
    value_lists = []
    for idx in range(curve_num):
        the_value = []
        for each_set in input_lists[idx]:
            print(each_set)
            the_value.append(each_set[2])
        value_lists.append(the_value)
    plt.figure(figsize=(5,3))
    ax = plt.gca()
    for idx in range(curve_num):
        ax.bar((np.asarray(x)+idx*0.4).tolist(), value_lists[idx], width=0.4, color= color_list[idx], label=input_names[idx], align='edge')
    plt.xticks(list(range(0,22,2)),x_labels, rotation=40, fontsize=font_size) 
    y_labels = ["0", "10k", "20k", "30k", "40k"]
    plt.yticks(list(range(0,40001,10000)) ,y_labels, fontsize=font_size)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', length=1)
    ax.xaxis.set_minor_locator(ticker.IndexLocator(base=1, offset=1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='best', fontsize=font_size)
    leg = plt.legend(fontsize=font_size)
    leg.get_frame().set_linewidth(0.0)
    plt.ylabel('No. of Reports',fontname = "Arial", fontsize=font_size)
    plt.xlabel('Length of Reports',fontname = "Arial", fontsize=font_size)
    plt.tight_layout()
    # plt.show()
    if save_fig:
        plt.savefig(save_fig, bbox_inches = 'tight',pad_inches = 0)
    else:
        plt.show()




def draw_date_figure(input_lists, input_names, save_fig=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    curve_num = len(input_names)
    x = range(len(input_lists[0]))
    x_labels = range(2004, 2020)
    color_list = ['cornflowerblue','darkorange','g','y'] ##teal
    value_lists = []
    for idx in range(curve_num):
        the_value = []
        for each_set in input_lists[idx]:
            print(each_set)
            the_value.append(each_set[2])
        value_lists.append(the_value)
    plt.figure(figsize=(5,3))
    ax = plt.gca()
    for idx in range(curve_num):
        ax.bar((np.asarray(x)+idx*0.4).tolist(), value_lists[idx], width=0.4, color= color_list[idx], label=input_names[idx], align='edge')
    plt.xticks(list(range(0,15)),x_labels, rotation=55, fontsize=font_size-2) 
    y_labels = ["0", "1k", "10k", "15k", "20k"]
    plt.yticks(list(range(0,20001,5000)) ,y_labels, fontsize=font_size)
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.tick_params(which='minor', length=1)
    # ax.xaxis.set_minor_locator(ticker.IndexLocator(base=1, offset=1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc='best')
    leg = plt.legend(fontsize=font_size)
    leg.get_frame().set_linewidth(0.0)
    plt.ylabel('No. of Reports',fontname = "Arial", fontsize=font_size)
    plt.xlabel('Years',fontname = "Arial", fontsize=font_size)
    plt.tight_layout()
    # plt.show()
    if save_fig:
        plt.savefig(save_fig, bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.show()


if __name__ == '__main__':
    MGH = "MGH_all_date.csv"
    BWH = "BWH_all_date.csv"
    draw_length_distribution(MGH,BWH, "length_distribution.pdf")
    draw_date_distribution(MGH, BWH, "date_distribution.pdf")
    exit(0)
    date_list, text_list = load_csv("BWH_all_date.csv",[1,2])
    # extract_date_distribution(date_list)
    extract_wordnum_distribution(text_list)
