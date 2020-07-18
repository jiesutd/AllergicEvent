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
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

font_size = 14 


def list_all_date_between_two_dates(start_date, end_date):
    delta = end_date - start_date       # as timedelta
    date_dict = {}
    for i in range(delta.days):
        day = start_date + timedelta(days=i)
        day = day.strftime("%Y-%m-%d")
        date_dict[day] = start_date.strftime("%Y-%m-%d")
    return date_dict




all_day_dict = {}
start_date = date(2004,7,1)
end_date = start_date
date_string_list = []
while end_date < date(2019,1,1):
    end_date = start_date + relativedelta(months = +6)
    sub_dict = list_all_date_between_two_dates(start_date, end_date)
    date_string_list.append(start_date.strftime("%Y-%m-%d"))
    all_day_dict.update(sub_dict)
    start_date = end_date
new_dict = {}
for k, v  in all_day_dict.items():
    new_dict[k.replace('-',"")] = v
all_day_dict.update(new_dict)
# print(all_day_dict)
# print(len(all_day_dict))
print(date_string_list)




def load_BWH(input_file):
    print("load file: ", input_file)
    df = pd.read_excel(input_file)
    df['Date'] = df['Date'].str.split(' ').str[0]
    df['Period'] = df['Date'].map(all_day_dict)
    period_count = []
    period_rate = []
    for date_string in date_string_list:
        temp_df = df[df['Period']==date_string]
        temp_all_count = temp_df.shape[0]
        temp_true_count = temp_df[temp_df['Gold'] == 1].shape[0]
        period_count.append(temp_true_count)
        period_rate.append(temp_true_count/temp_all_count)
    return period_count, period_rate 


def load_MGH(input_file):
    print("load file: ", input_file)
    df = pd.read_excel(input_file)
    df['Date'] = df['Date'].apply(str)
    df['Period'] = df['Date'].map(all_day_dict)
    period_count = []
    period_rate = []
    for date_string in date_string_list:
        temp_df = df[df['Period']==date_string]
        temp_all_count = temp_df.shape[0]
        temp_true_count = temp_df[temp_df['Gold'] == 1].shape[0]
        period_count.append(temp_true_count)
        if temp_true_count == 0:
            period_rate.append(0)
        else:
            period_rate.append(temp_true_count/temp_all_count)
    print(period_count)
    print(period_rate)
    return period_count, period_rate 
    



def plot_multi_in_one(x_lists, y_lists, name_list, x_name="X_Name", y_name="Y_name",title="Title", percentage=False, save_dir=None):
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


    for idx in range(model_num):
        new_x = []
        new_y = []
        for a, b in zip(x_lists[idx], y_lists[idx]):
            if b != 0:
                new_x.append(a)
                new_y.append(b)
        plt.plot(new_x, new_y, color=color_list[idx], lw = 3, label=name_list[idx])#, markersize=5)
    plt.xticks(list(range(len(date_string_list))), date_string_list, rotation=70, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(y_name ,fontname = "Arial", fontsize=font_size)
    plt.xlabel(x_name,fontname = "Arial",fontsize=font_size)
    if percentage:
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
    plt.tight_layout()
    plt.legend(loc='best',prop={'size':font_size})
    if save_dir:
        plt.savefig(save_dir)
    plt.show()



if __name__ == '__main__':
    # plot_multi_in_one([list(range(25)),[2,3,4]], [list(range(25)), [3,2,1]], ['a','b'])
    # exit(0)
    BWH_file = '../../Data/BWH_merge_all.xlsx'
    MGH_file = '../../Data/MGH_merge_all.xlsx'
    bwh_count, bwh_rate = load_BWH(BWH_file)
    mgh_count, mgh_rate = load_MGH(MGH_file)
    plot_multi_in_one([list(range(len(bwh_count))), list(range(len(mgh_count)))],[bwh_count, mgh_count], ["BWH", "MGH"], "Half year (start date)", "Number", "Allergic events number evolution")
    plot_multi_in_one([list(range(len(bwh_rate))), list(range(len(mgh_rate)))],[bwh_rate, mgh_rate], ["BWH", "MGH"], "Half year (start date)", "Rate", "Allergic events rate evolution")