# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-24 13:03:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-10-16 14:38:39
# -*- coding: utf-8 -*-
import pandas as pd


def split_id_to_two_columns(input_data, output_data):
    df = pd.read_excel(input_data)
    concated_id = df['ID'].tolist()
    system_list = []
    keyvalue = []
    for a in concated_id:
        s, k = a.split('-')
        system_list.append(s)
        keyvalue.append(k)
    df.insert(0, "KeyValue",keyvalue)
    df.insert(0, "System",system_list)
    df.to_excel(output_data, index=False)

if __name__ == '__main__':
    split_id_to_two_columns("HSR.BWH.att.all.annotated_top5800.xlsx", "HSR.BWH.att.all.annotated_top5800.sparateID.xlsx")

    
    

    
