# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-06-10 10:00:09
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-11-10 01:14:48

import pyodbc 
import json
import os
import numpy as np
import statistics


# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port

def connect_db(server, database, username, password, trust_connect=False):
    if trust_connect:
        cnxn = pyodbc.connect('Driver={SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
    else:
        cnxn = pyodbc.connect('Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    return cursor


def extract_case_dict(cursor, table_name, selected_columns = "*"):
    ''' Extract each case into a dict, return a list of dict
        args:
            cursor: database connection cursor
            table_name: database table name 
        return:
            case_list: list of dict ,each dict is the {column_name1:value1, column2:value2,.. }
    '''
    if isinstance(selected_columns, list):
        sql_string = "SELECT %s from %s;"%(",".join(selected_columns), table_name)
    else:
        sql_string = "SELECT %s from %s;"%(selected_columns, table_name)

    print("Extract Cases SQL: ", sql_string)
    cursor.execute(sql_string) 
    row = cursor.fetchone() 
    columns = [column[0] for column in cursor.description]
    case_list = []
    case_list.append(dict(zip(columns, row)))
    for row in cursor.fetchall():
        case_list.append(dict(zip(columns, row)))
    return case_list


def extract_dict_list(cursor, table_name, selected_columns = "*"):
    ''' Extract a dict, return a dict
        args:
            cursor: database connection cursor
            table_name: database table name 
        return:
            case_list: dict, key is the column name and value is the list of values in order. Like: {column_name1:[value11, value12], column2:[value21, value22],.. }
    '''
    if isinstance(selected_columns, list):
        sql_string = "SELECT %s from %s;"%(",".join(selected_columns), table_name)
    else:
        sql_string = "SELECT %s from %s;"%(selected_columns, table_name)
    case_dict = extract_dict_list_with_sql(cursor, sql_string)
    return case_dict


def extract_dict_list_with_sql(cursor, sql_string):
    ''' Extract a dict, return a dict
        args:
            cursor: database connection cursor
            sql_string: database sql query string
        return:
            case_list: dict, key is the column name and value is the list of values in order. Like: {column_name1:[value11, value12], column2:[value21, value22],.. }
    '''
    print("Extract Cases SQL: ", sql_string)
    cursor.execute(sql_string) 
    row = cursor.fetchone() 
    columns = [column[0] for column in cursor.description]
    column_num = len(columns)
    case_dict = {}
    for idx in range(column_num):
        column = columns[idx]
        case_dict[column] = [row[idx]]
    for row in cursor.fetchall():
        for idx in range(column_num):
            column = columns[idx]
            case_dict[column].append(row[idx])
    return case_dict



def calculate_number_of_person(cohort):
    print("Calculate number of person... Cohort:", cohort)
    cursor = connect_db("PHSSQLBI2", "Research_InpAllergicReactions", None, None, True)
    if cohort == "BWH":
        original_mrn_list, clean_mrn_set = extract_BWH_unique_MRN(cursor)
    elif cohort == "MGH_over_2016":
        original_mrn_list, clean_mrn_set = extract_MGH_over_2016_unique_MRN(cursor)
    elif cohort == "MGH_annotated":
        original_mrn_list, clean_mrn_set = extract_MGH_annotated_unique_MRN(cursor)
    elif cohort == "MGH_all":
        original_mrn_list, clean_mrn_set = extract_MGH_all_unique_MRN(cursor)

    covered_case_number_by_valid_mrn = 0 
    for each_case in original_mrn_list:
        if each_case in clean_mrn_set:
            covered_case_number_by_valid_mrn += 1 
    print("Covered reports number:", covered_case_number_by_valid_mrn)


def calculate_age_word_num_of_person(cohort):
    print("Calculate age of person... Cohort:", cohort)
    cursor = connect_db("PHSSQLBI2", "Research_InpAllergicReactions", None, None, True)
    gender_dict, ethnicity_dict, race_dict, dob_dict = build_demographic_dict(cursor)
    prefix_key = ""
    if cohort == "BWH":
        original_mrn_list, clean_mrn_set = extract_BWH_unique_MRN(cursor)
        prefix_key = "BWH"
    elif cohort == "MGH_over_2016":
        original_mrn_list, clean_mrn_set = extract_MGH_over_2016_unique_MRN(cursor)
        prefix_key = "MGH"
    elif cohort == "MGH_annotated":
        original_mrn_list, clean_mrn_set = extract_MGH_annotated_unique_MRN(cursor)
        prefix_key = "MGH"
    elif cohort == "MGH_all":
        original_mrn_list, clean_mrn_set = extract_MGH_all_unique_MRN(cursor)
        prefix_key = "MGH"
    elif cohort == "MGH_nokeyword":
        original_mrn_list, clean_mrn_set = extract_MGH_nokeyword_unique_MRN(cursor)
        prefix_key = "MGH"

    if cohort == "BWH":
        sql_string = "SELECT PERSON_MRN, EVENTDATE, BRIEFFACTUALDESCRIPTION from [dbo].[BWH_LEGACY_RSCH_TBL_INC_MAIN_FINAL]"
        legacy_cases = extract_dict_list_with_sql(cursor, sql_string)
        sql_string = "SELECT PERSON_MRN, EVENTDATE, BRIEFFACTUALDESCRIPTION from [dbo].[BWH_RL6_RSCH_TBL_INC_MAIN_FINAL]"
        rl_cases = extract_dict_list_with_sql(cursor, sql_string)
        output_cases = {}
        output_cases['PERSON_MRN'] = legacy_cases['PERSON_MRN'] + rl_cases['PERSON_MRN']
        output_cases['EVENTDATE'] = legacy_cases['EVENTDATE'] + rl_cases['EVENTDATE']
        output_cases["BRIEFFACTUALDESCRIPTION"] = legacy_cases['BRIEFFACTUALDESCRIPTION'] + rl_cases['BRIEFFACTUALDESCRIPTION']

    elif cohort == "MGH_over_2016":
        sql_string = "SELECT PERSON_MRN, EVENTDATE,BRIEFFACTUALDESCRIPTION from [dbo].[RSCH_TBL_INC_MAIN_FINAL] where [EVENTDATE] > '2016-02-29'"
        output_cases = extract_dict_list_with_sql(cursor, sql_string)
    elif cohort == "MGH_annotated":
        sql_string = ''' SELECT A.MRN, B.EVENTDATE, B.BRIEFFACTUALDESCRIPTION  from [dbo].[reviewed_incident_reports_with_keyvalue_unique] A 
                        left join [dbo].[RSCH_TBL_INC_MAIN_FINAL] B
                        on  A.KeyValue = B.KeyValue
                        where B.EVENTDATE is not null'''
        output_cases = extract_dict_list_with_sql(cursor, sql_string)
        output_cases["PERSON_MRN"] = output_cases.pop("MRN")
    elif cohort == "MGH_all":
        sql_string = "SELECT PERSON_MRN, EVENTDATE,BRIEFFACTUALDESCRIPTION from [dbo].[RSCH_TBL_INC_MAIN_FINAL]"
        output_cases = extract_dict_list_with_sql(cursor, sql_string)
    elif cohort == "MGH_nokeyword":
        # sql_string = ''' select B.PERSON_MRN, B.EVENTDATE, B.BRIEFFACTUALDESCRIPTION from [Research_InpAllergicReactions].[dbo].[MGH_all_matched_id] A
        #                 inner join [Research_InpAllergicReactions].[dbo].[RSCH_TBL_INC_MAIN_FINAL] B
        #                 on A.ID = B.KeyValue
        #                 where B.EVENTDATE is not null and B.EVENTDATE <= '2016-02-29' '''
        sql_string = '''select B.PERSON_MRN, B.EVENTDATE, B.BRIEFFACTUALDESCRIPTION from [Research_InpAllergicReactions].[dbo].[MGH_merge_all] A
                        left join [Research_InpAllergicReactions].[dbo].[RSCH_TBL_INC_MAIN_FINAL] B
                        on A.ID = B.KeyValue
                        where A.Dataset = 'Dataset II' '''
        output_cases = extract_dict_list_with_sql(cursor, sql_string)

    age_list = []
    description_word_count_list = []
    for mrn, e_date, description in zip(output_cases['PERSON_MRN'], output_cases['EVENTDATE'], output_cases['BRIEFFACTUALDESCRIPTION']):
        if mrn is None:
            continue
        key = prefix_key+":"+mrn
        if mrn in clean_mrn_set and key in dob_dict:
            the_dob = dob_dict[key]
            if the_dob is not None and e_date is not None:
                age = e_date - the_dob
                age_list.append(age.days/365)
        if description:
        	description_word_count_list.append(len(description.split()))
        else:
        	description_word_count_list.append(0)
    
    std = statistics.stdev(age_list)
    mean = statistics.mean(age_list)
    median = statistics.median(age_list)
    q1_x = np.percentile(np.asarray(age_list), 25, interpolation='midpoint')
    q3_x = np.percentile(np.asarray(age_list), 75, interpolation='midpoint')
    print("Age num: %s; mean: %s, std: %s; median:%s, 25pct:%s, 75pct:%s" %(len(age_list), mean, std, median, q1_x, q3_x))
    std = statistics.stdev(description_word_count_list)
    mean = statistics.mean(description_word_count_list)
    median = statistics.median(description_word_count_list)
    q1_x = np.percentile(np.asarray(description_word_count_list), 25, interpolation='midpoint')
    q3_x = np.percentile(np.asarray(description_word_count_list), 75, interpolation='midpoint')
    min_num = min(description_word_count_list)
    max_num = max(description_word_count_list)
    print("Word count num: %s; max:%s, min:%s, mean: %s, std: %s; median:%s, 25pct:%s, 75pct:%s" %(len(description_word_count_list),max_num, min_num, mean, std, median, q1_x, q3_x))
    
    return age_list,description_word_count_list 


'''
  SELECT count(A.MRN) from [dbo].[reviewed_incident_reports] A 
  left join [dbo].[RSCH_TBL_INC_MAIN_FINAL] B
  on CONVERT(NVARCHAR(MAX), A.Description) = B.BRIEFFACTUALDESCRIPTION
  where B.EVENTDATE is not null
'''




def calculate_demographic(cohort):
    cursor = connect_db("PHSSQLBI2", "Research_InpAllergicReactions", None, None, True)
    gender_dict, ethnicity_dict, race_dict, dob_dict = build_demographic_dict(cursor)
    if cohort == "BWH":
        original_mrn_list, clean_mrn_set = extract_BWH_unique_MRN(cursor)
        prefix_key = "BWH"
    elif cohort == "MGH_over_2016":
        original_mrn_list, clean_mrn_set = extract_MGH_over_2016_unique_MRN(cursor)
        prefix_key = "MGH"
    elif cohort == "MGH_annotated":
        original_mrn_list, clean_mrn_set = extract_MGH_annotated_unique_MRN(cursor)
        prefix_key = "MGH"
    elif cohort == "MGH_all":
        original_mrn_list, clean_mrn_set = extract_MGH_all_unique_MRN(cursor)
        prefix_key = "MGH"
    elif cohort == "MGH_nokeyword":
        original_mrn_list, clean_mrn_set = extract_MGH_nokeyword_unique_MRN(cursor)
        prefix_key = "MGH"
    print("Match information based on MRN...")
    ## calculate how many cases are covered by the clean_mrn_set
    covered_cases = 0 
    mrn_report_num_dict = {}
    for a in original_mrn_list:
    	if a in clean_mrn_set:
    		covered_cases += 1
    		if a in mrn_report_num_dict:
    			mrn_report_num_dict[a] += 1 
    		else:
    			mrn_report_num_dict[a] = 1
    num_list = list(mrn_report_num_dict.values())
    max_report_num = max(num_list)
    min_report_num = min(num_list)
    mean = statistics.mean(num_list)
    median = statistics.median(num_list)
    q1_x = np.percentile(np.asarray(num_list), 25, interpolation='midpoint')
    q3_x = np.percentile(np.asarray(num_list), 75, interpolation='midpoint')


    person_num = len(clean_mrn_set)
    gender_list = []
    ethnicity_list = []
    race_list = []
    dob_list = []
    for each_mrn in clean_mrn_set:
        key = prefix_key+":"+each_mrn
        if key in gender_dict:
            gender_list.append(gender_dict[key])
        else:
            gender_list.append("unknown")
        if key in ethnicity_dict:
            ethnicity_list.append(ethnicity_dict[key])
        else:
            ethnicity_list.append("unknown")
        if key in race_dict:
            race_list.append(race_dict[key])
        else:
            race_list.append("unknown")
    print("Cohort: ", cohort, "; unique mrn: ", len(clean_mrn_set), "; covered cases:", covered_cases)
    print("Report num: max:%s; Min:%s; mean:%s; median:%s; 25pct:%s; 75pct:%s"%(max_report_num, min_report_num,mean, median, q1_x, q3_x))
    print("Gender:", count_list(gender_list))
    print("Race:", count_list(race_list))
    print("Ethnicity:", count_list(ethnicity_list))
    


def count_list(input_list):
	output_dict = {}
	for a in input_list:
		if a in output_dict:
			output_dict[a] += 1 
		else:
			output_dict[a] = 1
	return output_dict





def build_demographic_dict(cursor):
    sql_string = "SELECT MRNSiteCD, PERSON_MRN, Gender, ethnicity, Race, BirthDTS from [dbo].[patient_demographics]"
    case_dict = extract_dict_list_with_sql(cursor, sql_string)
    gender_dict = {}
    ethnicity_dict = {}
    race_dict = {}
    dob_dict = {}
    for site, mrn, gender, eth, race, birth in zip(case_dict["MRNSiteCD"], case_dict["PERSON_MRN"], case_dict["Gender"], case_dict["ethnicity"], case_dict["Race"], case_dict["BirthDTS"]):
        if site is None:
            site = "None"
        if mrn is None:
            mrn = "None"
        key = site+":"+mrn
        gender_dict[key] = gender 
        ethnicity_dict[key] = eth 
        race_dict[key] = race
        dob_dict[key] = birth
    return gender_dict, ethnicity_dict, race_dict, dob_dict



def extract_MGH_over_2016_unique_MRN(cursor):
    sql_string = "SELECT PERSON_MRN from [dbo].[RSCH_TBL_INC_MAIN_FINAL] where [EVENTDATE] > '2016-02-29'"
    MGH_cases = extract_dict_list_with_sql(cursor, sql_string)
    mgh_mrn_list = MGH_cases['PERSON_MRN']
    unique_mrn = list(set(mgh_mrn_list))
    clean_mrn_set = set(clean_mrn(unique_mrn))
    print("MGH over 2016 MRN. all: %s; unique: %s; clean: %s"%(len(mgh_mrn_list), len(unique_mrn), len(clean_mrn_set)))
    return mgh_mrn_list, clean_mrn_set


def extract_MGH_all_unique_MRN(cursor):
    sql_string = "SELECT PERSON_MRN from [dbo].[RSCH_TBL_INC_MAIN_FINAL]"
    MGH_cases = extract_dict_list_with_sql(cursor, sql_string)
    mgh_mrn_list = MGH_cases['PERSON_MRN']
    unique_mrn = list(set(mgh_mrn_list))
    clean_mrn_set = set(clean_mrn(unique_mrn))
    print("MGH all MRN. all: %s; unique: %s; clean: %s"%(len(mgh_mrn_list), len(unique_mrn), len(clean_mrn_set)))
    return mgh_mrn_list, clean_mrn_set


def extract_MGH_annotated_unique_MRN(cursor):
    sql_string = "SELECT MRN from [dbo].[reviewed_incident_reports]"
    MGH_cases = extract_dict_list_with_sql(cursor, sql_string)
    mgh_mrn_list = MGH_cases['MRN']
    unique_mrn = list(set(mgh_mrn_list))
    clean_mrn_set = set(clean_mrn(unique_mrn))
    print("MGH annoatted MRN. all: %s; unique: %s; clean: %s"%(len(mgh_mrn_list), len(unique_mrn), len(clean_mrn_set)))
    return mgh_mrn_list, clean_mrn_set


def extract_MGH_nokeyword_unique_MRN_old(cursor):
    sql_string = '''
        select B.PERSON_MRN from [Research_InpAllergicReactions].[dbo].[MGH_all_matched_id] A
        inner join [Research_InpAllergicReactions].[dbo].[RSCH_TBL_INC_MAIN_FINAL] B
        on A.ID = B.KeyValue
        where B.EVENTDATE <= '2016-02-29'
    '''
    MGH_cases = extract_dict_list_with_sql(cursor, sql_string)
    mgh_mrn_list = MGH_cases['PERSON_MRN']
    unique_mrn = list(set(mgh_mrn_list))
    clean_mrn_set = set(clean_mrn(unique_mrn))
    print("MGH nokeyword MRN. all: %s; unique: %s; clean: %s"%(len(mgh_mrn_list), len(unique_mrn), len(clean_mrn_set)))
    return mgh_mrn_list, clean_mrn_set

def extract_MGH_nokeyword_unique_MRN(cursor):
    sql_string = '''
        select B.PERSON_MRN from [Research_InpAllergicReactions].[dbo].[MGH_merge_all] A
        left join [Research_InpAllergicReactions].[dbo].[RSCH_TBL_INC_MAIN_FINAL] B
        on A.ID = B.KeyValue
        where A.Dataset = 'Dataset II'
    '''
    MGH_cases = extract_dict_list_with_sql(cursor, sql_string)
    mgh_mrn_list = MGH_cases['PERSON_MRN']
    unique_mrn = list(set(mgh_mrn_list))
    clean_mrn_set = set(clean_mrn(unique_mrn))
    print("MGH nokeyword MRN. all: %s; unique: %s; clean: %s"%(len(mgh_mrn_list), len(unique_mrn), len(clean_mrn_set)))
    return mgh_mrn_list, clean_mrn_set




def extract_BWH_unique_MRN(cursor):
    BWH_legacy_cases = extract_dict_list(cursor, "[dbo].[BWH_LEGACY_RSCH_TBL_INC_MAIN_FINAL]", 'PERSON_MRN')
    legacy_mrn_list = BWH_legacy_cases['PERSON_MRN']
    BWH_rl_cases = extract_dict_list(cursor, "[dbo].[BWH_RL6_RSCH_TBL_INC_MAIN_FINAL]", 'PERSON_MRN')
    rl_mrn_list = BWH_rl_cases['PERSON_MRN']
    combined_cases_list = legacy_mrn_list + rl_mrn_list
    print("Legacy MRN: %s; RL MRN: %s"%(len(legacy_mrn_list), len(rl_mrn_list)))
    unique_mrn = list(set(combined_cases_list))
    clean_mrn_set = set(clean_mrn(unique_mrn))
    print("Unique MRN for both: %s, after cleaned: %s"%(len(unique_mrn), len(clean_mrn_set)))
    return combined_cases_list, clean_mrn_set


def clean_mrn(mrn_list):
    clean_mrn_list = []
    for mrn in mrn_list:
        if mrn is None:
            continue
        if mrn.isdigit():
            if int(mrn) > 0:
                clean_mrn_list.append(mrn)
    return clean_mrn_list


def calculate_age_word_num_of_both_MGH_BWH():
    BWH_age, BWH_word_num = calculate_age_word_num_of_person("BWH")
    MGH_age, MGH_word_num = calculate_age_word_num_of_person("MGH_all")
    age_list = BWH_age + MGH_age
    import statistics
    std = statistics.stdev(age_list)
    mean = statistics.mean(age_list)
    median = statistics.median(age_list)
    q1_x = np.percentile(np.asarray(age_list), 25, interpolation='midpoint')
    q3_x = np.percentile(np.asarray(age_list), 75, interpolation='midpoint')
    print("Age num: %s; mean: %s, std: %s; median:%s, 25pct:%s, 75pct:%s" %(len(age_list), mean, std, median, q1_x, q3_x))
    description_word_count_list = BWH_word_num + MGH_word_num
    mean = statistics.mean(description_word_count_list)
    median = statistics.median(description_word_count_list)
    q1_x = np.percentile(np.asarray(description_word_count_list), 25, interpolation='midpoint')
    q3_x = np.percentile(np.asarray(description_word_count_list), 75, interpolation='midpoint')
    min_num = min(description_word_count_list)
    max_num = max(description_word_count_list)
    print("Word count num: %s; max:%s, min:%s, mean: %s, std: %s; median:%s, 25pct:%s, 75pct:%s" %(len(description_word_count_list),max_num, min_num, mean, std, median, q1_x, q3_x))
    



def extract_MGH_nokeyword_case_ID():
    cursor = connect_db("PHSSQLBI2", "Research_InpAllergicReactions", None, None, True)
    sql_string = '''
        select B.KeyValue from [Research_InpAllergicReactions].[dbo].[MGH_all_matched_id] A
        inner join [Research_InpAllergicReactions].[dbo].[RSCH_TBL_INC_MAIN_FINAL] B
        on A.ID = B.KeyValue
        where B.EVENTDATE <= '2016-02-29'
    '''
    MGH_cases = extract_dict_list_with_sql(cursor, sql_string)
    mgh_keyvalue_list = MGH_cases['KeyValue']
    print("MGH nokeyword keyvalue number", len(mgh_keyvalue_list))
    with open("nonkeyword.txt", 'w') as f:
        for a in mgh_keyvalue_list:
            f.write(str(a)+"\n")
    return mgh_keyvalue_list
if __name__ == '__main__':
    # extract_MGH_nokeyword_case_ID()
    # exit(0)
    # calculate_age_word_num_of_both_MGH_BWH()
    # exit(0)
    calculate_age_word_num_of_person("MGH_nokeyword")
    # exit(0)
    calculate_demographic("MGH_nokeyword")
    # calculate_number_of_person("MGH_all")
