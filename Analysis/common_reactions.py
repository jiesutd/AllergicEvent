# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-04-02 23:04:17
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2020-01-06 00:48:56
from file_io import *
import nltk
import pickle
import pandas as pd
import sys

original_full_string = '''
rxn, reaction, anaphylaxis, hypotension, angioedema, edema, rhinitis, 
bronchospasm, asthma, wheezing, dyspnea, shortness of breath, SOB, 
hives, urticaria, itching, pruritus, pruritis, rash, swelling, tightness, 
closing, shock, gadolinium, contrast, vaccine, midazolam, versed, 
latex, chlorhexidine, curium, fentanyl, morphine, dilaudid, heparin, lovenox, 
enoxaparin, dalteparin, fragmin, vancomycin,  penicillin, PCN,  
floxacin, sulfa, sulpha, bactrim, tegretol, dilantin, carbamazepine, 
ritux, rituximab,  taxol, aspirin, ibuprofen, motrin, advil, 
naproxen, naprosyn, iodine, phenytoin, food, milk, egg, fish, shellfish, crab, 
lobster, shrimp, nut, almond, cashew, pine, pecan, pistachio, wheat, soy, 
benadryl, diphenhydramine, anti-histamine, allegra, steroid, prednisone, prednisolone, pred, albuterol, 
epi-pen
'''
prefix_string = "allerg-, vanc-, cef-, ceph-, carbo-, oxali-, antihistamine-, diphen-, fexofen-, epi-, solu-"
suffix_string = "-caine, -mycin, -cillin, -platin"

original_full_keywords = original_full_string.lower().strip('\n').replace('\n', "").split(', ')
prefix_keywords = prefix_string.lower().replace("-", "").split(", ")
suffix_keywords = suffix_string.lower().replace("-", "").split(", ")
print("Keyword num:",len(original_full_keywords), len(prefix_keywords), len(suffix_keywords))


def in_original_keywords(input_word):
	if input_word in original_full_keywords:
		return True 
	for each in prefix_keywords:
		if input_word.startswith(each):
			return True 
	for each in suffix_keywords:
		if input_word.endswith(each):
			return True 
	return False



def mining_words_with_large_attention(input_file1, input_file2=None):
	attention, prob = xlsx_extract_column(input_file1, ["E", "C"], True)
	total_number = len(attention)
	assert(total_number == len(prob))
	print("Total num:", total_number)
	if input_file2:
		attention_add, prob_add = xlsx_extract_column(input_file2, ["E", "C"], True)
		attention = attention + attention_add
		prob = prob + prob_add
		print("Updated total number: %s + %s -> %s"%(total_number, len(attention_add),total_number+ len(attention_add) ))
		total_number += len(attention_add)
	count_threshold = [0]*10
	for idx in range(total_number):
		probability = float(prob[idx])
		for idy in range(10):
			if idy/10.0<=probability<(idy+1)/10.0:
				count_threshold[idy] += 1

	print(count_threshold)
	# exit(0)


	positive_list = []
	for idx in range(total_number):
		probability = float(prob[idx])
		if probability > 0.8:
			positive_list.append(attention[idx])		
	print(len(positive_list))
	total_word_dict = {}
	for each_string in positive_list:
		# word_dict = extract_significant_phrase_each_note(each_string)
		word_dict = extract_significant_word_each_note(each_string, 1)
		for each_word in word_dict.keys():
			if each_word in total_word_dict:
				total_word_dict[each_word] += word_dict[each_word]
			else:
				total_word_dict[each_word] = word_dict[each_word]
	total_word_dict = [ (v,k) for k,v in total_word_dict.items() ]
	total_word_dict.sort(reverse=True) # natively sort tuples by first element
	count = 0
	for v,k in total_word_dict:
		if v > 0:
			if len(k.split()) > 0:
				print(k, v)
				# print("%s (%d)" % (k,v))
				count += 1
	    # print(k)
	print("Total keywords:", len(total_word_dict), "after filter:", count)


def extract_significant_word_each_note(note_string, over_sigma = 3):
	pairs = note_string.split(' ')
	word_dict = {}
	for each_pair in pairs:
		word_weight = each_pair.rsplit('|',1)
		word = word_weight[0].lower()
		try :
			a = float(word_weight[-1])
		except:
			continue

		if float(word_weight[-1]) > over_sigma:
			if not in_original_keywords(word):
			# if True:
				if word in word_dict:
					word_dict[word] += 1 
				else:
					word_dict[word] = 1
	return word_dict

def extract_significant_phrase_each_note(note_string, over_sigma = 2):
	pairs = note_string.split(' ')
	phrase_dict = {}
	pair_num = len(pairs)
	prev_significant = False
	the_phrase = ""
	phrase_length = 0
	for idx in range(pair_num):
		each_pair = pairs[idx]
		word_weight = each_pair.rsplit('|',1)
		word = word_weight[0].lower()
		try:
			std_att = float(word_weight[-1])
		except ValueError:
			continue
		if  std_att> over_sigma:
			if prev_significant:
				the_phrase += " "+word
				phrase_length += 1
			else:
				prev_significant = True 
				the_phrase = word
				phrase_length = 1
		else:
			if phrase_length > 1:
				if the_phrase in phrase_dict:
					phrase_dict[the_phrase] += 1 
				else:
					phrase_dict[the_phrase] = 1 
			prev_significant = False
			the_phrase = ""
			phrase_length = 0
	if phrase_length > 1:
		if the_phrase in phrase_dict:
			phrase_dict[the_phrase] += 1 
		else:
			phrase_dict[the_phrase] = 1 
	prev_significant = False
	the_phrase = ""
	phrase_length = 0
	return phrase_dict


def in_original_keywords(input_word):
	if input_word in original_full_keywords:
		return True 
	for each in prefix_keywords:
		if input_word.startswith(each):
			return True 
	for each in suffix_keywords:
		if input_word.endswith(each):
			return True 
	return False



def mining_notes_without_original_keywords(input_file, output_file):
	original, attention, prob, ids, sigma  = xlsx_extract_column(input_file, ["D", "E", "C", "A", "F"], True)

	total_number = len(attention)
	assert(total_number == len(prob))
	print("Total num:", total_number)
	positive_att_list = []
	positive_org_list = []
	prob_list = []
	id_list = []
	sigma_list = []
	for idx in range(total_number):
		probability = float(prob[idx])
		if probability < 10:
			positive_att_list.append(attention[idx])
			positive_org_list.append(original[idx])
			prob_list.append(probability)
			id_list.append(ids[idx])
			sigma_list.append(sigma[idx])
	pos_num = len(positive_att_list)
	print("POS num:", pos_num)
	without_keyword_num = 0
	with open(output_file,'w') as cwriter:
		the_writer = csv.writer(cwriter)
		the_writer.writerow(["ID", "Gold", "Pred", "Des", "Des_attention", ">4Sigma Found"])
		for idx in range(pos_num):
			if not note_include_original_keywords(positive_org_list[idx]):
				# print(positive_att_list[idx]+"\n")
				without_keyword_num += 1 
				the_writer.writerow([id_list[idx], "--", prob_list[idx], positive_org_list[idx], positive_att_list[idx],sigma_list[idx]])
	print("Positive without keyword num:", without_keyword_num)



def note_include_original_keywords(note_string):
	note_string = note_string.strip("\n").lower()
	words = nltk.word_tokenize(note_string)
	for word in words:
		if in_original_keywords(word):
			return True 
	return False

def average_weight_with_frequency(input_file1, input_file2=None):
	attention, prob = xlsx_extract_column(input_file1, ["E", "C"], True)
	total_number = len(attention)
	assert(total_number == len(prob))
	print("Total num:", total_number)
	if input_file2:
		attention_add, prob_add = xlsx_extract_column(input_file2, ["E", "C"], True)
		attention = attention + attention_add
		prob = prob + prob_add
		print("Updated total number: %s + %s -> %s"%(total_number, len(attention_add),total_number+ len(attention_add) ))
		total_number += len(attention_add)
	count_threshold = [0]*10
	for idx in range(total_number):
		probability = float(prob[idx])
		for idy in range(10):
			if idy/10.0<=probability<(idy+1)/10.0:
				count_threshold[idy] += 1
	print(count_threshold)

	positive_list = []
	for idx in range(total_number):
		probability = float(prob[idx])
		if probability > 0.8:
		# if True:
			positive_list.append(attention[idx])		
	print(len(positive_list))
	total_word_dict = {}
	whole_string = " ".join(positive_list)
	whole_word_count = len(whole_string.split())
	total_word_dict = extract_weight_value_num(whole_string)
	total_word_avg_score = {}
	for k, v in total_word_dict.items():
		total_word_avg_score[k] = (v[1]+0.)/v[0]
	total_word_avg_score = [ (v,k) for k,v in total_word_avg_score.items() ]
	total_word_avg_score.sort(reverse=True) # natively sort tuples by first element
	count = 0
	output_draw_list = []
	for v,k in total_word_avg_score:
		if v > 0 or in_original_keywords(k):
			only_expert_key =False
			if v <= 0:
				only_expert_key =True  
			if len(k.split()) > 0:# and total_word_dict[k][0] > 5:
				the_list = [k, v, (total_word_dict[k][0]+0.)/whole_word_count, total_word_dict[k][0], in_original_keywords(k), only_expert_key ]
				output_draw_list.append(the_list)
				print(k, v, (total_word_dict[k][0]+0.)/whole_word_count, total_word_dict[k][0], in_original_keywords(k) , only_expert_key)
				# print("%s (%d)" % (k,v))
				count += 1
	    # print(k)
	print("Total keywords:", len(total_word_avg_score), "after filter:", count)
	print("Whole word count: ", whole_word_count)
	# draw_word_distribution(output_draw_list)
	with open("//cifs2/allergycrico$/Year 1_ RL Solutions NLP Studies/Project 1 - Identifying hypersensitivity reactions/Jie/Data/output_draw_list1.pk", 'wb') as fout:
		pickle.dump(output_draw_list, fout)
	return output_draw_list


def extract_key_reactions_from_validated_cases(case_xlsx):
	df = pd.read_excel(case_xlsx)
	print("All positive num:", df.shape[0])
	total_case_num = df.shape[0]
	reaction_dict = {}
	reaction_dict.update({a:"Itching" for a in ["itching", "itchy", "itchiness", "pruritus", "pruritis"]})
	reaction_dict.update({a:"Shortness of breath" for a in ["shortness of breath", "sob", "dyspnea", "breathing"]})
	reaction_dict.update({a:"Bronchospasm/wheezing/ chest tightness" for a in ["bronchospasm", "tightness", "wheezing", "tenderness"]})
	reaction_dict.update({a:"Rash" for a in ["rash"]})
	reaction_dict.update({a:"Anaphylaxis" for a in ["anaphylaxis", "anaphylactic", "shock"]})
	reaction_dict.update({a:"Hives" for a in ["hives", "hive", "urticaria"]})
	reaction_dict.update({a:"Angioedema" for a in ["angioedema", "swelling", "edema", "swollen"]})
	reaction_dict.update({a:"Throat tightness" for a in ["closing"]})
	reaction_dict.update({a:"Erythema or flushing" for a in ["erythema", "flushing", "warmth", "hot", "reddness", "redness", "burning"]})  
	reaction_dict.update({a:"Tachycardia" for a in ["tachycardia"]})
	reaction_dict.update({a:"Gastrointestinal symptoms" for a in ["nausea", "vomiting", "vomited"]})
	reaction_dict.update({a:"Fever or chills" for a in ["fever", "chills"]})
	reaction_dict.update({a:"Sneezing" for a in ["sneezed", "sneezing"]})
	reaction_dict.update({a:"Nasal congestion" for a in ["congestion"]})
	reaction_dict.update({a:"Cough" for a in ["cough", "coughing"]})
	reaction_dict.update({a:"Headache" for a in ["headache"]})
	reaction_dict.update({a:"Sensation" for a in ["tingling", "sensation"]})
	reaction_dict.update({a:"Hematologic side effect reactions" for a in ["bleed", "hematoma", "numbness"]})
	reaction_dict.update({a:"Altered mental status" for a in ["syncopal", "seizure"]})
	reaction_dict.update({a:"Hypotension" for a in ["hypotension"]})
	reaction_dict.update({a:"Skin break" for a in ["blistering", "bruising", "bruise", "bump", "lump", "abrasion", "laceration", "sore"]})
	reaction_dict.update({a:"Slurred speech" for a in ["slurred"]})
	reaction_dict.update({a:"Rhinitis" for a in ["rhinitis"]})
	reaction_dict.update({a:"Asthma" for a in ["asthma"]})
	reaction_dict.update({a:"Tremor" for a in ["shaking"]})
	reaction_dict.update({a:"Infiltration" for a in ["infiltrate", "infiltrated", "infiltration"]})
	reaction_dict.update({a:"General Terms (allergy/hypersensistivity)" for a in ["allergy", "allergic", "allergies", "hypersensitivity"]})
	reaction_dict.update({a:"General terms (reaction/symptoms) " for a in ["rxn", "reactions", "symptoms", "discomfort", "episode"]})
	# print(reaction_dict, len(reaction_dict))
	reaction_type_list = list(set(reaction_dict.values()))
	result_list = [[0,0] for idx in range(len(reaction_type_list))]  
	reaction_type_distribution = dict(zip(reaction_type_list, result_list))
	for attention_text in df['Des_attention']:
		attention_text = attention_text.lower()
		keyword_dict = extract_weight_max_value(attention_text)
		for keyword in keyword_dict:
			if keyword in reaction_dict:
				reaction_type = reaction_dict[keyword]
				reaction_type_distribution[reaction_type][0] += 1
	
	for k, v in reaction_type_distribution.items():
		reaction_type_distribution[k][1] = (v[0]+0.)/total_case_num
	print("**"*50)
	for k, v in reaction_type_distribution.items():
		print(k, v)





def draw_word_distribution(input_list=None):
	## input_list: [a], 
	## 		a:[k, v, feq, count, ifkey, only_expert_key]
	#			K: weyword
	#			V: average_score
	#			feq: frequence
	#			count: word count
	#			ifkey(boolen): if in original keywords
	#			only_expert_key(boolen): if only in expert created
	import seaborn as sns
	import matplotlib.pyplot as plt
	import pandas as pd
	from adjustText import adjust_text
	font_size = 12
	if not input_list:
		infile = open('../../Data/output_draw_list1.pk', 'rb')
		input_list = pickle.load(infile)
	words = []
	avg_scores = []
	freqs = []
	key_words = []
	key_avg_scores = []
	key_freqs = []
	exp_key_words = []
	exp_key_avg_scores = []
	exp_key_freqs = []
	ifkeys = []
	print("Load Total number:", len(input_list))
	for k, v, freq, count, ifkey, only_expert_key in input_list:
		print(count, only_expert_key)
		if  count <= 5:
			continue
		if ifkey and count and not only_expert_key:
			key_words.append(k)
			key_avg_scores.append(v)
			key_freqs.append(freq)
		elif only_expert_key:
			exp_key_words.append(k)
			exp_key_avg_scores.append(v)
			exp_key_freqs.append(freq)
		else:
			words.append(k)
			avg_scores.append(v)
			freqs.append(freq)
		if ifkey:
			print(k, "*",)
		else:
			print(k,)
		ifkeys.append(ifkey)
	print("Vis Total number:", len(ifkeys))
	df = pd.DataFrame(list(zip(words, avg_scores,freqs,ifkeys)), 
               columns =['word', 'score', 'freq', 'ifkey'])
	all_avg_scores = np.asarray(avg_scores+key_avg_scores)
	avg_scores =np.asarray(avg_scores)
	key_avg_scores = np.asarray(key_avg_scores)
	exp_key_avg_scores = np.asarray(exp_key_avg_scores)

	norm = plt.Normalize(all_avg_scores.min(), all_avg_scores.max())
	norm_y = norm(avg_scores)
	key_norm_y = norm(key_avg_scores)
	exp_key_norm_y = norm(exp_key_avg_scores)
	key_color = []
	b =  key_avg_scores.tolist()
	for a in b:
		# if a <0.1:
		# 	key_color.append('cornflowerblue')
		# elif a<0.2:
		# 	key_color.append('royalblue')
		# elif a <0.5:
		# 	key_color.append('slateblue')
		# elif a < 0.8:
		key_color.append('green')
		# else:
		# 	key_color.append('darkblue')
	exp_key_color = ['orange']*len(exp_key_avg_scores.tolist())
	the_color = []
	b =  avg_scores.tolist()
	for a in b:
		# if a <0.1:
		# 	the_color.append('lightcoral')
		# elif a<0.2:
		# 	the_color.append('salmon')
		# elif a <0.4:
		# 	the_color.append('tomato')
		# elif a < 0.9:
		the_color.append('blue')
		# else:
		# 	the_color.append('darkred')
	plt.figure(figsize=(13,9))
	plt.xscale('log')
	plt.yscale('symlog')
	plt.scatter(freqs, avg_scores, marker='o', c=the_color)
	plt.scatter(key_freqs, key_avg_scores, marker='s', c=key_color)
	plt.scatter(exp_key_freqs, exp_key_avg_scores, marker='^', c=exp_key_color)
	texts = []
	text_font_size = font_size-2
	for i, txt in enumerate(words):
		texts.append(plt.annotate(txt, (freqs[i]+freqs[i]*(10**0.02-1), avg_scores[i]-avg_scores[i]*(10**0.01-1)), color='blue', fontsize=text_font_size))
	for i, txt in enumerate(key_words):
		texts.append(plt.annotate(txt, (key_freqs[i]*(1+10**0.02-1), key_avg_scores[i]*(1-(10**0.01-1))), color= 'green', fontsize=text_font_size))
	for i, txt in enumerate(exp_key_words):
		print(txt, exp_key_freqs[i], exp_key_avg_scores[i])
		texts.append(plt.annotate(txt, (exp_key_freqs[i]*(1+(10**0.02-1)), exp_key_avg_scores[i]*(1-(10**(-0.01)-1))), color= 'orange', fontsize=text_font_size))
	#adjust_text(texts)
	plt.ylabel('Word Importance',fontname = "Arial", fontsize=font_size+3)
	plt.xlabel('Word Frequency',fontname = "Arial", fontsize=font_size+3)
	y_ticks = [-10e-2, 0,  10e-1,10e0, 10e1]
	plt.yticks(y_ticks, fontsize=font_size)
	plt.ylim([-3*10e-2,10e0])
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.tight_layout()
	plt.tick_params(labelsize=font_size+3)
	plt.savefig("../../Data/Hkey_vis.pdf") 
	plt.show()





def extract_weight_value_num(note_string):
	pairs = note_string.split(' ')
	word_dict = {}
	for each_pair in pairs:
		word_weight = each_pair.rsplit('|',1)
		word = word_weight[0].lower()
		try :
			weight = float(word_weight[-1])
		except:
			continue
		if word in word_dict:
			word_dict[word] = [word_dict[word][0] + 1, word_dict[word][1]+weight]
		else:
			word_dict[word] = [1,weight]
	return word_dict

def extract_weight_max_value(note_string):
	pairs = note_string.split(' ')
	word_dict = {}
	for each_pair in pairs:
		word_weight = each_pair.rsplit('|',1)
		word = word_weight[0].lower()
		try :
			weight = float(word_weight[-1])
		except:
			continue
		if word in word_dict:
			word_dict[word] = max(word_dict[word],weight)
		else:
			if weight > 2:
				word_dict[word] = weight
	return word_dict


if __name__ == '__main__':

	extract_key_reactions_from_validated_cases(sys.argv[1])