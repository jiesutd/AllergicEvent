# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2020-02-19 10:21:05


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np
import subprocess

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', size='9pt'):
	assert(len(text_list) == len(attention_list))
	word_num = len(text_list)
	with open(latex_file,'w') as f:
		f.write(r'''\documentclass['''+size+r''']{standalone}
\special{papersize=310mm,197mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}'''+'\n')
		string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
		for idx in range(word_num):
			string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
		string += "\n}}}"
		f.write(string+'\n')
		f.write(r'''\end{document}''')

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def read_text(input_string):
	pairs = input_string.split()
	word_list = []
	weight_list = []
	for pair in pairs:
		word, weight = pair.rsplit("|",1)
		weight = float(weight)
		word_list.append(word)
		weight_list.append(weight)
	weight_list = norm_weight(weight_list)
	word_list = clean_word(word_list)
	print(weight_list)
	return word_list, weight_list

def norm_weight(weight_list):
	new_weight_list = []
	for weight in weight_list:
		if weight > 5:
			weight = 5
		elif weight < 0:
			weight = 0
		weight = weight*20
		new_weight_list.append(weight)
	return new_weight_list

def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


if __name__ == '__main__':
	input_string = r'''
An|2.308 xxxxx|4.454 to|-0.812 rule|-0.091 out|-0.848 a|-0.431 xxxxx|1.430 xxxxx|1.50 was|-0.245 written|0.434 for|-0.415 both|0.676 xx|1.055 as|0.775 well|-0.250 as|-0.523 the|-0.831 xxxx|-0.314 xxx|-0.894 
'''
	words, weights = read_text(input_string)
	color = 'green'
	generate(words, weights, "sample.tex", color)
	subprocess.run(['/Library/TeX/Distributions/Programs/texbin/pdflatex', 'sample.tex'])
	subprocess.Popen(['open -a Preview sample.pdf'],shell=True)