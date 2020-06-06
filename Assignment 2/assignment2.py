import os
import re
import nltk
import sys
import json
import jsonlines
import pandas as pd
from nltk import ngrams
from collections import Counter
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


def merge_files(list_of_files):
	with open('main_text_file.txt', 'w') as outfile:
		for file in list_of_files:
			with open(file) as input_:
				outfile.write(input_.read())

#l1=os.listdir("E:\Bauhaus Universitat Weimar\SoSem 2020\Intro to NLP\Assignments\Assignment 2\coca-samples-text")
#merge_files(l1)

def trigram_lang_model():
	data_fr = {"3-gram": None, "probab": None}
	with open("main_text_file.txt", encoding="utf8") as g:
		my_dict = defaultdict(lambda : 0)
		my_3grams_count = defaultdict(lambda : 0)
		assign_1 = g.read().lower()
		assign_1 = assign_1.replace('@','')
		assign_1 = assign_1.replace('<p>','')
		obj_1 = assign_1.split()
		#print(obj_1[:100])
		my_2_grams = ngrams(obj_1, 2)
		count_2grams = Counter(my_2_grams)
		my_3_grams_1 = ngrams(obj_1[:round(len(obj_1)/2)], 3)
		my_3_grams_2 = ngrams(obj_1[round(len(obj_1)/2):], 3)
	
		count_3grams_1 = Counter(my_3_grams_1)
		count_3grams_2 = Counter(my_3_grams_2)

		data= {'Hey its me':0}
		i = 0
		df = pd.DataFrame(data.items())

		for ele in (count_3grams_1, count_3grams_2):
			for item in ele:
				i=i+1
				df.loc[i] = [TreebankWordDetokenizer().detokenize(item),(count_3grams_1[item] + count_3grams_2[item])/count_2grams[item[:2]]]
				if ele == "count_3grams_1":		
					if item in count_3grams_2:
						del count_3grams_2[item]
		print(df)

			#for item in my_3grams_count:
			#	my_dict[item] = my_3grams_count[item]/count_2grams[item[:2]]
			#print(my_dict[:100])

		

def test_my_idea():
	my_dict = defaultdict(lambda : 0)
	my_3grams_count = defaultdict(lambda : 0)
	str1 = "he is@ @from @@ <p> the east past ."
	str1 = str1.replace('@','')
	str1 = str1.replace('<p>','')
	#tokenizer = RegexpTokenizer(r'\w+')
	obj_1 = str1.split()
	print(obj_1)
	my_2_grams = ngrams(obj_1, 2)
	count_2grams = Counter(my_2_grams)
	my_3_grams = ngrams(obj_1, 3)
	for gram in my_3_grams:
		my_3grams_count[gram] = my_3grams_count[gram] +1
	print(my_3grams_count)
	#count_3grams = Counter(my_3_grams)
	for item in my_3grams_count:
		my_dict[item] = my_3grams_count[item]/count_2grams[item[:2]]
	print(my_dict) 


	#with jsonlines.open('3gram_lang_model.jsonl') as reader:
	#	my_sent = sentence_.split()
	#	my_2_grams = ngrams(obj_1, 3)
		
			


#check_jsonline()
#test_my_idea()
#trigram_lang_model()