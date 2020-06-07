import os
import re
import nltk
import sys
import math
import jsonlines
from nltk import ngrams
from collections import Counter
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


def merge_files(list_of_files):
	with open('main_text_file.txt', 'w') as outfile:
		for file in list_of_files:
			with open(file) as input_:
				outfile.write(input_.read())

#l1=os.listdir("E:\Bauhaus Universitat Weimar\SoSem 2020\Intro to NLP\Assignments\Assignment 2\coca-samples-text")
#merge_files(l1)

def trigram_lang_model():
	#data_fr = {"3-gram": None, "probab": None}
	with open("main_text_file.txt", encoding="utf8") as g:
		#with jsonlines.open('3gram_lang_model.jsonl', mode='w') as writer:
			my_dict = defaultdict(lambda : 0)
			my_3grams_count = defaultdict(lambda : 0)
			assign_1 = g.read().lower()
			assign_1 = assign_1.replace('@','')
			assign_1 = assign_1.replace('<p>','')
			obj_1 = assign_1.split()
			my_2_grams = ngrams(obj_1, 2)
			count_2grams = Counter(my_2_grams)
			my_3_grams_1 = ngrams(obj_1[:round(len(obj_1)/2)], 3)
			my_3_grams_2 = ngrams(obj_1[round(len(obj_1)/2):], 3)
			
			count_3grams_1 = Counter(my_3_grams_1)
			count_3grams_2 = Counter(my_3_grams_2)
			"""
			for ele in (count_3grams_1, count_3grams_2):
				for item in ele:
					data_fr["3-gram"] = item
					data_fr["probab"] = (count_3grams_1[item] + count_3grams_2[item])/count_2grams[item[:2]]
					writer.write(data_fr)
					if ele == "count_3grams_1":		
						if item in count_3grams_2:
							del count_3grams_2[item]
			"""
	return (count_3grams_1, count_3grams_2, count_2grams)
		

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


def using_lang_model(sentence_):
	(c_3_1, c_3_2, c_2) = trigram_lang_model()
	for sent in sentence_:
		sent_3grams = ngrams(sent.split(), 3)
		prob_ = 0
		for each_gram in sent_3grams:
			print((c_3_1[each_gram] + c_3_2[each_gram]),c_2[each_gram[:2]])
			prob_ += math.log((c_3_1[each_gram] + c_3_2[each_gram])/c_2[each_gram[:2]])
		likelihood_sente = math.exp(prob_)
		print("Likelihood for getting "+sent+" is: ", likelihood_sente)

data = ("he is from the east .", "she is from the east .", 
	"he is from the west .", "she is from the west .")
using_lang_model(data)



#test_my_idea()