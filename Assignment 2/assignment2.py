import os
import re
import nltk
import sys
import math
import jsonlines
import pandas as pd 
import numpy as np 
from nltk import ngrams
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


def exercise_1():
	# Most of the code comes from
	# https://www.nltk.org/book/ch05.html
	# Aparently by default uses treebank that is the Penn treebank tag sistem, 
	#woudl be nice to print the list of tags types for each system
	# https://www.nltk.org/_modules/nltk/tag.html
	p = ["It is sunny throughout the year.", "Telling good jokes is an art that comes naturally to some people, but for others it takes practice and hard work.",
     "Research on adult-learned second language(L2) has provided considerable insight into the neurocognitive mechanisms underlying the learning and processing of L2 grammar."]
	tokens = [word_tokenize(i) for i in p]
	peenTreebank = [pos_tag(i) for i in tokens]
	universal = [pos_tag(i, tagset='universal', lang="eng") for i in tokens]
	print(peenTreebank[0][0])
	for x in range(len(peenTreebank)):
		for y in range(len(peenTreebank[x])):
			print(peenTreebank[x][y][0],"|",universal[x][y][1],"|",peenTreebank[x][y][1])


def merge_files(list_of_files):
	with open('main_text_file.txt', 'w') as outfile:
		for file in list_of_files:
			with open(file) as input_:
				outfile.write(input_.read())


def language_model():
	with open("main_text_file.txt", encoding="utf8") as d1:
		3gram_lang = defaultdict(lambda : 0)
		assign_1 = d1.read().lower()
		assign_1 = assign_1.replace('@','')
		assign_1 = assign_1.replace('<p>','')
		obj_1 = assign_1.split()
		my_2_grams = ngrams(obj_1, 2)
		count_2grams = Counter(my_2_grams)
		my_3_grams = ngrams(obj_1, 3)
		count_3grams = Counter(my_3_grams)
		for ele in (count_3grams_1):
			3gram_lang[ele] = count_3grams[ele]/count_2grams[ele[:2]]
	return 3gram_lang

def prob_sent(lang_model, sentence_):
	for sent in sentence_:
		sent_3grams = ngrams(sent.split(), 3)
		prob_ = 0
		for each_gram in sent_3grams:
			prob_ += math.log((language_model[each_gram]))
		likelihood_sente = math.exp(prob_)
		print("Likelihood for getting ( "+sent+" ) is: ", likelihood_sente)


def word_predict(sentence_, lang_model):
	s1= sentence_.lower().split()
	s1 = s1[-2:] 
	next_word = ''
	while(next_word!="."):
		words = dict()
		for key,val in lang_model.items():
			if key[0:2] == tuple(s1):
				words[key[2]] = val
		if len(words)!=0:
			next_word = max(words, key=words.get)
			sentence_ = sentence_+" "+ next_word+" "
			s1[0]=s1[1]
			s1[1]= next_word
		else:
			print("No words in model")
			break
	print("The predicted sentence is: ",sentence_)


to_predict = ["the adventure of", "a student is"]
data = ("he is from the east .", "she is from the east .", 
	"he is from the west .", "she is from the west .")
files = ['text_acad.txt', 'text_blog.txt', 'text_fic.txt',
 'text_mag.txt', 'text_news.txt', 'text_spok.txt', 
 'text_tvm.txt', 'text_web.txt']

def exercise_2():
	merge_files(files)
	3gram_lang_model = language_model()
	prob_sent(3gram_lang_model, data)
	for each in to_predict:
		word_predict(each, 3gram_lang_model)
		print("\n")



exercise_1()
merge_files(files)
language_model()
exercise_2()