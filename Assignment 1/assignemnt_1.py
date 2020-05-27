import string
from tabulate import tabulate
import math
import nltk
import re
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from collections import Counter

def top_3_mostfreq_n_grams(seq_, n):
	my_grams = ngrams(seq_, n)
	counting_ngrams = Counter(my_grams)
	ngrams_sorted = sorted(counting_ngrams, key=counting_ngrams.get, reverse= True)

	return(ngrams_sorted[0:3])


def c_zipf_distribution(file_name, alpha):
	with open(file_name) as g:
		c = 0
		assign_1 = g.read().lower()
		tokenizer = RegexpTokenizer(r'\w+')
		obj_1 = tokenizer.tokenize(assign_1)
		x = Counter(obj_1)
		list_tokens = sorted(x, key=x.get)
		for index, val in enumerate(list_tokens, start=1):
			P_word = x[val]/len(obj_1)
			c += P_word*((index)**alpha)
		total_c_averaged = c/len(list_tokens)
		print("The averaged c of the doc: " + file_name + " is: ", total_c_averaged)


def descriptive_statistics(file_name):
	with open(file_name) as g:
		assign_1 = g.read().lower()
		tokenizer = RegexpTokenizer(r'\w+')
		obj_1 = tokenizer.tokenize(assign_1)
		x = Counter(obj_1)
		list_tokens = sorted(x, key=x.get)
		type_token_ratio = len(list_tokens)/len(obj_1)
		mean_token_length = sum(len(tok) for tok in obj_1)/ len(obj_1)
		entropy = -sum((x[word]/len(obj_1)*(math.log((x[word]/len(obj_1)),2))) for word in list_tokens)
		print(tabulate([['Number of tokens', len(obj_1)], ['Vocabulary size', len(list_tokens)],
			['Type-token-ratio', type_token_ratio], ['Mean token length', mean_token_length],
			['Entropy', entropy], ['Top 3 most frequent 1-grams', top_3_mostfreq_n_grams(obj_1, 1)], 
			['Top 3 most frequent 2-grams', top_3_mostfreq_n_grams(obj_1, 2)],
			['Top 3 most frequent 1-grams', top_3_mostfreq_n_grams(obj_1, 3)]], headers=["Terms", file_name]))
		

def exercise_2():
	c_zipf_distribution("raven.txt", 1)
	c_zipf_distribution("gullivers-travels.txt", 1)

def exercise_3():
	descriptive_statistics("raven.txt")
	print("\n")
	descriptive_statistics("gullivers-travels.txt")


exercise_3()