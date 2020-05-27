import string
import math
import nltk
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from collections import Counter

def top_3_mostfreq_n_grams(seq_, n):
	my_grams = ngrams(seq_, n)
	counting_ngrams = Counter(my_grams)
	ngrams_sorted = sorted(counting_ngrams, key=counting_ngrams.get, reverse= True)

	return(ngrams_sorted[0:3])

def z_scores(tok_obj_1, tok_obj_2, tok_obj_3, tok_obj_cur):
	z_scores_local = []
	x = Counter(tok_obj_1)
	y = Counter(tok_obj_2)
	z = Counter(tok_obj_3)
	ref_count = Counter(tok_obj_cur)
	list_tokens_cur = sorted(ref_count, key=ref_count.get, reverse= True)
	#list_tokens_1 = sorted(x, key=x.get, reverse= True)
	#list_tokens_2 = sorted(y, key=y.get, reverse= True)
	#list_tokens_3 = sorted(z, key=z.get, reverse= True)
	for word in list_tokens_cur:
		mean_all = ((x[word]/len(tok_obj_1) + y[word]/len(tok_obj_2) + z[word]/len(tok_obj_3))/3)
		std_dev = math.sqrt((((x[word]+y[word]+z[word]) - mean_all)**2)/(len(tok_obj_1)+len(tok_obj_2)+len(tok_obj_3)))
		z_score_1 = ((ref_count[word]/len(tok_obj_cur))- mean_all)/std_dev
		z_scores_local.append(z_score_1)

	return z_scores_local


def c_zipf_distribution(file_name, alpha):
	with open(file_name, encoding="utf8") as g:
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
	with open(file_name, encoding="utf8") as g:
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
			['Top 3 most frequent 3-grams', top_3_mostfreq_n_grams(obj_1, 3)]], headers=["Terms", file_name]))
		

def exercise_4_b(file_1, file_2, file_3):
	with open(file_1, encoding="utf8") as first_f, open(file_2, encoding="utf8") as second_f, open(file_3, encoding="utf8") as third_f:
		assign_1 = first_f.read().lower()
		assign_2 = second_f.read().lower()
		assign_3 = third_f.read().lower()	
		tokenizer = RegexpTokenizer(r'\w+')
		obj_1 = tokenizer.tokenize(assign_1)
		obj_2 = tokenizer.tokenize(assign_2)
		obj_3 = tokenizer.tokenize(assign_3)
		z_scores_doc1 = z_scores(obj_1, obj_2, obj_3, obj_1)
		z_scores_doc2 = z_scores(obj_1, obj_2, obj_3, obj_2)
		z_scores_doc3 = z_scores(obj_1, obj_2, obj_3, obj_3)
		#z_score_1 = [((x[word]/len(obj_1)) - ((x[word]/len(obj_1) + y[word]/len(obj_2) + z[word]/len(obj_3))/3))/ ]
		#print(z_scores_doc1[0:10])
		data = {file_1: z_scores_doc1[0:50], file_2: z_scores_doc2[0:50], file_3: z_scores_doc3[0:50]}
		df = pd.DataFrame(data)
		df.plot(kind='bar')
		plt.show()


def exercise_2():
	c_zipf_distribution("raven.txt", 1)
	c_zipf_distribution("gullivers-travels.txt", 1)

def exercise_3():
	descriptive_statistics("raven.txt")
	print("\n")
	descriptive_statistics("gullivers-travels.txt")

def exercise_4_a():
	descriptive_statistics("austen.txt")
	print("\n")
	descriptive_statistics("bronte.txt")
	print("\n")
	descriptive_statistics("disputed.txt")


#exercise_4_a()
#exercise_2()
#print("\n")
#exercise_3()
exercise_4_b("austen.txt", "bronte.txt","disputed.txt")