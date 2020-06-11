"""
Lab Class NLP: Words
Assignment 2
Satpute, Ankit : Mtr. 120825
Becker, Gabriel : Mtr. 120770
Ali, Ukasha : Mtr. 120798
"""
import math
from nltk import ngrams
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict


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
	# This function merges all the files and then converts them into main_text_file.txt and writes it in the folder.
	with open('main_text_file.txt', 'w') as outfile:
		for file in list_of_files:
			with open(file) as input_:
				outfile.write(input_.read())

def language_model():
	"""
	This function turns the main_text_file.txt into lowercase and then removes @ and <p> and tokenizes the document then
	converts it into 2 grams and 3 grams and then to evaluate the probability the count of the element of 3 grams is
	divided by the element of 2 grams.
	"""
	with open("main_text_file.txt", encoding="utf8") as d1:
		lang_3gram = defaultdict(lambda : 0)
		assign_1 = d1.read().lower()
		assign_1 = assign_1.replace('@','')
		assign_1 = assign_1.replace('<p>','')
		obj_1 = assign_1.split()
		my_2_grams = ngrams(obj_1, 2)
		count_2grams = Counter(my_2_grams)
		my_3_grams = ngrams(obj_1, 3)
		count_3grams = Counter(my_3_grams)
		for ele in (count_3grams):
			lang_3gram[ele] = count_3grams[ele]/count_2grams[ele[:2]]
	return lang_3gram

def prob_sent(lang_model, sentence_):
	"""
	This function takes sentences as input and converts it into 3 grams first and then for each
	gram in the 3 grams it calculates the logarithm of the probability of that n-gram and in the end, it calculates
	exp of summation of all previous individual values.
	"""
	for sent in sentence_:
		sent_3grams = ngrams(sent.split(), 3)
		prob_ = 0
		for each_gram in sent_3grams:
			prob_ += math.log(lang_model[each_gram])
		likelihood_sente = math.exp(prob_)
		print("Likelihood for getting ( "+sent+" ) is: ", likelihood_sente)


def word_predict(sentence_, lang_model):
	"""
	The following function predicts the next word in the given sentence until a full stop comes.
	"""
	s1= sentence_.lower().split()
	"""Now considering last two token of the given sentence only"""
	s1 = s1[-2:]
	next_word = ''
	while(next_word!="."):
		words = dict()
		for key,val in lang_model.items():
			if key[0:2] == tuple(s1):
				words[key[2]] = val
		if len(words)!= 0:
			next_word = max(words, key=words.get)
			sentence_ = sentence_+" "+ next_word
			s1[0]=s1[1]
			s1[1]= next_word
		else:
			print("No words in model")
			break
	print("The predicted sentence is: ",sentence_)


to_predict = ["the adventure of", "a student is", "easiest way to", "manager of the firm"]
data = ("he is from the east .", "she is from the east .",
		"he is from the west .", "she is from the west .",
		"we have a problem .", "this is right .")
files = ['text_acad.txt', 'text_blog.txt', 'text_fic.txt',
 'text_mag.txt', 'text_news.txt', 'text_spok.txt',
 'text_tvm.txt', 'text_web.txt']

def exercise_2():
	merge_files(files)
	langmodel_3gram = language_model()
	prob_sent(langmodel_3gram, data)
	for each in to_predict:
		word_predict(each, langmodel_3gram)
		print("\n")


exercise_1()
print("\n")
exercise_2()