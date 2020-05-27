from nltk.tokenize import RegexpTokenizer
from collections import Counter

def exercise_2(file_name, alpha):
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

exercise_2("raven.txt", 1)
exercise_2("gullivers-travels.txt", 1)