#!/usr/bin/env python
# coding: utf-8

# In[12]:


import spacy
from collections import Counter
from tabulate import tabulate
import pandas as pd
from nltk import ngrams
from spacy_wordnet.wordnet_annotator import WordnetAnnotator


# # Task 1

# In[13]:


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
columns = ['Sent_1','POS_1', 'Sensekey_1','Sent_2','POS_2','Sensekey_2', 'Sent_3','POS_3','Sensekey_3']
df_ = pd.DataFrame(columns=columns)
sent1 = "do you train for passing tests or do you train for creative inquiry"
encent = nlp(sent1)
df_['POS_1'] = [token.pos_ for token in encent]
df_['Sent_1']= [token.text for token in encent]
df_['Sensekey_1']= [token._.wordnet.lemmas()[0].key() if len(token._.wordnet.lemmas()) is not 0 else '-' for token in encent]

sent2 = "how it is we have so much information, but know so little"
encent = nlp(sent2)
df_['POS_2'] = [token.pos_ for token in encent]
df_['Sent_2']= [token.text for token in encent]
df_['Sensekey_2']= [token._.wordnet.lemmas()[0].key() if len(token._.wordnet.lemmas()) is not 0 else '-' for token in encent]

sent3 = "it is the responsibility of intellectuals to speak the truth and expose lies"
encent = nlp(sent3)
df_['POS_3'] = [token.pos_ for token in encent]
df_['Sent_3']= [token.text for token in encent]
df_['Sensekey_3']= [token._.wordnet.lemmas()[0].key() if len(token._.wordnet.lemmas()) is not 0 else '-' for token in encent]

# for token in encent:
#     print(token.text, token.pos_)
#     if len(token._.wordnet.synsets()) is not 0: print(token._.wordnet.synsets()[0])
#     if len(token._.wordnet.lemmas()) is not 0: print(token._.wordnet.lemmas()[0].key())
#     break
print(df_)


# # Task 2

# In[14]:


with open('1000-reviews.txt', encoding="utf8") as g:
    file_ = g.read().split()
    vocab = list(set(file_))
    df_ = pd.DataFrame(0 ,index=vocab, columns=vocab)
    #print(len(vocab))
    grams11 = ngrams(file_, 11)
    for gram in grams11:
        gram = list(gram)
        mid_ele = gram[5]
        gram.remove(mid_ele)
        for in_g in gram:
            df_[in_g][mid_ele] += 1
    #print(len(grams11))
    print(df_)


# In[15]:


print(df_['zelda']['does'])
print(df_['fighting']['does'])
print(df_['nintendo']['does'])
print(df_['is']['does'])
print(df_['red']['does'])
df_.shape


# In[16]:


import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=10, svd_solver='randomized',  random_state=40)
pca.fit(df_)
final = pca.transform(df_)


# In[6]:


len(final)


# In[7]:


final[0]


# In[ ]:


with open("word2vec.txt", "w") as text_file:
    text_file.write()


# In[ ]:




