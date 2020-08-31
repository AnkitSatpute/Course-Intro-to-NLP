#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import spacy
import os
import string, re
import numpy as np
import operator
import codecs
import jsonlines
sys.path.append('/home/pogo6249/Desktop/vroniplag_corpus/seed_and_extension')
import perfo_measures
from semantic_text_similarity.models import WebBertSimilarity
model = WebBertSimilarity(device='cpu')
from collections import defaultdict, OrderedDict, namedtuple
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sklearn.cluster import DBSCAN
np.set_printoptions(threshold=sys.maxsize)


# In[ ]:


TREF, TOFF, TLEN = 'this_reference', 'this_offset', 'this_length'
SREF, SOFF, SLEN = 'source_reference', 'source_offset', 'source_length'
EXT = 'is_external'
Annotation = namedtuple('Annotation', [TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT])
TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT = range(7)

#test_cases
A = Annotation('my_file',1,25,'src_file',3,55,True)
B = Annotation('my_file',8,25,'src_file',6,55,True)
C = Annotation('susp',3583,80,'src',1300,68,True)
D = Annotation('susp',3593,698,'src',1300,698,True)


# In[ ]:


DELETECHARS = ''.join([string.punctuation, string.whitespace])
#LENGTH = 50
#p=['Cloutier.  Over the years, he has finished in the money in over 30 WSOP poker game events.', "Clotiia. Over the years, he completed 30 more than money in the event of the World Series of Poker is a poker game."]
p = ["We like to travel and go to various places", "Travelling and going to lot of places is liked by them"]
#p = ['We like to travel and go to various places','he has finished in the money in over 30 WSOP poker game events.']


# In[ ]:


#test
from semantic_text_similarity.models import WebBertSimilarity

model = WebBertSimilarity(device='cpu') #defaults to GPU prediction

model.predict([("She won an olympic gold medal","The women is an olympic champion")])


# In[ ]:


import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
import math
def scoring(pair):
    import math
    query_vec_1, query_vec_2 = bc.encode(pair)
    cosine = np.dot(query_vec_1, query_vec_2) / (np.linalg.norm(query_vec_1) * np.linalg.norm(query_vec_2))
    #return cosine
    return 1/(1 + math.exp(-100*(cosine - 0.95)))

with BertClient(port=5555, port_out=5556, check_version=False) as bc:
    for i in range(1):
        print("Similarity of Pair : ",scoring(p))


# In[ ]:


def tokenize_sentence(text):
    my_dict=OrderedDict()
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    sent_to_pipe = nlp(text)
    sent_tokens = [[sent.string.strip(), sent.start_char, sent.end_char] for sent in sent_to_pipe.sents]
    for sen in sent_tokens:
        temp_sent= list(sen[0])
        for i in list(temp_sent):
            if i in DELETECHARS:
                del i
        sen[0] = ''.join(temp_sent)
    for idx,sent in enumerate(sent_tokens):
        my_dict[idx] = (sent)
    return my_dict


# In[ ]:


class seed_and_extension_baseline:
    def __init__(self, susp, src):
        self.susp = susp
        self.src = src
        
    def process(self):
        self.preprocess()
        b = self.extension()
        return b
    
    def preprocess(self):
        susp_fp = codecs.open(self.susp, 'r', 'utf-8')
        self.susp_text = susp_fp.read()
        self.tokens_susp = tokenize_sentence(self.susp_text)
        susp_fp.close()
        src_fp = codecs.open(self.src, 'r', 'utf-8')
        self.src_text = src_fp.read()
        self.tokens_src = tokenize_sentence(self.src_text)
        src_fp.close()
        
    def seed(self):
        seeds = []
#uncomment if you want to use web-service
#         with BertClient(port=5555, port_out=5556, check_version=False) as bc:
#             for keyd1, value_d1 in self.tokens_susp.items():
#                 for keyd2, value_d2 in self.tokens_src.items():
#                     pair1 = [value_d1[0],value_d2[0]]
#                     print(pair1)
#                     query_vec_1, query_vec_2 = bc.encode(pair1)
#                     cosine = np.dot(query_vec_1, query_vec_2) / (np.linalg.norm(query_vec_1) * np.linalg.norm(query_vec_2))
#                     score = 1/(1 + math.exp(-100*(cosine - 0.95)))
#                     print(score)
#                     if score > 0.5:
#                         print([keyd1, keyd2])
#                         seeds.append([keyd1, keyd2])
        
        for keyd1, value_d1 in self.tokens_susp.items():
            for keyd2, value_d2 in self.tokens_src.items():
                if model.predict([(value_d1[0],value_d2[0])]) > 3:
                    seeds.append([keyd1, keyd2])
        
        return seeds
    
    def extension(self):
        detected_clust= []
        matches = self.seed()
        
        if len(matches) < 1:
            detected_clust.append([0,0,0,0])
            return detections_clust
        
        db_default = DBSCAN(eps = 4, min_samples = 2).fit(matches)
        data = {k: [] for k in db_default.labels_}

        for idx,val in enumerate(db_default.labels_):
            data[val].append(matches[idx])
            
        if -1 in data.keys(): del data[-1]    
            
        if len(data) == 0: 
            detected_clust.append([0,0,0,0])
            return detections_clust
        
        for clust_cent in data.keys():
            elements = data[clust_cent]
            off_doc_1 = self.tokens_susp[elements[0][0]][1]
            len_doc_1 = self.tokens_susp[elements[-1][0]][2] - off_doc_1
            off_doc_2 = self.tokens_src[elements[0][1]][1]
            len_doc_2 = self.tokens_src[elements[-1][1]][2] - off_doc_2
            detected_clust.append([off_doc_1, len_doc_1, off_doc_2, len_doc_2])
        #print(detected_clust)
        return detected_clust


# In[ ]:


def baseline_checking(file_, dir1 , dir2):
    cases = []
    i = 0
    detections = []
    with jsonlines.open(file_) as reader:
        for item in reader:
            i = i + 1
            case= Annotation(item['suspicious-id'], item['alignments'][0]['suspicious-alignment-start'],
                            item['alignments'][0]['suspicious-alignment-end']-item['alignments'][0]['suspicious-alignment-start'],
                            item['source-id'], item['alignments'][0]['source-alignment-start'],
                            item['alignments'][0]['source-alignment-end']-item['alignments'][0]['source-alignment-start'], True)
            cases.append(case)
            susp_doc = os.path.join(dir1, item['suspicious-id'])
            src_doc = os.path.join(dir2, item['source-id'])
            a = seed_and_extension_baseline(susp_doc, src_doc)
            detection_from_baseline = a.process()
            #print(detection_from_baseline)
            for val in detection_from_baseline:
                detection = Annotation(item['suspicious-id'], val[0], val[1],
                            item['source-id'], val[2] , val[3] , True)
                detections.append(detection)
            if i > 2:
                break
    return cases, detections


# ## Pan13 (test, no obfuscation)

# In[ ]:


dir_2 = "/home/pogo6249/Desktop/vroniplag_corpus/data/pan13-ta-test-and-training/test/src"
dir_1 = "/home/pogo6249/Desktop/vroniplag_corpus/data/pan13-ta-test-and-training/test/susp"
#src_dir = os.path.join
cases_o, detections_o = baseline_checking("/home/pogo6249/Desktop/vroniplag_corpus/data/pan13-ta-test-and-training/test/PAN13_TA_test_NO_Obfs.jsonl", dir_1, dir_2)

recall, precision = perfo_measures.micro_avg_recall_and_precision(cases_o, detections_o)
granu = perfo_measures.granularity(cases_o, detections_o)
pladget = perfo_measures.plagdet_score(recall,precision, granu)
print("recall: ", recall, " precision: ", precision, "Granularity: ", granu, "Pladget: ", pladget)


# In[ ]:


#for the remianing test add directory for different obfuscation and repeat the process


# ## Test_cases

# In[ ]:


#defaults to GPU prediction
#summary
s1 = 'From a helicopter the wavewashed beach looks as if the worst oil spill in U.S. history had never touched it. Silvery sticks of driftwood poke through a deep blanket of snow and smooth gray pebbles roll in the surf under the gaze of a bald eagle perched in a shoreside spruce. But the view doesnt impress Joe Bridgman of the Alaska Department of Environmental Conservation. Dashing out as the chopper lands he digs into the cobble beach and quickly finds what he knew he would. Oil he says. Smell it The pungent odor of petroleum wafts through the air as the hole turns black with crude oil an oozing remnant of the 10.8 million gallons spilled into Prince William Sound last March 24 by the tanker Exxon Valdez. Bridgman scoops up a shovelful of gravel lugs it to the waters edge and dumps it in. A rainbow sheen of oil spreads across the water. Hundreds of gallons of oil are locked up under this beach he says. And this isnt isolated. There are hundreds of beaches all over the sound that are still oiled and the oil is slowly bleeding out. The beaches can look beautiful at the surface but you can dig down in this case just a few inches below the surface and find lots of oil. Now is that a threat or isnt it A year after the wreck of the Exxon Valdez the question clings like the oil under this Perry Island beach. Certainly the worst is over thousands of dead birds no longer wash up on shorelines as they did last summer. But assessing the continuing damage wrought by the nations most extensive  and expensive  oil spill has just begun. As a growing slick of lawyers haggles over who is to blame Exxon Corp. and government agencies debate how to clean up whats left and scientists track wildlife populations first steps on the long road to recovery. Any hope of a quick solution faded last summer as oil from the Exxon Valdez spread across 1100 miles of Alaskas wild southern coast. A cleanup army of 12000 workers polished rocks by hand blasted beaches with hot water and sprayed fertilizer to promote the growth of oileating microbes. But when Exxon suspended its 2 billion cleanup in midSeptember it had recovered only 5 percent to 9 percent of the oil spilled state officials estimate. About 20 percent to 40 percent is believed to have evaporated. That leaves 50 percent to 75 percent of the oil in the water on the ocean bottom or on beaches. Some was soaked up by unwilling sponges the seabirds eagles and sea otters whose carcasses now lie frozen in five vans in an Anchorage storage yard awaiting their day as physical evidence in court. Workers found more than 1000 dead otters a sizable chunk of the spill areas total population of 15000 to 22000. Many of Prince William Sounds 3000 bald eagles also suffered at least 151 died most poisoned by scavenging the oily remains of some of the 34400 dead seabirds recovered. Those numbers alone make the Valdez spill the most lethal ever but scientists say the actual death count is much higher estimating that up to 90 percent of the seabirds caught in oil sank from sight or drifted out to sea. Exxon notes the spill did not wipe out any species and says surviving animals and birds will rebuild populations. But that may take up to 70 years for some hardhit seabird colonies U.S. Fish and Wildlife Service researchers say. We never claimed that the spill put any animal on the endangered species list but thats missing the point said Fish and Wildlife spokesman Bruce Batten. Its still the greatest humancaused wildlife disaster that this agency knows about. Oily carcasses were an obvious measure of the spills impact but victims also included less visible members of the ecosystem such as young salmon and tiny intertidal creatures. Assessment studies for these populations are not finished and even preliminary findings are hard to come by  researchers have been told by lawyers to save their findings for court where it seems nearly everyone involved in the spill is headed. Capt. Joseph Hazelwood skipper of the Exxon Valdez is on trial this month in Anchorage on charges including criminal mischief and drunken driving of his vessel and a federal grand jury recently issued criminal indictments against Exxon starting a case that could take years to finish. Exxon already faces more than 150 civil lawsuits. Fishermen sued because of lost seasons. Tourboat operators sued because fewer people wanted to cruise an oiled sound. The state sued claiming the company was negligent in responding to the spill only to be countersued by Exxon which claimed state officials hindered the use of chemical dispersants that could have broken up large quantities of oil early on. Information about the spill is filtered through this litigious atmosphere making much of it suspect. Exxon distributes beforeandafter pictures of cleaned beaches Bridgman and other state officials accusing Exxon of mythmaking eagerly make room for journalists on flights to oiled beaches. State officials cite an October survey that showed 117 miles of shoreline remained moderately or heavily oiled with oil more than two feet deep in some spots. They say observers flying over the sound still report 15 to 20 oil sheens bleeding off beaches daily. Exxon officials meanwhile say their winter monitoring of 64 sites shows wind and waves have scoured away on average more than half the surface oil left in September and up to 80 percent of the buried oil. From a laymans point of view whats left out there is really insignificant said Exxon scientist Andy Teal.'
s2 = 'A year after the wreck of the Exxon Valdez the Alaskan environmental authorities report that hundreds of gallons of oil are locked up under hundreds of beaches all over Prince William Sound. The worst is over but assessing the continuing damage has just begun. When Exxons 2 billion dollar cleanup ended in midSeptember it had recovered only 5 to 9 percent of the oil spilled. Thousands of animals and birds have died. The skipper of the Exxon Valdez is scheduled for trial and a federal grand jury issue criminal indictments against Exxon which already faces more than 150 civil lawsuits.'
model.predict([(s1, s2)])


# In[ ]:


#exact difference
s3 = 'Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.'
s4 = 'ELMo is a novel way to represent words in vectors or embeddings. These word embeddings are helpful in achieving state-of-the-art (SOTA) results in several NLP tasks: NLP scientists globally have started using ELMo for various NLP tasks, both in research as well as the industry'
model.predict([(s3, s4)])


# In[ ]:


#random
s5 = 'TORTA DI NOCI CIOCCOLATA E ZENZERO or WALNUT CAKE WITH CHOCOLATE AND CANDIED GINGER 3 eggs 250300 ml1125 cup brown sugar 100 g35 oz butter 150 g53 oz chopped walnuts 100 g35 oz chopped dark chocolate 3 tsp finely chopped candied ginger you can add more if you want I just wanted it as a background flavour 300 ml125 cup flour Mix egg and sugar in a bowl while you are melting the butter on low heat.'
s6 = 'TORTA NOCI AND CANDIED eggs 250300 ml 1125 millenary 35 g 53 walnuts 100 g 35 tsp finely sugarcoat ginger you add more if you want I wanted it as ml 125 flourMix testis while you are melting.' 
model.predict([(s5, s6)])


# In[ ]:


#translation
s7 = 'Cloutier.  Over the years, he has finished in the money in over 30 WSOP poker game events.  He has also won five first place WSOP gold bracelets in various events.  In the main Championship event of Texas Holdem, he has finished twice in second-place, and once each in third place and fifth place. Billy Baxter, one of the other big names in poker, dominated the deuce-to-seven draw poker WSOP tournament for many years, with five first place finishes and two second place finishes. Many other household poker names have graced the World Series of Poker over the years.  Names such as Mickey Appleman, Bobby Baldwin, Doyle Brunson, Johnny Chan, Johnny Moss, and "Amarillo Slim" Preston.'
s8 = 'Clotiia. Over the years, he completed 30 more than money in the event of the World Series of Poker is a poker game. He also won first place five World Series of Poker bracelet with different events. Poker Championships main event, 2 room and once in every Tuesday and Thursday over the occasion. The Billy Baxter poker, seven of the World Series of Poker champion to many years in one of the big name draw poker other hell ruled with five first place finishes and two gmrim Instead of two. World Series of Poker, and over the years many other household wreath. The Miki aplmn, Bobby Baldwin, Doyle total, Johnny Chan Johnny moss, Amarillo Slim Name "and" Preston Love.'
model.predict([(s7, s8)])


# In[ ]:


#no obfuscation
s9= 'Map of . 1683. French version Peru Map of . 1683. Das K nigreich Perou Peru Historical map of by . Ca. 1720 Peru Herman Moll Historical map of by Emanuel Bowen. Ca. 1750 Peru Historical map of by Benard. Ca. 1750 Peru Historical map of by Rigobert Bonne. Ca. 1780 Peru Historical map of South America by Diogo Homem. Ca 1558 Leo Belgicus - map of the Low Countries (1611) Historical map of the Bay of Baj (1888)'
s10 = 'Map of . 1683. French version Peru Map of . 1683. Das K nigreich Perou Peru Historical map of by . Ca. 1720 Peru Herman Moll Historical map of by Emanuel Bowen. Ca. 1750 Peru Historical map of by Benard. Ca. 1750 Peru Historical map of by Rigobert Bonne. Ca. 1780 Peru Historical map of South America by Diogo Homem. Ca 1558 Leo Belgicus - map of the Low Countries (1611) Historical map of the Bay of Baj (1888)'
model.predict([(s9, s10)])

