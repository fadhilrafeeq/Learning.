##LDA Implementation.

import numpy as np
import pandas as pd
from  nltk.tokenize import RegexpTokenizer
import re
import nltk
from scipy import sparse
from scipy.sparse import csc_matrix
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
import string
from sklearn.feature_extraction import text
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random

random.seed(100)

from sklearn.datasets import fetch_20newsgroups
inputData = fetch_20newsgroups(subset = 'all')['data']
type(inputData)

textData = pd.DataFrame(inputData[:1000])
textData.columns = ['tweet']
nltk.download('stopwords')

stop_words_list = list(stopwords.words('english'))
stop = set(stop_words_list)

lemma = WordNetLemmatizer()

stemmer = PorterStemmer()

def preprocessing_Tweets(x):
    document  = re.sub(r'\W',' ', x)
    document = re.sub(r'[0-9]',' ',document)
    document = re.sub(r'_',' ',document)
    document = re.sub(r'\s+[a-zA-z]\s+',' ',document)
    document = re.sub(r'\s+',' ',document, flags = re.I)
    document = document.lower()
    document = document.split()
    document = " ".join([i for i in document if i not in stop])
    
    return document

columns = ["number","tweet"]
dataClean = pd.DataFrame(columns=columns)

for tweet in range(textData.shape[0]):
    temp = preprocessing_Tweets(textData.iloc[tweet,0])
    temp = re.sub(r'[^\x00-\x7F]+',' ',temp)
    dataClean.loc[tweet,"number"] = tweet
    dataClean.loc[tweet,"tweet"] = temp
    
tfVec = CountVectorizer(stop_words= stop_words_list)

train_data = tfVec.fit_transform(dataClean.iloc[:,-1])
docwords = train_data.todense()
tf_feat_names = tfVec.get_feature_names_out()

num_components = 10
model = LatentDirichletAllocation(n_components=num_components)

lda_matrix = model.fit_transform(train_data)

lda_components = model.components_
terms = tf_feat_names

for index, component in enumerate(lda_components):
    zipped = zip(terms, component)
    top_terms_key = sorted(zipped, key = lambda t : t[1], reverse=True)[:7]
    top_terms_list = list(dict(top_terms_key).keys())
    print("Topic"+str(index)+": ",top_terms_list)
    
    
######################################################################

import pyLDAvis.gensim
from gensim.models.ldamodel import LdaModel
import gensim.corpora  as corpora
from gensim.models.coherencemodel import CoherenceModel

textData = dataClean.iloc[:,1]

text_tokens = [[text for text in doc.split()] for doc in list(textData)]
dict_los = corpora.Dictionary(text_tokens)
print(dict_los.token2id)

corpus = [dict_los.doc2bow(doc) for doc in text_tokens]
print("Number of Unique Tokes : %d " %len(dict_los))
print("Number of Docs %d" %len(corpus))

no_of_topics = 6
chunksize = 500
passes = 20 
iters = 400 
eval_every = 1

temp = dict_los[0]
id2word = dict_los.id2token

ldamodel = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha = 'auto', eta='auto', iterations=iters, num_topics = no_of_topics, passes = passes, eval_every=eval_every)
print(ldamodel.print_topics())

print("perplexity: ", ldamodel.log_perplexity(corpus))

coh_model_lda = CoherenceModel(
    model = ldamodel,
    texts = text_tokens,
    dictionary=dict_los,
    coherence='u_mass')

coh_model_lda_val = coh_model_lda.get_coherence()

print("Coherence Score: ", coh_model_lda_val)

pyLDAvis.enable_notebook()
LDavis_prep = prepare(ldamodel, corpus, dict_los, n_jobs =1, R = 20, sort_topics=True)

pyLDAvis.save_html(LDavis_prep, 'C:\\Users\\fadhi\\OneDrive\\Desktop\\LDA.html')
pyLDAvis.display(LDavis_prep)

from pyLDAvis.gensim_models import prepare
