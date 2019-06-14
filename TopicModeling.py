# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:34:11 2019

@author: ND68005
"""

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
#! pip install gensim
#!pip install pyLDAvis
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
os.chdir(r"C:/Users/ND68005/Desktop")
os.getcwd()
import pandas as pd 
df = pd.read_csv("Book1.csv", sep='\t', engine='python')

# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


freq_words(df['Incident Description'])

fdist = FreqDist(' '.join([text for text in df["Incident Description"]]).split())
#print(fdist)

words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

# selecting top 20 most frequent words
d = words_df.nlargest(columns="count", n = 30) 
plt.figure(figsize=(20,5))
ax = sns.barplot(data=d, x= "word", y = "count")
ax.set(ylabel = 'Count')
plt.show()

# remove unwanted characters, numbers and symbols
df['Incident Description'] = df['Incident Description'].str.replace("[^a-zA-Z#]", " ")
# FRemove stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

# remove short words (length < 3)
df['Incident Description'] = df['Incident Description'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i.lower() not in stop_words])
    return rev_new
  
# remove stopwords from the text
df['reviews'] = df['Incident Description'].apply(lambda x: ''.join([w for w in remove_stopwords(x.split())]))
#make entire text lowercase and exclude short words
df['reviews'] = df['reviews'].apply(lambda x: ''.join([w for w in x.lower()]))

#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

# Lemmatize with POS Tag
import nltk
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


#Lexicon Normalization
#performing stemming and Lemmatization
nltk.download('punkt')
# Tokenized reviews
nltk.download('averaged_perceptron_tagger')
df['reviews1'] = df['reviews'].apply(lambda x: ([w for w in word_tokenize(x)]))
reviews2=[]
for sent in df['reviews1']:
    reviews2.append(' '.join([lem.lemmatize(w,get_wordnet_pos(w)) for w in sent]))

reviews_2=[]
for sent in df['reviews1']:
    reviews_2.append([lem.lemmatize(w,get_wordnet_pos(w)) for w in sent])    

df['reviews2'] = reviews2
freq_words(df['reviews2'], 35)

#Building an LDA model

#We will start by creating the term dictionary of our corpus, where every unique term is assigned an index

dictionary = corpora.Dictionary(reviews_2)

# Create document term matrix using the dictionary

doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)
lda_model.print_topics()

#Topics Visualization
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.show(vis)
pyLDAvis.save_html(vis,manish.html)
