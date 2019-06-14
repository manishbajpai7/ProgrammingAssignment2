import numpy as np
import os
os.chdir(r"C:\\Users\\ND56397\\Desktop\\Incident Analytics")


#reading local files
import pandas as pd
train = pd.read_csv('subclause.csv',encoding = "ISO-8859-1")

#word count
#train['word_count'] = train['subclause'].apply(lambda x: len(str(x).split(" ")))
#train[['subclause','word_count']].head()

#character count
#train['char_count'] = train['subclause'].str.len() 
#train[['subclause','char_count']].head()

#average word length
#def avg_word(sentence):
  #words = sentence.split()
  #return (sum(len(word) for word in words)/len(words))

#train['avg_word'] = train['subclause'].apply(lambda x: avg_word(x))
#train[['subclause','avg_word']].head()


#Number of stopwords
#from nltk.corpus import stopwords
#stop = stopwords.words('english')

#train['stopwords'] = train['subclause'].apply(lambda x: len([x for x in x.split() if x in stop]))
#train[['subclause','stopwords']].head()


#special Characters
#train['hastags'] = train['subclause'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
#train[['subclause','hastags']].head()


#numerics
#train['numerics'] = train['subclause'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#train[['subclause','numerics']].head()


#uppercase words
#train['upper'] = train['subclause'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
#train[['subclause','upper']].head()


#lowercase conversion
train['subclause'] = train['subclause'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['subclause'].head()



#remove punctuation
train['subclause'] = train['subclause'].str.replace('[^\w\s]','')
train['subclause'].head()


#remove stopwords
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['subclause'] = train['subclause'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


#Stemming
#from nltk.stem import PorterStemmer
#st = PorterStemmer()
#train['subclause'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


#Lemmatization
#from textblob import Word
#train['subclause'] = train['subclause'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#train['subclause'].head()


#Lexical Diversity 
#len(set(train))/len(train)


#ContentFraction
#def content_fraction(csv_f):
 #   stopwords = nltk.corpus.stopwords.words('english')
  #  content = [w for w in train if w.lower() not in stopwords]
   # return len(content) / len(train)
#content_fraction(nltk.corpus.reuters.words())

#TFIDF
tf1 = (train['subclause'][1:3]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['subclause'].str.contains(word)])))
tf1

tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1



#remove special characters
from nltk import re
train['subclause']=re.sub('[^A-Za-z0-9]', ' ', str(trai     n['subclause']))


#tokenize 
#train['subclause'].dropna(inplace=True)
#train['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['subclause']), axis=1)
#train['tokenized_sents'].head()


#POS Tagging
from nltk import word_tokenize, pos_tag
from functools import partial
tok_and_tag = lambda x: pos_tag(word_tokenize(x))
train['Tag'] = train['subclause'].apply(tok_and_tag)

#Extracting POS tagged words
#train['Tagged'] = train['subclause'].apply(lambda x: ' '.join(TextBlob(x).noun_phrases))
#df = pd.DataFrame()
#df['Tag'] = [('unclear', 'JJ'), ('incomplete', 'JJ'), ('instruction', 'NN'), ('given', 'VBN')]
#df['words'] = [i[0] for i in df['Tag']]
#df['tags'] = [i[1] for i in df['Tag']]
#df['Tagged2'] = np.where(df['tags']=='NN', df['words'], np.nan)
#df.drop(['words','tags'],1,inplace=True)
#print(df)

#Extracting POS Words#
train['Tagged3']= train['subclause'].apply(lambda x:' '.join([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(x)) if pos[0] == 'NN']))



###########Most common words#################  
#from nltk import FreqDist
#fdist1 = FreqDist(csv_f)
#print(fdist1)
#for tokens in sorted(fdist1):
 #   print(tokens, '->', fdist1[tokens], end='; ') 
#fdist1.most_common(5)    


#clustering#
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents = train['Tagged3']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
train.to_csv('C:\\Users\\ND56397\\Desktop\\Incident Analytics\\resultpoccommunication.csv')
