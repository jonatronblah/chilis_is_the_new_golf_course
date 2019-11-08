import sklearn
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os



nltk.download('stopwords')	
#stop = stopwords.words('english')


#function to remove stop words, punctuation, lemmetize words
#pos set to verb - try with different settings?
lemmatizer = WordNetLemmatizer()

def text_process(text):
    nopunct = [char for char in text if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    a = ''
    i = 0
    for i in range(len(nopunct.split())):
        b = lemmatizer.lemmatize(nopunct.split()[i], pos='v')
        a = a+b+' '
    return ' '.join([word for word in a.split() if word not in stopwords.words('english')])


#import raw data
df = pd.read_csv('path\to\chilis_is_the_new_golf_course\office_quotes_raw.csv')

#removed colons from char names
df['character'] = df['character'].str.replace('[^a-zA-Z]', '')
#removed actions in brackets
df['line'] = df['line'].str.replace('\[.*?\]', '')

#subset characters
characters = ['Michael', 'Dwight', 'Jim', 'Pam', 'Kelly'] 

#used 4 characters for model accuracy sanity check
#characters = ['Michael', 'Dwight', 'Jim', 'Pam'] 

df = df[df['character'].isin(characters)]

#subset phrases over 5 words
df = df[df['line'].str.split().str.len() > 5]

#generate random samples n=200 per class for model evaluation
'''
size = 200        # sample size
replace = True  # with replacement
np.random.seed(seed=123456)
fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
df = df.groupby('character', as_index=False).apply(fn)
'''

#train/test split, transform, vectorize
df['processed'] = df['line'].apply(lambda x: text_process(x))
X = df['processed']
y = df['character']
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=5)

#dump testing data for flask app
#pickle.dump(X_test, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\X_test.sav', 'wb'))

#pickle.dump(y_test, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\y_test.sav', 'wb'))

#dump training data for flask app
#pickle.dump(X_train, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\X_train.sav', 'wb'))

#pickle.dump(y_train, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\y_train.sav', 'wb'))


#bow pipeline

bow_transformer=CountVectorizer().fit(X_train)
text_bow_train=bow_transformer.transform(X_train)
text_bow_test=bow_transformer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(text_bow_train, y_train)

office_pipe = make_pipeline(bow_transformer, classifier)

office_pipe.score(X_train, y_train)

office_pipe.score(X_test, y_test)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\multiNB_bow.sav', 'wb'))

classifier = ComplementNB()
classifier.fit(text_bow_train, y_train)

office_pipe = make_pipeline(bow_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\compNB_bow.sav', 'wb'))

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(text_bow_train, y_train)

office_pipe = make_pipeline(bow_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\rfc_bow.sav', 'wb'))

classifier = SVC(probability=True)
classifier.fit(text_bow_train, y_train)

office_pipe = make_pipeline(bow_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\svc_bow.sav', 'wb'))



#tfidf pipeline

tfidf_transformer=TfidfVectorizer().fit(X_train)
text_tfidf_train=tfidf_transformer.transform(X_train)
text_tfidf_test=tfidf_transformer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(text_tfidf_train, y_train)

office_pipe = make_pipeline(tfidf_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\multiNB_tfidf.sav', 'wb'))

classifier = ComplementNB()
classifier.fit(text_tfidf_train, y_train)

office_pipe = make_pipeline(tfidf_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\compNB_tfidf.sav', 'wb'))

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(text_tfidf_train, y_train)

office_pipe = make_pipeline(tfidf_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\rfc_tfidf.sav', 'wb'))

classifier = SVC(probability=True)
classifier.fit(text_tfidf_train, y_train)

office_pipe = make_pipeline(tfidf_transformer, classifier)

pickle.dump(office_pipe, open(r'C:\Users\jonathan\Desktop\DAPT Docs\fall 2019\text mining\group project\svc_tfidf.sav', 'wb'))







