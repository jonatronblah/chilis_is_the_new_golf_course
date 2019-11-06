from flask import render_template, request, redirect
from werkzeug.utils import secure_filename
from app import app
from app.forms import ModelForm
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline
import os


lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')	
home = app.instance_path

X_test = pickle.load(open(home + "/testing_data/X_test.sav", 'rb'))
y_test = pickle.load(open(home + "/testing_data/y_test.sav", 'rb'))

def text_process(text):
    nopunct = [char for char in text if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    a = ''
    i = 0
    for i in range(len(nopunct.split())):
        b = lemmatizer.lemmatize(nopunct.split()[i], pos='v')
        a = a+b+' '
    return ' '.join([word for word in a.split() if word not in stopwords.words('english')])



@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = ModelForm()
    #if form.validate_on_submit():
    if request.method == 'POST':
        quote = form.quote.data
        quote = [text_process(quote)]
        
        
        
        m = form.model.data
        m_filename = secure_filename(m.filename)
        m.save(home + "/model.sav")
        pipe = pickle.load(open(home + "/model.sav", 'rb'))
        
        #prepare quote data for pipeline
        pred = pipe.predict(quote)[0]
        proba = [str(i) for i in pipe.predict_proba(quote)[0]]
        
        classes = [str(i) for i in pipe.classes_]
        
        items = []
        for e, i in enumerate(classes):
            an_item = dict(name=i, probability=proba[e])
            items.append(an_item)
        score = pipe.score(X_test, y_test)
        
        pipe_detail = [pipe.named_steps[i] for i in pipe.named_steps.keys()]
        
            
        
      
        
        
        
        
        
        return render_template('predict.html', items=items, score=score, pipe_detail=pipe_detail, form=form)
       





    return render_template('base.html', form=form)
