

import mysql.connector
from flask import Flask, render_template, request, url_for, flash, redirect
# from regform import RegistrationForm
from flask_mysqldb import MySQL
from flask_wtf import FlaskForm
#from pyreadline import execfile
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo
import string
from collections import Counter


import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'



class tp(FlaskForm):
    username = StringField('username', validators=[DataRequired(), Length(min=2, max=20)])
    submit = SubmitField('submit')


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sem6project"
)

app = Flask(__name__)
obj = MySQL(app)
app.config['SECRET_KEY'] = '123456'

@app.route('/', methods=['GET', 'POST'])
def home2():
    return render_template('home.html', title='Home')

@app.route('/sentiment', methods=['GET', 'POST'])
def home():
    forms = tp()
    #if request.method == "POST":
        #username = request.form['username']
        # print(username)
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM comments")
    myresult = mycursor.fetchall()
        # username = input(" enter username ")
    
    

    for x in myresult:
        file = open('read2.txt', 'w')
        #for y in x:
        #print("y=",x[0])
        username = x[0]
        y = x[1]
        
        file.write(y)

        file.close()

        text = open('read2.txt', encoding='utf-8').read()
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        # Using word_tokenize because it's faster than split()
        tokenized_words = word_tokenize(cleaned_text, "english")

        # Removing Stop Words
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                final_words.append(word)

        # Lemmatization - From plural to single + Base form of a word (example better-> good)
        lemma_words = []
        for word in final_words:
            word = WordNetLemmatizer().lemmatize(word)
            lemma_words.append(word)

        emotion_list = []
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')

                if word in lemma_words:
                    emotion_list.append(emotion)

        #print(emotion_list)
        w = Counter(emotion_list)
        #print(w)
        sentiment_analyse(cleaned_text,username)

        fig, ax1 = plt.subplots()
        ax1.bar(w.keys(), w.values())
        fig.autofmt_xdate()
        plt.savefig('static/img/graph.jpg')
    
    # mycur = mydb.cursor()
    
    # sql = "select * from comments where sentiment=%s"
    # sentiment = ("neutral",)
    # mycur.execute(sql,sentiment)
   
    
    
    # myresults = mycur.fetchall()
    # for i in myresults:
    #     print("working hai")
    mycur = mydb.cursor()
    sql = "select * from comments where sentiment = %s"
    sentiment = ("negative",)
    mycur.execute(sql,sentiment)
    myres = mycur.fetchall()
    
    return render_template('dash.html',data=myres)



@app.route('/delete/<string:id_data>', methods = ['GET'])
def delete(id_data):
    flash("Record Has Been Deleted Successfully")
    mycur = mydb.cursor()
    mycur.execute("DELETE FROM comments WHERE id=%s", (id_data,))
    mydb.commit()
    return redirect(url_for('home'))

@app.route('/deletepost/<string:id_data>', methods = ['GET'])
def deletepost(id_data):
    flash("Record Has Been Deleted Successfully")
    mycur = mydb.cursor()
    mycur.execute("DELETE FROM comments WHERE post_id=%s", (id_data,))
    mydb.commit()
    mycur = mydb.cursor()
    mycur.execute("DELETE FROM post WHERE id=%s", (id_data,))
    mydb.commit()

    return redirect(url_for('posts'))

@app.route('/posts', methods=['GET', 'POST'])
def posts():
    forms = tp()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM post")
    myresult = mycursor.fetchall()
    
    
    for x in myresult:
        file = open('read5.txt', 'w')
        #for y in x:
        #print("y=",x[0])
        image = x[4]
        img = cv2.imread(image)
        text = pytesseract.image_to_string(img)
        mycursor = mydb.cursor()

        sql = "UPDATE post SET content = %s WHERE pic = %s"
        val = (text, image)

        mycursor.execute(sql, val)

        mydb.commit()
        username = x[0]
        y = x[3]
        
        file.write(y)

        file.close()

        text = open('read5.txt', encoding='latin-1').read()
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        # Using word_tokenize because it's faster than split()
        tokenized_words = word_tokenize(cleaned_text, "english")

        # Removing Stop Words
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                final_words.append(word)

        # Lemmatization - From plural to single + Base form of a word (example better-> good)
        lemma_words = []
        for word in final_words:
            word = WordNetLemmatizer().lemmatize(word)
            lemma_words.append(word)

        emotion_list = []
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')

                if word in lemma_words:
                    emotion_list.append(emotion)

        #print(emotion_list)
        w = Counter(emotion_list)
        #print(w)
        sentiment_analyse1(cleaned_text,username)
    mycur = mydb.cursor()
    sql = "select * from post where sentiment = %s"
    sentiment = ("negative",)
    mycur.execute(sql,sentiment)
    myres = mycur.fetchall()
    
    return render_template('dashpost.html',data=myres)




@app.route('/analysis',methods=['GET','POST'])
def analysis():
    forms = tp()
    #if request.method == "POST":
    #username = request.form['username']
    #print(username)
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM comments")
    myresult = mycursor.fetchall()
        #username = input(" enter username ")
    #print(myresult)
    file = open('read3.txt', 'w')
    for x in myresult:

        y = x[1]
        print(y)
        file.write(y)


    file.close()

    text = open('read3.txt', encoding='utf-8').read()
    lower_case = text.lower()
    cleaned_text2 = lower_case.translate(str.maketrans('', '', string.punctuation))

    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text2, "english")

    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)

    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in lemma_words:
                emotion_list.append(emotion)

    #print(emotion_list)
    w = Counter(emotion_list)
    #print(w)
    sentiment_analyse2(cleaned_text2)

    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    plt.savefig('static/img/graph2.jpg')




    return render_template('Dashboard.html', title='Home')






def sentiment_analyse(sentiment_text,username):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)



    if score['neg'] > score['pos']:
        sentiment = "negative"

        mycursor = mydb.cursor()

        sql = "UPDATE comments SET sentiment = %s WHERE id = %s"
        val = (sentiment, username)

        mycursor.execute(sql, val)

        mydb.commit()

        
    elif score['neg'] < score['pos']:
        sentiment = "positive"

        # mycursor.execute("INSERT INTO ml values")
        mycursor = mydb.cursor()

        sql = "UPDATE comments SET sentiment = %s WHERE id = %s"
        val = (sentiment, username)

        mycursor.execute(sql, val)

        mydb.commit()

        
    else:
        sentiment = "neutral"
        mycursor = mydb.cursor()

        sql = "UPDATE comments SET sentiment = %s WHERE id = %s"
        val = (sentiment, username)

        mycursor.execute(sql, val)

        mydb.commit()

def sentiment_analyse1(sentiment_text,username):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)



    if score['neg'] > score['pos']:
        sentiment = "negative"

        mycursor = mydb.cursor()

        sql = "UPDATE post SET sentiment = %s WHERE id = %s"
        val = (sentiment, username)

        mycursor.execute(sql, val)

        mydb.commit()

        
    elif score['neg'] < score['pos']:
        sentiment = "positive"

        # mycursor.execute("INSERT INTO ml values")
        mycursor = mydb.cursor()

        sql = "UPDATE post SET sentiment = %s WHERE id = %s"
        val = (sentiment, username)

        mycursor.execute(sql, val)

        mydb.commit()

        
    else:
        sentiment = "neutral"
        mycursor = mydb.cursor()

        sql = "UPDATE post SET sentiment = %s WHERE id = %s"
        val = (sentiment, username)

        mycursor.execute(sql, val)

        mydb.commit()

        


def sentiment_analyse2(sentiment_text2):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text2)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

        # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

        # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
        # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
            # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


@app.route('/algo', methods=['GET', 'POST'])
def algo():
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM post")
    myresult = mycursor.fetchall()
    file = open('read4.txt', 'w')
    for x in myresult:

        y = x[3]
        #print(y)
        file.write(y)
    file.close()
    # Loading Label encoder
    labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))

    # Loading TF-IDF Vectorizer
    Tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))

    # Loading models
    SVM = pickle.load(open('svm_trained_model.sav', 'rb'))
    #print(SVM)
    Naive = pickle.load(open('nb_trained_model.sav', 'rb'))

    # Text from social media app --->
    sample_text = str(open('read4.txt', 'r'))
    sample_text_processed = text_preprocessing(sample_text)
    sample_text_processed_vectorized = Tfidf_vect.transform([sample_text_processed])
    #print(sample_text_processed_vectorized)

    Encoder = LabelEncoder()

    # Prediction --->

    prediction_SVM = SVM.predict(sample_text_processed_vectorized)
    prediction_Naive = Naive.predict(sample_text_processed_vectorized)

    #print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM)[0])

    #print("Prediction from NB Model:", labelencode.inverse_transform(prediction_Naive)[0])

    a = labelencode.inverse_transform(prediction_SVM)[0]
    if a== "__label__1":
        data = "Not self harm"
    else :
        data = "self harm"
    b= labelencode.inverse_transform(prediction_Naive)[0]
    if b == "__label__2":
        data2 = "self harm"
    else:
        data2 = "not a self harm"

    return render_template('algo.html', title='Home',data=data,data2=data2)



if __name__ == '__main__':
    app.run(debug=True)