import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline

import os

import sqlite3

app = Flask(__name__)


@app.route('/input/<name>')
def display_name(name):
	return render_template('result2.html', output = name)


def get_prediction_proba(docx):
        df = pd.read_csv('dataset.csv', names=['text', 'emotion'])
        df['Clean_Text']=df['text'].apply(nfx.remove_userhandles)
        df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
        Xfeatures =df['Clean_Text']
        Ylabels=df['emotion']

        x_train,x_test,y_train,y_test = train_test_split(Xfeatures,Ylabels,test_size=0.3,random_state=42)

        pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(solver='lbfgs', max_iter=200))])
        pipe_lr.fit(x_train,y_train)
        pipe_nb = Pipeline(steps=[('cv',CountVectorizer()),('nb',naive_bayes.MultinomialNB(alpha=0.3))])
        pipe_nb.fit(x_train,y_train)
        n_estimators = 10
        pipe_svm = Pipeline(steps=[('cv',CountVectorizer()),('svm',BaggingClassifier(svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True),max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=-1))])
        pipe_svm.fit(x_train,y_train)
        pipe_rf = Pipeline(steps=[('cv',CountVectorizer()),('rf',RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,random_state=42))])
        pipe_rf.fit(x_train,y_train)

        emotion_list = ['anger','fear','joy','love','sadness','suprise']
        emoji = {'joy':'😀','sadness':'😔','anger':'😠','fear':'😨','love':'🥰','suprise':'😲'}

        result_lr = pipe_lr.predict_proba([docx])
        result_nb = pipe_nb.predict_proba([docx])
        result_svm = pipe_svm.predict_proba([docx])
        result_rf = pipe_rf.predict_proba([docx])

        max_lr = np.max(pipe_lr.predict_proba([docx]))
        max_nb = np.max(pipe_nb.predict_proba([docx]))
        max_svm = np.max(pipe_svm.predict_proba([docx]))
        max_rf = np.max(pipe_rf.predict_proba([docx]))
        if(max_lr>max_nb and max_lr>max_svm and max_lr>max_rf):
                return [result_lr,pipe_lr.predict([docx])[0]]
        elif(max_nb>max_lr and max_nb>max_svm and max_nb>max_rf):
                return [result_nb,pipe_nb.predict([docx])[0]]
        elif(max_svm>max_lr and max_svm>max_nb and max_svm>max_rf):
                return [result_svm,pipe_svm.predict([docx])[0]]
        else:
                return [result_rf,pipe_rf.predict([docx])[0]]


@app.route('/TEXT_EMOTION_DETECTOR' , methods = ['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        f = open("database.txt","a")
        user = request.form['text']
        f.write("Text ->  "+user+"\n")
        ex1 = user
        probability = get_prediction_proba(ex1)
        predicted_emotion = probability[1]
        probability = probability[0][0]
        for i in range(len(probability)):
            probability[i]=probability[i]*100
        emotion_list = ['anger','fear','joy','love','sadness','suprise']
        savefile='static/graph.png'
        if os.path.exists(savefile):
            os.remove(savefile)
            plt.clf()
        plt.bar([0,10,20,30,40,50],probability,tick_label = emotion_list,width=5,color=['red','green','yellow','hotpink','lightgrey','orange'])
        plt.xlabel('Emotions')
        plt.ylabel('Confidence Level (%)')
        plt.title('Predicted Probability')
        plt.savefig(savefile)
        confidence = np.max(probability)
        f.write("Predicted emotion ->  "+str(predicted_emotion)+"\n")
        f.write("Confidence rate -> "+ str(confidence)+"\n\n")
        sentence = str(predicted_emotion)+" With confidence: {:.2f}".format(confidence)+"%"
        f.close()
        return redirect(url_for('display_name',name = sentence))
        

@app.route('/moreinfo')
def sav():
	return(""" <head>
        <link rel="icon" type="image/x-icon" href="logo.jpeg" />
    <title>MORE INFO</title>
<style>

body{
     background-color: #0892d0; //rgb(2, 26, 58);
    }

.top{
color : rgb(136, 153, 230);
text-align: center;
text-shadow: 5px 5px 10px burlywood;
}

.h1{
    background-color: yellow;
    color:black;
    border-color : black;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
}

.h3{
    background-color: black;
    color:white;
    border-color : yellow;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
}
.h5{
    background-color: yellow;
    color:black;
    border-color : black;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
}

.h6{
    background-color: black;
    color:white;
    border-color : yellow;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
}

.h7{
    background-color: yellow;
    color:black;
    border-color : black;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
}

.h8{
   background-color: black;
    color:white;
    border-color : yellow;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
}

.h9{
    background-color: yellow;
    color:black;
    border-color : black;
    padding-top: 4px;
    padding-bottom: 4px;
    padding-left: 40px;
    border-style: solid;
    border-radius: 70px;
    border-width: 5px;
    border-color: black;
}
</style>
</head>
<body>
    <div class = "top">
		<H1 style="color:yellow"><b>**** MORE INFO ****</b></H1>
    </div>

    <div class="h1">
        <h2 style="text-align: center;text-decoration: underline;"><b>Dataset information</b></h2>
        <p id="h2">
            <h3> <li>Dataset contains text messages with six basic emotions such as: anger, fear, joy, love, sadness, and surprise. The dataset contains a total of 20,000 instances and 2 features (text and emotion). Out of 20,000, nearly 70% (14,000) of the dataset is used for training purpose and the rest 30% (6,000) is used for testing purpose.<br><br>
	<li>As a part of the data preprocessing process, the dataset is cleaned by removing the unwanted words, punctuations, etc. </li>
	<br><br><li>Dataset Link: -<a href=" https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp">https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp</a></h3></li>
    </p>
    </div>
    <br><br>

    <div class="h3">
        <h2 style="text-align:center;text-decoration: underline;"><b>Algorithms</b></h2>
        <p id="h4">
            <h3>
                <li>For every machine learning algorithm, it has its own advantages and disadvantages over the other algorithms, especially for a classification problem, each algorithm behaves differently. To get the best of the machine learning problem, implementation of more than one ML algorithm can come in handy in many cases.
	<br><br><li>Similarly, the project incorporates 4 different classification machine learning algorithms, which are:
                 <br><br><li>Logistic Regression</li>
                 <li>Naïve Bayes Classifier</li>
                 <li>Support Vector Machine (SVM)</li>
                 <li>Random Forest Classifier</li>

            </h3>
        </p>
    </div><br><br>

    <div class="h5">
        <h2 style="text-align: center;text-decoration: underline;"><b>Logistic Regression</b></h2>
        <h3>
            <li>Logistic Regression is one of the most popular machine learning algorithms, which comes under the supervised learning technique. It is mainly used to predict the output of categorical dependent variable with the help of independent variable. The intention behind using logistic regression is to find the best fitting model to describe the relationship between the dependent and the independent variable.
	<br><br><li>In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values. The curve from the logistic function indicates the likelihood of something such as whether the emotion is anger, fear, joy, love, sadness, or surprise. Logistic regression is a significant machine learning algorithm because it has the ability to provide probabilities and classify new data using continuous and discrete datasets.
        </h3>

    </div>
    <br><br>

    <div class="h6">
        <h2 style="text-align: center;text-decoration: underline;"><b>Navie Bayes Classifier</b></h2>

        <h3>
            <li>Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e., every pair of features being classified is independent of each other. 
<br><br><li>Basically, the Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. The advantages of this algorithm are that, it is easy and fast to predict class of test dataset and also performs well in multi-class prediction. When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data. When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.

        </h3>
    </div>
    <br><br>

    <div class="h7">
        <h2 style="text-align: center;text-decoration: underline;"><b>Support Vector Machine(SVM)</b></h2>
      <h3>
        <li>Support Vector Machine (SVM) is a supervised learning algorithm, which is primarily, used for classification problem. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
        <br><br><li>SVM algorithm is most commonly used in emotion detection problems.
    
      </h3>
    </div>
    <br><br>
    <div class="h8">
        <h2 style="text-align: center;text-decoration: underline;"><b>Random Forest Classifier</b></h2>

        <h3>
            <li>Random Forest Classifier algorithm is a supervised machine learning algorithm which is based on the concept of ensemble learning, which is the process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
	<br><br><li>Random forest classifier contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output. 
	<br><br><li>Random forest classifier is also a common algorithm for emotion detection since it takes less training time as compared to other algorithms, like SVM. It predicts output with high accuracy, even for the large dataset and it runs efficiently.

        </h3>

    </div>
    <br><br>
    <div class="h9">
    <h2 style="text-align: center;text-decoration: underline;"><b>Accuracy</b></h2>

    <h3 style="text-align: center;">
       <li> Logistic Regression - 88.85%</li>
        <li>Naïve Bayes Classifier - 80.46%</li>
        <li>Support Vector Machine (SVM) - 85.51%</li>
        <li>Random Forest Classifier - 87.31%</li>

    </h3>
    </div> 
    
</body>
""")

@app.route('/details')
def graph():
	return("""
		<head>
        <link rel="icon" type="image/x-icon" href="logo.jpeg" />
	<title>DETAILS ABOUT PREDICTED EMOTION</title>
	<style type="text/css">

		body{
			background-color : rgb(255,255,0);
			padding-right: 100px;
			padding-left: 100px;
			
			background-size: 100%;
			font-size : 30px;
		}

		h1{
			background-color: black;
			color :white;
			text-align: center;
			border-radius: 20px;
			padding-bottom: 10px;
		}
		.image{
			padding-left:27%;
            padding-bottom:50px;
            padding-top:50px;
            
            background-color : blue;
            border : solid;
		}
        img{
        border:solid;
        }
        .outer{
        padding-left:100px;
        padding-right:150px;
        }


	</style>
</head>
<body>

     <h1 id="find">
     	DETAILS OF PREDICTED EMOTION
     </h1>

     <div class="outer">

      <div class="image">

        <div id="image">
     	   <img src="../static/graph.png" alt="graph" align="center" style="align-content: center;">
        </div>

      </div>
     </div>
</body>
""")


@app.route('/getrecom')
def recon():
	return("""
    <head>
    <link rel="icon" type="image/x-icon" href="logo.jpeg" />
    <title>Recommendation page</title>
    <style>
    
    body{
    
    background-color: rgb(1, 1, 90);
    text-align:center;
}
.top{
	color : black;
	text-align: center;
	text-shadow: 5px 5px 10px burlywood;
}
#head{
color:white;
}


.middle{
    text-align: center;
    padding-top: 15px;
    padding-right: 50px;
}
.left{
    text-align: center;
    padding-top: 15px;
    padding-right: 50px;


}
.right{
    text-align: center;
    padding-top: 15px;
    padding-right: 50px;

}
#btn{
    color: white;
    background-color:black;
    border : solid yellow;
    padding-left: 60px;
    padding-right: 60px;


}

#btn1{
   color: white;
    background-color:black;
    border : solid yellow;
    padding-left: 60px;
    padding-right: 60px;
}

#btn2{
    color: white;
    background-color:black;
    border : solid yellow;
   
    padding-left: 60px;
    padding-right: 60px;
}
.sng{
    background-color: rgb(255,255,0);
    margin-left:20px ;
    margin-right: 20px;
    border-width: 4px;
     border-radius:90px;
     border-color: white 10px;
     border-style: solid;
}

.qte{
    background-color: rgb(255,255,0);
    margin-left:20px ;
    margin-right: 20px;
    border-width: 4px;
     border-radius: 90px;
     border-color: white 10px;
     border-style: solid;
}
.vid{
    background-color: rgb(255,255,0);
    margin-left:20px ;
    margin-right: 20px;
    border-width: 4px;
     border-radius: 90px;
     border-color: white 10px;
     border-style: solid;
}

 </style>
</head>
<body>
    <div class = "top">
		<H1 id="head"><b>**** WELCOME TO THE WORLD OF MOTIVATION ****</b></H1>
    </div>
    <br>
    <div class="sng">
        <h2>MOTIVATIONAL SONG</h2>
        <p id="sngsen">
            <b><h3> It is proven by science that motivational songs improve your focus,raise morale,and generally make you feel happier that will give you a dose of inspiration to face the day ahead<b><h3>
        </p>
    </div>

    <div class="middle">
        <button id="btn" onclick="finish1()"><h2><b>Get motivation song<b><h2></button>
        <h1 id="res1"></h1>
    </div>
    <div class="">

    </div>
    <div class="qte">
        <h2>MOTIVATIONAL QUOTE</h2>
        <p id="qtesen">
           <h3> These motivational quotes provide you with a quick and timely burst of wisdom to get focus back and offering the inspiration needed for the day or occasion <h3>
        </p>
    </div>
    <div class="right">
        <button id="btn1" onclick="finish2()"><h2><b>Get quote</h2><b></button>
        <h1 id="res2">

        </h1>
    </div>
    <div class="vid">
        <h2>MOTIVATIONAL VIDEO</h2>
        <p id="vidsen">
            <h3>purpose of a motivational video is to inspire viewers and are not meant to train peoples instead they are carefully designed to inspire people<h3>

        </p>
    </div>
    <div class="left">
        <button id="btn2" onclick="finish3()"><h2><b> Get motivation video</h2></b></button>
        <h1 id="res3"></h1>
    </div>
     <script>
     var buton=window.localStorage.getItem("answer");
if(buton=='joy'){

var butn1 = document.getElementById('btn');
var rest1 = document.getElementById('res1');
var butn2 = document.getElementById('btn1');
var rest2 = document.getElementById('res3');
var butn3 = document.getElementById('btn2');
var rest3 = document.getElementById('res2');

let q1= "CLICK HERE!";
let q11=q1.link("https://youtu.be/gdZLi9oWNZg");

let q2= "CLICK HERE!";
let q22=q2.link("https://youtu.be/m7Bc3pLyij0");

let q3= "CLICK HERE!";
let q33=q3.link("https://youtu.be/g-tNJrAgP54");

let q4= "CLICK HERE!";
let q44=q4.link("https://youtu.be/YKLX3QbKBg0");

let q5= "CLICK HERE!";
let q55=q5.link("https://youtu.be/I7fA8hdrqp8");

let q6= "CLICK HERE!";
let q66=q6.link("https://youtu.be/HgiiY9TLtX8");

let q7= "CLICK HERE!";
let q77=q7.link("https://www.instagram.com/p/CVDN00ftY2J/?utm_medium=share_sheet");

let q8= "CLICK HERE!";
let q88=q8.link("https://www.instagram.com/p/CDMMUhVjOmf/?utm_medium=copy_link");

let q9= "CLICK HERE!";
let q99=q9.link("https://www.instagram.com/tv/CVhQx9HohAt/?utm_medium=copy_link");

var sngg = [q11,q22,q33]
function finish1(){
rest1.innerHTML= sngg[Math.floor(Math.random()*sngg.length)];
}
var qute = [q44,q55,q66]
function finish3(){
rest2.innerHTML= qute[Math.floor(Math.random()*qute.length)];
}

var vido = [q77,q88,q99]
function finish2(){
rest3.innerHTML= vido[Math.floor(Math.random()*vido.length)];
}
}
else if(buton=='sadness')
{

var butn1 = document.getElementById('btn');
var rest1 = document.getElementById('res1');
var butn2 = document.getElementById('btn1');
var rest2 = document.getElementById('res3');
var butn3 = document.getElementById('btn2');
var rest3 = document.getElementById('res2');

let v1= "CLICK HERE!";
let v11=v1.link("https://youtu.be/Az-mGR-CehY");

let v2= "CLICK HERE!";
let v22=v2.link("https://youtu.be/J6enOG547lk");

let v3= "CLICK HERE!";
let v33=v3.link("https://youtu.be/RgKAFK5djSk");

let v4= "CLICK HERE!";
let v44=v4.link("https://youtu.be/wnHW6o8WMas");

let v5= "CLICK HERE!";
let v55=v5.link("https://youtu.be/k9zTr2MAFRg");

let v6= "CLICK HERE!";
let v66=v6.link("https://youtu.be/UrfpkvvRTns");

let v7= "CLICK HERE!";
let v77=v7.link("https://www.instagram.com/reel/CSUz9dpBMjB/?utm_medium=copy_link");

let v8= "CLICK HERE!";
let v88=v8.link("https://www.instagram.com/reel/CVk20gZhhTx/?utm_medium=copy_link");

let v9= "CLICK HERE!";
let v99=v9.link("https://www.instagram.com/reel/CLYeys5BXkO/?utm_medium=share_sheet");

var sngg = [v11,v22,v33]
function finish1(){
rest1.innerHTML= sngg[Math.floor(Math.random()*sngg.length)];
}
var qute = [v44,v55,v66]
function finish3(){
rest2.innerHTML= qute[Math.floor(Math.random()*qute.length)];
}

var vido = [v77,v88,v99]
function finish2(){
rest3.innerHTML= vido[Math.floor(Math.random()*vido.length)];
}
}

else if(buton=='love')
{

var butn1 = document.getElementById('btn');
var rest1 = document.getElementById('res1');
var butn2 = document.getElementById('btn1');
var rest2 = document.getElementById('res3');
var butn3 = document.getElementById('btn2');
var rest3 = document.getElementById('res2');

let l1= "CLICK HERE!";
let l11=l1.link("https://youtu.be/kffacxfA7G4");

let l2= "CLICK HERE!";
let l22=l2.link("https://youtu.be/2Vv-BfVoq4g");

let l3= "CLICK HERE!";
let l33=l3.link("https://youtu.be/J-dv_DcDD_A");

let l4= "CLICK HERE!";
let l44=l4.link(" https://www.instagram.com/p/CLbliTen1Bt/?utm_medium=copy_link");

let l5= "CLICK HERE!";
let l55=l5.link("https://youtu.be/P4p1OcZCBOY");

let l6= "CLICK HERE!";
let l66=l6.link("https://youtu.be/8lRwZdYtyt8");

let l7= "CLICK HERE!";
let l77=l7.link("https://www.instagram.com/reel/CI0wUT3BrE4/?utm_medium=copy_link");

let l8= "CLICK HERE!";
let l88=l8.link("https://www.instagram.com/p/CBk8NCJnvUl/?utm_medium=copy_link");

let l9= "CLICK HERE!";
let l99=l9.link("https://www.instagram.com/p/CV18NzGhtTZ/?utm_medium=copy_link");

var sngg = [l11,l22,l33]
function finish1(){
rest1.innerHTML= sngg[Math.floor(Math.random()*sngg.length)];
}
var qute = [l44,l55,l66]
function finish3(){
rest2.innerHTML= qute[Math.floor(Math.random()*qute.length)];
}

var vido = [l77,l88,l99]
function finish2(){
rest3.innerHTML= vido[Math.floor(Math.random()*vido.length)];
}
}

else if(buton=='anger')
{

var butn1 = document.getElementById('btn');
var rest1 = document.getElementById('res1');
var butn2 = document.getElementById('btn1');
var rest2 = document.getElementById('res3');
var butn3 = document.getElementById('btn2');
var rest3 = document.getElementById('res2');

let a1= "CLICK HERE!";
let a11=a1.link("https://youtu.be/VLutlPcJ0Dc");

let a2= "CLICK HERE!";
let a22=a2.link("https://youtu.be/RQ6iZmTTMnI");

let a3= "CLICK HERE!";
let a33=a3.link("https://youtu.be/bnL_u9WMzl8");

let a4= "CLICK HERE!";
let a44=a4.link("https://www.instagram.com/tv/CVYcoOthGsI/?utm_medium=copy_link");

let a5= "CLICK HERE!";
let a55=a5.link("https://youtu.be/F22ZvJR2mss");

let a6= "CLICK HERE!";
let a66=a6.link("https://youtu.be/1pSHahOyEWA");

let a7= "CLICK HERE!";
let a77=a7.link("https://www.instagram.com/p/CVLJSM3K0Mm/?utm_medium=copy_link");

let a8= "CLICK HERE!";
let a88=a8.link("https://www.instagram.com/p/BVK_maZhNgM/?utm_medium=copy_link");

let a9= "CLICK HERE!";
let a99=a9.link("https://www.instagram.com/p/CUnHLlsoYUW/?utm_medium=copy_link");

var sngg = [a11,a22,a33]
function finish1(){
rest1.innerHTML= sngg[Math.floor(Math.random()*sngg.length)];
}
var qute = [a44,a55,a66]
function finish3(){
rest2.innerHTML= qute[Math.floor(Math.random()*qute.length)];
}

var vido = [a77,a88,a99]
function finish2(){
rest3.innerHTML= vido[Math.floor(Math.random()*vido.length)];
}
}

else if(buton=='fear')
{

var butn1 = document.getElementById('btn');
var rest1 = document.getElementById('res1');
var butn2 = document.getElementById('btn1');
var rest2 = document.getElementById('res3');
var butn3 = document.getElementById('btn2');
var rest3 = document.getElementById('res2');

let f1= "CLICK HERE!";
let f11 =f1.link("https://youtu.be/b5BNUa_op2o");

let f2= "CLICK HERE!";
let f22=f2.link("https://youtu.be/j5-yKhDd64s");

let f3= "CLICK HERE!";
let f33=f3.link("https://youtu.be/403FGqa-Uv8");

let f4= "CLICK HERE!";
let f44=f4.link("https://youtu.be/HnrogLw6SOQ");

let f5= "CLICK HERE!";
let f55=f5.link("https://youtu.be/DnS3vDtOkbs");

let f6= "CLICK HERE!";
let f66=f6.link("https://youtu.be/Pv26KMWG0Vk");

let f7= "CLICK HERE!";
let f77=f7.link("https://www.instagram.com/p/CFye-Ocj6h9/?utm_medium=copy_link");

let f8= "CLICK HERE!";
let f88=f8.link("https://www.instagram.com/p/CVxx4nGgXB-/?utm_medium=copy_link");

let f9= "CLICK HERE!";
let f99=f9.link("https://www.instagram.com/p/B8CMgnInRFq/?utm_medium=copy_link");

var sngg = [f11,f22,f33]
function finish1(){
rest1.innerHTML= sngg[Math.floor(Math.random()*sngg.length)];
}
var qute = [f44,f55,f66]
function finish3(){
rest2.innerHTML= qute[Math.floor(Math.random()*qute.length)];
}

var vido = [f77,f88,f99]
function finish2(){
rest3.innerHTML= vido[Math.floor(Math.random()*vido.length)];
}
}

else if(buton=='surprise')
{

var butn1 = document.getElementById('btn');
var rest1 = document.getElementById('res1');
var butn2 = document.getElementById('btn1');
var rest2 = document.getElementById('res3');
var butn3 = document.getElementById('btn2');
var rest3 = document.getElementById('res2');

let s1= "CLICK HERE!";
let s11=s1.link("https://youtu.be/dhYOPzcsbGM");

let s4= "CLICK HERE!";
let s44=s4.link("https://youtu.be/kTJczUoc26U");

let s7= "CLICK HERE!";
let s77=s7.link("https://www.instagram.com/reel/CJk_c8BACXV/?utm_medium=copy_link");

let s8= "CLICK HERE!";
let s88=s8.link("https://www.instagram.com/p/CF-B0xgnZgJ/?utm_medium=copy_link");

let s9= "CLICK HERE!";
let s99=s9.link("https://www.instagram.com/p/BR3hcfdj04u/?utm_medium=copy_link");

var sngg = [s11]
function finish1(){
rest1.innerHTML= sngg[Math.floor(Math.random()*sngg.length)];
}
var qute = [s44]
function finish3(){
rest2.innerHTML= qute[Math.floor(Math.random()*qute.length)];
}

var vido = [s77,s88,s99]
function finish2(){
rest3.innerHTML= vido[Math.floor(Math.random()*vido.length)];
}
}
else {
    alert("oops you have entered wrong emotion");
}

     
     
     </script>
    
    
    
</body>

    
""")
                   


if __name__ == '__main__':
    
	app.debug = True
	app.run()
	app.run(debug = True)
        
        

        
