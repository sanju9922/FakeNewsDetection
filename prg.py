import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

data_fake = pd.read_csv("/content/Fake.csv")


data_true = pd.read_csv('/content/True.csv')

data_fake.head()

data_fake["class"]=0
data_true["class"]=1


data_fake.shape,data_true.shape

data_true.head()

news=pd.concat([data_fake,data_true],axis=0)

news.head()


news.tail()

news.isnull().sum()

news= news.drop(['title','subject','date'],axis=1)

news = news.sample(frac=1)  #Reshuffling

news.head()

news.reset_index(inplace = True)

news.head()

news.drop(['index'],axis=1,inplace=True)

news.head()

def wordopt(text):
  # to convert into lower case
  text = text.lower()
  # to remove the URLs
  text = re.sub(r'https?://\S+|www\.\S+','',text)
  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)
  # Remove punctuation
  text = re.sub(r'[^\w\s]', '', text)
  # Remove digits
  text = re.sub(r'\d', '', text)
  # Remove newline characters
  text = re.sub(r'\n',' ', text)
  return text

news['text']= news['text'].apply(wordopt)

news['text']

x=news['text']
y=news['class']

x

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)

x_train.shape
x_test.shape
#!pip show scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()


xv_train =vectorization.fit_transform(x_train)

xv_test = vectorization.transform(x_test)

xv_test

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(xv_train,y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test,y_test)

print(classification_report(y_test,pred_lr))

from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()
DTC.fit(xv_train,y_train)
pred_dtc = DTC.predict(xv_test)
DTC.score(xv_test,y_test)
print(classification_report(y_test,pred_dtc))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(xv_train,y_train)
predict_rfc= rfc.predict(xv_test)
rfc.score(xv_test,y_test)
print(classification_report(y_test,predict_rfc))
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(xv_test,y_test)
pred_gbc = gbc.predict(xv_test)
gbc.score(xv_test,y_test)
print(classification_report(y_test,pred_gbc))
def output_label(n):
  if n==0:
    return "It is a fake news"
  elif n==1:
    return "It is a genuine news"

def manual_testing (news):
    testing_news = {"text": [news]} # Corrected syntax for defining dictionary
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization. transform(new_x_test) # Assuming 'vectorization' is your vectorizer object
    pred_lr= LR.predict(new_xv_test)
    # pred_dtc = dtc.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    return "\n\nLR Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_label (pred_lr[0]), output_label(pred_gbc[0]), output_label(pred_rfc[0]))

news_article=(str(input()))

manual_testing(news_article)