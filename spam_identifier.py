import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mail_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\combined_data.csv")

print(mail_data.head())
print(mail_data.shape)
print(mail_data.isnull().sum())
#stemming  remove multiple words like programmers ,programming,programmer to program )and etc.
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)
mail_data['text']=mail_data['text'].apply(stem)
#X variable contain features and y contin label
X=mail_data['text'].values
print(X)
Y=mail_data['label'].values
# convert the text data into vectorizer
vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)
# train and test data
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
#accuracy score of training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)

#accuracy score of testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(testing_data_accuracy)

# testing the model
input_data=('You are 1 step away from winning a car! Use your team to join the KOL vs HYD Guaranteed Contest & win a car on Howzat! https://hwzat.in/e97801P9LJb')
input_data_as_vectorizer=vectorizer.transform([input_data])
input_data_array=input_data_as_vectorizer.toarray()
predicton = model.predict(input_data_array)
# 0 assign for ham and 1 assign for spam
if(predicton[0]==0):
    print("The Mail is NOT Spam")
else:
    print("The Mail is Spam")


from joblib import dump,load
dump(model,"Spam_filter.joblib")
dump(vectorizer,"Vectorizer.joblib")