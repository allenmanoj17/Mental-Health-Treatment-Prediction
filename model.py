# import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# models
from sklearn.ensemble import RandomForestClassifier

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score


#Bagging
from sklearn.neighbors import KNeighborsClassifier

# Read the data
df = pd.read_csv('survey.csv')

# Drop columns
df = df.drop(['comments'], axis= 1)
df = df.drop(['state'], axis= 1)
df = df.drop(['Timestamp'], axis= 1)

defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

for feature in df:
    if feature in intFeatures:
        df[feature] = df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        df[feature] = df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        df[feature] = df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)

# Gender

gender = df['Gender'].str.lower()
gender = df['Gender'].unique()
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in df.iterrows():

    if str.lower(col.Gender) in male_str:
        df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    if str.lower(col.Gender) in female_str:
        df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    if str.lower(col.Gender) in trans_str:
        df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

stk_list = ['A little about you', 'p']
df = df[~df['Gender'].isin(stk_list)]

#Add Ranges to Age
df['Age'].fillna(df['Age'].median(), inplace = True)
s = pd.Series(df['Age'])
s[s<18] = df['Age'].median()
df['Age'] = s
s = pd.Series(df['Age'])
s[s>120] = df['Age'].median()
df['Age'] = s

df['age_range'] = pd.cut(df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

df['self_employed'] = df['self_employed'].replace([defaultString], 'No')
df['work_interfere'] = df['work_interfere'].replace([defaultString], 'Don\'t know' )

#Encoding data
labelDict = {}
for feature in df:
    le = preprocessing.LabelEncoder()
    le.fit(df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df[feature] = le.transform(df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
    
for key, value in labelDict.items():print(key, value)

# Drop 'Country'
df = df.drop(['Country'], axis= 1)

print(df.isnull().sum())  ## NO NULL values

# Scaling Age
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# saving the dataframe
# df.to_csv('aftermath.csv')

feature_cols = ['Gender', 'self_employed', 'family_history', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
X = df[feature_cols]
y = df.treatment

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

methodDict = {}
rmseDict = ()

# feature importances 
# order -> Age, Gender, family_history, benefits, care_options, anonymity, leave, work_interfere

# Model 
classifier= RandomForestClassifier(n_estimators= 100, criterion="entropy", random_state=42)  
classifier.fit(X_train, y_train) 
y_pred= classifier.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, y_pred)
print("ACCURACY OF THE RandomForestClassifier MODEL: ", accuracy*100)

#from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
# Generate predictions with the best method
#clf = AdaBoostClassifier()
#clf.fit(X, y)
#dfTestPredictions = clf.predict(X_test)
#accuracy = metrics.accuracy_score(y_test, dfTestPredictions)
#print("ACCURACY OF THE ADABOOST MODEL: ", accuracy*100)
#ACCURACY OF THE ADABOOST MODEL:  80.34603174603175

#knn = KNeighborsClassifier(n_neighbors=7)
#knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
#print("ACCURACY OF THE KNN MODEL: ", metrics.accuracy_score(y_test, y_pred)) 
# ->  ACCURACY OF THE KNN MODEL:  0.7992063492063492

import pickle
pickle.dump(classifier, open('model.pkl','wb'))