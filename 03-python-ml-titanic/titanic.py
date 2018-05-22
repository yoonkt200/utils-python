#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:24:00 2018

@author: yoon
"""

import pandas as pd

train_file = "/Users/yoon/Downloads/train.csv"
test_file = "/Users/yoon/Downloads/test.csv"
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train = train.drop('Ticket.1', 1)
train = train.drop('No', 1)
train = train.drop('구매지역', 1)
train = train.drop('lastname', 1)
train = train.drop('family#', 1)
                   
test = test.drop(test.columns[0], axis=1)

# head show, null check
train.head(20)
train.isnull().sum()
test.isnull().sum()

# no columns labeling
test.columns = train.columns
test.isnull().sum()

# Cabin column 전처리 : 캐비닛 유무로 /// 캐비닛 알파벳번호
train['Cabin'].value_counts()
train['Cabin'].fillna(0, inplace=True)
train.loc[train['Cabin']!=0, 'Cabin'] = 1

test['Cabin'].fillna(0, inplace=True)
test.loc[test['Cabin']!=0, 'Cabin'] = 1

train.head()

# 생존자, 비생존자 캐비닛 유무 : 약 25% 차이
from scipy import stats
unsurvived = train.loc[train['Survived']==0]
survived = train.loc[train['Survived']==1]

unsurvived['Cabin'].value_counts()
print(58/(381+58) * 100)
survived['Cabin'].value_counts()
print(105/(168+105) * 100)

# Pclass
train['Pclass'].value_counts()
test['Pclass'].value_counts()

# Name Drop /// MR, MRS, MISS 나누기
train['Name_Grade'] = ""
train['Name_Grade'] = train.apply(lambda row: mr_grade(row['Name']), axis=1)
def mr_grade(row):
    if 'Rev.' in row:
        return "1"
    elif 'Major.' in row:
        return "1"
    elif 'Dr.' in row:
        return "1"
    elif 'Mme.' in row:
        return "2"
    elif 'Mrs.' in row:
        return "2"
    elif 'Jonkheer.' in row:
        return "3"
    elif 'Countess.' in row:
        return "3"
    elif 'Mr.' in row:
        return "4"
    elif 'Master.' in row:
        return "5"
    elif 'Miss.' in row:
        return "6"
    else:
        return "7"

test['Name_Grade'] = ""
test['Name_Grade'] = test.apply(lambda row: mr_grade(row['Name']), axis=1)

train = train.drop('Name', 1)
test = test.drop('Name', 1)

# Ticket Drop
train = train.drop('Ticket', 1)
test = test.drop('Ticket', 1)

# Sex : Null 36 ~ Drop
train['Sex'].value_counts()
no_sex = train[train['Sex'].isnull()] # all die
unsurvived = train.loc[train['Survived']==0]
survived = train.loc[train['Survived']==1]

unsurvived['Sex'].value_counts()
print(53/(439) * 100)
survived['Sex'].value_counts()
print(188/(273) * 100)

male = train[train['Sex'] == 'male']
male["Fare"].mean()
female = train[train['Sex'] == 'female']
female["Fare"].mean()

train = train.dropna(subset=['Sex'])

# embarked : Null 2 ~ Drop
train['Embarked'].value_counts()
train = train.dropna(subset=['Embarked'])
test = test.dropna(subset=['Embarked'])

# SibSp, Parch : Parch N delete ~ Drop
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Parch'] = train['Parch'].apply( lambda x: x.replace('한명', '0') )
train['Parch'] = train[train['Sex'] != 'N']

# make Family_size
train['Parch'] = train['Parch'].apply(pd.to_numeric)
test['Parch'] = test['Parch'].apply(pd.to_numeric)
train["Family_size"] = train["SibSp"] + train["Parch"] + 1
test["Family_size"] = test["SibSp"] + test["Parch"] + 1

train = train.drop('Parch', 1)
test = test.drop('Parch', 1)
train = train.drop('SibSp', 1)
test = test.drop('SibSp', 1)

# Age
no_age = train[train['Age'].isnull()]
has_age = train[train['Age'].notnull()]

y = has_age['Age']
has_age_origin = has_age
has_age = has_age.drop('Age', 1)
no_has_age = no_age.drop('Age', 1)
no_df_selected_dummies = pd.get_dummies(no_has_age)

# dummies
df_selected_dummies = pd.get_dummies(has_age)
df_selected_dummies = df_selected_dummies.drop("Name_Grade_3", 1)
df_selected_dummies = df_selected_dummies.drop("Name_Grade_7", 1)

# regression
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(df_selected_dummies, y)
print(forest.feature_importances_)


# 학습평가
age = forest.predict(no_df_selected_dummies)
b = pd.Series(age)
b = b.to_frame()
b.columns = ["Age"]
b.reset_index()

no_age = no_age.drop('Age', 1)
no_age = no_age.reset_index()
no_age = no_age.drop('index', 1)
c = no_age.join(b)

has_age_origin = has_age_origin.reset_index()
has_age_origin = has_age_origin.drop('index', 1)
c = c.reset_index()
c = c.drop('index', 1)

############################################################

train_trimed = pd.read_csv('/Users/yoon/Downloads/out1.csv')
test_trimed = pd.read_csv('/Users/yoon/Downloads/out_test.csv')
train_trimed.isnull().sum()
test_trimed.isnull().sum()
train_trimed.columns

# training
y = train_trimed['Survived']
X = train_trimed.drop('Survived', 1)

from sklearn.model_selection import train_test_split

X_selected_dummies = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_selected_dummies, y, test_size=0.3, random_state=0)

# classification : random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
print("accuracy: %.2f" %accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, X_train, y_train, cv=10)
print(scores)

# SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("accuracy: %.2f" %accuracy_score(y_test, y_pred))
scores = cross_val_score(forest, X_train, y_train, cv=10)
print(scores)

# classification : xgb
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("accuracy: %.2f" %accuracy_score(y_test, y_pred))

scores = cross_val_score(model, X_train, y_train, cv=10)
print(scores)

# predict list
testSet = test_trimed.drop('Survived', 1)
testSet_dummies = pd.get_dummies(testSet)
y_pred = model.predict(testSet_dummies)

# =============================================================================
# trimed_testSet = testSet.drop('Name_Grade', 1)
# trimed2_testSet = trimed_testSet.drop('Age', 1)
# df_selected_dummies = pd.get_dummies(trimed2_testSet)
# age_pred = forest.predict(df_selected_dummies)
# y_pred = pd.Series(age_pred)
# 
# df_y_pred = y_pred.to_frame()
# df_y_pred.columns = ["Age"]
# df_y_pred.reset_index()
# 
# df_y_pred.to_csv('test_pred_age.csv')
# =============================================================================

y_pred = pd.Series(y_pred)

df_y_pred = y_pred.to_frame()
df_y_pred.columns = ["Age"]
df_y_pred.reset_index()
df_y_pred = df_y_pred.drop('index', 1)

df_y_pred.to_csv('final_y_pred.csv')