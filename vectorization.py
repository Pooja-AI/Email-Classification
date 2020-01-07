import pandas as pd
import glob,os
import numpy as np
import itertools
import warnings
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from data_Preprocessing import DataPreprocessing
# Prepare data

#f='All_Cat_Data_v1.xlsx'
#resp = DataPreprocessing()
#data_df = resp.preprocessing(f)


class Embedding:


    def input_data(self,data_df):
        X = data_df.loc[:,'Input_text']
        Y = data_df.loc[:,'Category']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=.33, random_state=1, stratify=data_df['Category'])
        #print(self.X_train.shape, self.X_test.shape, self.Y_train.shape, self.Y_test.shape)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def Countvectorization(self,X_train, X_test):

        ## CountVectorizer
        count_vectorizer = CountVectorizer(stop_words='english')
        count_train = count_vectorizer.fit_transform(self.X_train)
        count_test = count_vectorizer.transform(self.X_test)
        #print(count_train,count_test)
        return count_train,count_test

    def TfIdfVectorization(self,X_train, X_test):

        ## TfIdfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_train = tfidf.fit_transform(self.X_train)
        tfidf_test = tfidf.transform(self.X_test)
        #print(tfidf_train,tfidf_test)
        return tfidf_train,tfidf_test

