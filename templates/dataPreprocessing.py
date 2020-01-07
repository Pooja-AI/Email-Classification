import pandas as pd
import glob,os
import numpy as np
import itertools
import warnings
import string
#from file import upload_file
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
#filename = 'All_Cat_Data_v1.xlsx'
class DataPreprocessing():
    
      # Defining a function to remove punctuations, convert Input_text to lowercase and remove stop words
    def process_Input_text(Input_text):
        no_punctuations = [char for char in Input_text if char not in string.punctuation]
        no_punctuations = ''.join(no_punctuations)
        clean_words = [word.lower() for word in no_punctuations.split() if word.lower() not in stopwords.words('english')]
        clean_words = [lemmatizer.lemmatize(lem) for lem in clean_words]
        clean_words = " ".join(clean_words)
        return clean_words
    
    def preprocessing(filename):
        data_df = pd.read_excel(filename)
        print(data_df.shape)
        print(data_df['Category'].unique())
        data_null=data_df[data_df.isnull().any(axis=1)]
        print(data_null.shape)
        data_df["Body"].fillna("The information contained in this message may be privileged and confidential. It is intended to be read only by the individual or entity to whom it is addressed or by their designee. If the reader of this message is not the intended recipient, you are on notice that any distribution of this message, in any form, is strictly prohibited. If you have received this message in error, please immediately notify the sender and delete or destroy any copy of this message", inplace = True) 



        data_df['Input_text'] = data_df['Body']+""+data_df['Subject']+""+data_df['Sender']
        data_df['Input_text']=data_df['Input_text'].str.replace(r'[^\x00-\x7F]+', '')
        data_df['Input_text'] = data_df['Input_text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
        data_df['Input_text'] = data_df['Input_text'].str.replace('\d+', '')
        data_df['Input_text'] = data_df['Input_text'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')

        lemmatizer = WordNetLemmatizer()
      


        data_df['Input_text'] = data_df['Input_text'].apply(process_Input_text)

        data_df[['Input_text','Category']].to_csv('Input_data.csv')


        input_data = pd.read_csv('Input_data.csv')

        #dataPreprocessing()




