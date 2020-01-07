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

class DataPreprocessing:
    lemmatizer = WordNetLemmatizer()
   
    def process_Input_text(self,Input_text):
        no_punctuations = [char for char in Input_text if char not in string.punctuation]
        no_punctuations = ''.join(no_punctuations)
        clean_words = [word.lower() for word in no_punctuations.split() if word.lower() not in stopwords.words('english')]
        clean_words = [self.lemmatizer.lemmatize(lem) for lem in clean_words]
        clean_words = " ".join(clean_words)
        return clean_words
    
    def preprocessing(self,filename):
        self.data_df = pd.read_excel(filename)
        #print(self.data_df.shape)
        #print(self.data_df['Category'].unique())
        self.data_null=self.data_df[self.data_df.isnull().any(axis=1)]
        #print(self.data_null.shape)
        self.data_df["Body"].fillna("The information contained in this message may be privileged and confidential. It is intended to be read only by the individual or entity to whom it is addressed or by their designee. If the reader of this message is not the intended recipient, you are on notice that any distribution of this message, in any form, is strictly prohibited. If you have received this message in error, please immediately notify the sender and delete or destroy any copy of this message", inplace = True) 



        self.data_df['Input_text'] = self.data_df['Body']+""+ self.data_df['Subject']+""+ self.data_df['Sender']
        self.data_df['Input_text']= self.data_df['Input_text'].str.replace(r'[^\x00-\x7F]+', '')
        self.data_df['Input_text'] = self.data_df['Input_text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
        self.data_df['Input_text'] = self.data_df['Input_text'].str.replace('\d+', '')
        self.data_df['Input_text'] = self.data_df['Input_text'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
       

        self.data_df['Input_text'] = self.data_df['Input_text'].apply(self.process_Input_text)
        self.data_df[['Input_text','Category']].to_csv('Input_data.csv')
        #print( data_df.columns)
        
        
        sms_count = pd.value_counts(self.data_df['Category'], sort= True)
        print(sms_count)
        ax = sms_count.plot(kind='bar', figsize=(10,10), color= ["green", "orange","red","yellow","pink","blue","Grey","teal","gold"], fontsize=13)

        ax.set_alpha(0.8)
        ax.set_title("Percentage of mails")
        ax.set_ylabel("Count of classes");
        ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

        totals = []
        for i in ax.patches:
            totals.append(i.get_height())
         
            total = sum(totals)

        # set individual bar lables using above list
        for i in ax.patches:
            string = str(round((i.get_height()/total)*100, 2))+'%'
        # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x()+0.16, i.get_height(), string, fontsize=13, color='black')
            ax.figure.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/data.png')
        
        return  self.data_df
    


#resp_class = DataPreprocessing()
#resp_fun = resp_class.preprocessing('All_Cat_Data_v1.xlsx')
#input_data = resp_class.input_data('Input_data.csv')
    


