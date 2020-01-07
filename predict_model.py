from data_Preprocessing import DataPreprocessing
from models import model
from nltk.corpus import stopwords
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data_df = pd.read_csv('Input_data.csv')
X = data_df.loc[:,'Input_text']
Y = data_df.loc[:,'Category']
labels = data_df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=1, stratify=data_df['Category'])

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
labels = LabelEncoder()
y_train_labels_fit = labels.fit(Y_train)
y_train_lables_trf = labels.transform(Y_train)
#print(labels.classes_)
   
#tstStr=['hi valeria check get 5k item short period time customer shortage help freundliche grüße best regard vasilios adamos rutronik elektronische bauelemente gmbh team leader purchasing industriestraße 2 de75228 ispringen phone 49 7231 8011728 fax 49 7231 8011633 email vasiliosadamosrutronikcom httpwwwrutronikcom httpwwwrutronikcom rutronik24 next generation ecommerce httpwwwrutronik24com httpwwwrutronik24com email including attachment may contain business trade secret confidential legally protected information received email error hereby notified review use copying distribution strictly prohibited please inform u immediately destroy email thank geschäftsführer helmut rudel thomas rudel markus krieg marco nabinger dr gregor sommer sitz der gesellschaft 75228 ispringen registergericht amtsgericht mannheim hrb 503663 printing please think environment bsp149 h6327 sp001058818 5k shortly availablevasiliosadamosrutronikcom']

#tstStr2 = ['dear liza enclosed new order schedule please confirm within 2 working day kindly confirming cancellation push out thanks advance katalin katalin palmuller logistics buyer creating value increase customer competitiveness zalaegerszeg zrínyi út 38 h8900 36 92 50 7211 direct katalintothnepalmullerflexcom mailtokatalintothnepalmullerflexcom legal disclaimer information contained message may privileged confidential intended read individual entity addressed designee reader message intended recipient notice distribution message form strictly prohibited received message error please immediately notify sender delete destroy copy message infineon wk 512018katalin tothne palmuller']

class predict:

    def remove_punctuation(self,s):
        lst = []
        for line in s:
	      
            line = re.sub(r'[^ -~]', ' ',line)
            line = re.sub(r'xD', ' ',line)
            line = re.sub(r'[\xa0\s]+', ' ', line)
            line = re.sub('[^A-Za-z: \n]+', ' ', line)
            line = re.sub('(\w+@\w+.com)',' ',line)
            line = re.sub('\s(\w){2}\s',' ',line)
            line = re.sub('(To:|From:|Sent:|Cc:|Subject:).*$','',line)
            lst.append(line)
            return lst
            


    def prediction(self,test_data):
        data = self.remove_punctuation(test_data)
        count_test = count_vectorizer.transform(data).toarray()

        nb_count = pickle.load(open('./models/nb.pkl', 'rb'))
        nb_tfidf = pickle.load(open('./models/nb_tfidf.pkl','rb'))

        svm_count = pickle.load(open('./models/svmc.pkl', 'rb'))
        svm_tfidf = pickle.load(open('./models/svmc_tfidf.pkl', 'rb'))
    
        randomclassifier_count = pickle.load(open('./models/randomclassifier.pkl', 'rb'))
        randomclassifier_tfidf = pickle.load(open('./models/randomclassifier_tfidf.pkl', 'rb'))
     
        dtc_count = pickle.load(open('./models/dtc.pkl', 'rb'))
        dtc_tfidf = pickle.load(open('./models/dtc_tfidf.pkl', 'rb'))
  
        logreg_count = pickle.load(open('./models/logreg.pkl', 'rb'))
        logreg_tfidf = pickle.load(open('./models/logreg_tfidf.pkl', 'rb'))

        xgb_count = pickle.load(open('./models/xgb.pkl', 'rb'))
        xgb_tfidf = pickle.load(open('./models/xgb_tfidf.pkl', 'rb'))
   
        modelknn_count = pickle.load(open('./models/modelknn.pkl', 'rb'))
        modelknn_tfidf = pickle.load(open('./models/modelknn_tfidf.pkl', 'rb'))
        
        nbCount_Prediction = nb_count.predict(count_test)
        nbTfidf_Prediction = nb_tfidf.predict(count_test)
        svmCount_Prediction=svm_count.predict(count_test)
        svmTfidf_Prediction=svm_tfidf.predict(count_test)
        randomclassifierCount_Prediction=randomclassifier_count.predict(count_test)
        randomclassifierTfidf_Prediction=randomclassifier_tfidf.predict(count_test)
        dtccount_Prediction = dtc_count.predict(count_test)
        dtctfidf_Prediction = dtc_tfidf.predict(count_test)
        logregCount_Prediction = logreg_count.predict(count_test)
        logregtfidf_Prediction = logreg_tfidf.predict(count_test)
        xgbcount_Prediction = xgb_count.predict(count_test)
        xgbtfidf_Prediction = xgb_tfidf.predict(count_test)
        modelknncount_Prediction = modelknn_count.predict(count_test)
        modelknntfidf_Prediction = modelknn_tfidf.predict(count_test)
        
        
        npCount_acc =np.max((nb_count.predict_proba(count_test)))
        npTfidf_acc=np.max((nb_tfidf.predict_proba(count_test)))
        svmCount_acc=np.max((svm_count.predict_proba(count_test)))
        svmTfidf_acc=np.max((svm_tfidf.predict_proba(count_test)))
        randomCount_acc=np.max((randomclassifier_count.predict_proba(count_test)))
        randomTfidf_acc=np.max((randomclassifier_tfidf.predict_proba(count_test)))
        dtcCount_acc=np.max((dtc_count.predict_proba(count_test)))
        dtcTfidf_acc=np.max((dtc_tfidf.predict_proba(count_test)))
        logCount_acc=np.max((logreg_count.predict_proba(count_test)))
        logTfidf_acc=np.max((logreg_tfidf.predict_proba(count_test)))
        xgbCount_acc=np.max((xgb_count.predict_proba(count_test)))
        xgbTfidf_acc=np.max((xgb_tfidf.predict_proba(count_test)))
        knnCount_acc=np.max((modelknn_count.predict_proba(count_test)))
        knnTfidf_acc=np.max((modelknn_tfidf.predict_proba(count_test)))
       
        prediction_df  = {'Models':['NaiveBayes_count', 'NaiveBayes_tfidf', 'SVM_Count',
                                  'SVM_TfIdf','RandomForest_Count','RandomForest_Tfidf',
                                 'DecisionTree_Count','DecisionTree_Tfidf','LogisticRegression_Count',
                                 'LogisticRegression_Tfidf','XGBoost_Count','XGBoost_tfidf',
                                 'KNN_Count','KNN_Tfidf'],
                          'Prediction':[nbCount_Prediction, nbTfidf_Prediction,
                                 svmCount_Prediction, svmTfidf_Prediction,
                                randomclassifierCount_Prediction,randomclassifierTfidf_Prediction,
                                dtccount_Prediction,dtctfidf_Prediction,
                                logregCount_Prediction,logregtfidf_Prediction,
                                xgbcount_Prediction,xgbtfidf_Prediction,
                                modelknncount_Prediction,modelknntfidf_Prediction],
                         
                         'Accuracy':[npCount_acc,npTfidf_acc,svmCount_acc,svmTfidf_acc,
                                    randomCount_acc,randomTfidf_acc,dtcCount_acc,dtcTfidf_acc,
                                    logCount_acc,logTfidf_acc,xgbCount_acc,xgbTfidf_acc,
                                    knnCount_acc,knnTfidf_acc]} 

        
        
        result = pd.DataFrame(prediction_df)
        #result.to_html('sample.html')
        return result


