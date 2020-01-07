from sklearn.naive_bayes import MultinomialNB
from data_Preprocessing import DataPreprocessing
from vectorization import Embedding
from sklearn import svm
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


import os

#filename = 'Input_data.csv'


class model:

        
    def multinomialNB(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        
        #print("--------------------MultinomialNB-CountVector----------")
        nb = MultinomialNB()
        nb.fit(count_train,Y_train )
        nb_pred_train = nb.predict(count_train)
        nb_pred_test = nb.predict(count_test)
        nb_pred_train_proba = nb.predict_proba(count_train)
        nb_pred_test_proba = nb.predict_proba(count_test)
        
        #print('The accuracy for the training data is {}'.format(nb.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(nb.score(count_test, Y_test)))
        
        pickle.dump(nb, open('./models/nb.pkl', 'wb'))

        #print("--------------------MultinomialNB-TFIDF----------")

        nb_tfidf = MultinomialNB()
        nb_tfidf.fit(tfidf_train, Y_train)
        nb_pred_train_tfidf = nb_tfidf.predict(tfidf_train)
        nb_pred_test_tfidf = nb_tfidf.predict(tfidf_test)

        nb_tfidf_pred_train_proba = nb_tfidf.predict_proba(tfidf_train)
        nb_tfidf_pred_test_proba =nb_tfidf.predict_proba(tfidf_test)

        #print('The accuracy for the training data is {}'.format(nb_tfidf.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(nb_tfidf.score(tfidf_test,Y_test)))
        
        # save the model to disk
        pickle.dump(nb_tfidf, open('./models/nb_tfidf.pkl', 'wb'))
        
        return nb_pred_test,nb_pred_test_tfidf
        
    def supportVectorMachine(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        
        #print("--------------------svm-CountVector----------")
        svmc = svm.SVC(kernel='linear',probability=True)
        svmc.fit(count_train,Y_train)
        svmc_pred_train = svmc.predict(count_train)
        svmc_pred_test = svmc.predict(count_test)
        #svmc_pred_train_proba = svmc.predict_proba(count_train)
        #svmc_pred_test_proba = svmc.predict_proba(count_test)
        #print('The accuracy for the training data is {}'.format(svmc.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(svmc.score(count_test, Y_test)))
        pickle.dump(svmc, open('./models/svmc.pkl', 'wb'))
        
        #print("--------------------svm-TfIdf----------")
        
        svmc_tfidf = svm.SVC(kernel='linear',probability=True)
        svmc_tfidf.fit(tfidf_train,Y_train)
        svmc_pred_train_tfidf = svmc_tfidf.predict(tfidf_train)
        svmc_pred_test_tfidf = svmc_tfidf.predict(tfidf_test)
        #print('The accuracy for the training data is {}'.format(svmc_tfidf.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(svmc_tfidf.score(tfidf_test, Y_test)))
        
        pickle.dump(svmc_tfidf, open('./models/svmc_tfidf.pkl', 'wb'))
        
        return svmc_pred_test,svmc_pred_test_tfidf
        
    def decisionTreeClassifier(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        #print("--------------------DecisionTree-Count----------")
        dtc = DecisionTreeClassifier()
        dtc.fit(count_train,Y_train)
        dtc_pred_train = dtc.predict(count_train)
        dtc_pred_test = dtc.predict(count_test)
        dtc_pred_train_proba = dtc.predict_proba(count_train)
        dtc_pred_test_proba = dtc.predict_proba(count_test)
        pickle.dump(dtc, open('./models/dtc.pkl', 'wb'))
        #print('The accuracy for the training data is {}'.format(dtc.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(dtc.score(count_test, Y_test)))
        #print("--------------------decision-TfIdf----------")
        dtc_tfidf = DecisionTreeClassifier()
        dtc_tfidf.fit(tfidf_train,Y_train)
        dtc_pred_train_tfidf = dtc_tfidf.predict(tfidf_train)
        dtc_pred_test_tfidf =dtc_tfidf.predict(tfidf_test)
        dtc_pred_train_proba_tfidf = dtc_tfidf.predict_proba(tfidf_train)
        dtc_pred_test_proba_tfidf = dtc_tfidf.predict_proba(tfidf_test)
        #print('The accuracy for the training data is {}'.format(dtc_tfidf.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(dtc_tfidf.score(tfidf_test, Y_test)))
        pickle.dump(dtc_tfidf, open('./models/dtc_tfidf.pkl', 'wb'))
        return dtc_pred_test,dtc_pred_test_tfidf 
        
    def randomClassifier(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        #print("--------------------RF-Count----------")
        randomclassifier=RandomForestClassifier(n_estimators=100)
        randomclassifier.fit(count_train,Y_train)
        random_predict=randomclassifier.predict(count_test)
        random_pred_train = randomclassifier.predict(count_train)
        random_pred_test = randomclassifier.predict(count_test)
        random_pred_train_proba = randomclassifier.predict_proba(count_train)
        random_pred_test_proba = randomclassifier.predict_proba(count_test)
        pickle.dump(randomclassifier, open('./models/randomclassifier.pkl', 'wb'))
        #print('The accuracy for the training data is {}'.format(randomclassifier.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(randomclassifier.score(count_test, Y_test)))
        #print("--------------------RF-TfIdf----------")
        randomclassifier_tfidf=RandomForestClassifier(n_estimators=100)
        randomclassifier_tfidf.fit(tfidf_train,Y_train)
        random_predict_tfidf=randomclassifier_tfidf.predict(tfidf_test)
        random_pred_train_tfidf = randomclassifier_tfidf.predict(tfidf_train)
        random_pred_test_tfidf = randomclassifier_tfidf.predict(tfidf_test)
        random_pred_train_proba_tfidf = randomclassifier_tfidf.predict_proba(tfidf_train)
        random_pred_test_proba_tfidf = randomclassifier_tfidf.predict_proba(tfidf_test)
        #print('The accuracy for the training data is {}'.format(randomclassifier_tfidf.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(randomclassifier_tfidf.score(tfidf_test, Y_test)))
        pickle.dump(randomclassifier_tfidf, open('./models/randomclassifier_tfidf.pkl', 'wb'))
        
        return random_pred_test,random_pred_test_tfidf
        
    def LogisticRegression(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        #print("-----------LogisticRegression------------")
        logreg = LogisticRegression()
        logreg.fit(count_train, Y_train)
        logreg_train = logreg.predict(count_train)
        logeg_test = logreg.predict(count_test)
        logreg_train_proba = logreg.predict_proba(count_train)
        logreg_test_proba = logreg.predict_proba(count_test)
        pickle.dump(logreg, open('./models/logreg.pkl', 'wb'))

        #print('The accuracy for the training data is {}'.format(logreg.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(logreg.score(count_test, Y_test)))
        
        logreg_tfidf = LogisticRegression()
        logreg_tfidf.fit(tfidf_train, Y_train)
        logreg_tfidf_train = logreg.predict(tfidf_train)
        logreg_tfidf_test = logreg.predict(tfidf_test)
        logreg_train_proba_tfidf = logreg.predict_proba(tfidf_train)
        logreg_test_proba_tfidf = logreg.predict_proba(tfidf_test)
        pickle.dump(logreg_tfidf, open('./models/logreg_tfidf.pkl', 'wb'))

        #print('The accuracy for the training data is {}'.format(logreg_tfidf.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(logreg_tfidf.score(tfidf_test, Y_test)))
        
        return logeg_test,logreg_tfidf_test
        
    def XGBootClassification(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        xgb = XGBClassifier()
        xgb.fit(count_train, Y_train)
        xgb_pred_train = xgb.predict(count_train)
        xgb_pred_test = xgb.predict(count_test)
        xgb_pred_train_proba = xgb.predict_proba(count_train)
        xgb_pred_test_proba = xgb.predict_proba(count_test)
        pickle.dump(xgb, open('./models/xgb.pkl', 'wb'))

        #print('The accuracy for the training data is {}'.format(xgb.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(xgb.score(count_test, Y_test)))
        
        
        xgb_tfidf = XGBClassifier()
        xgb_tfidf.fit(tfidf_train, Y_train)
        xgb_pred_train_tfidf = xgb_tfidf.predict(tfidf_train)
        xgb_pred_test_tfidf = xgb_tfidf.predict(tfidf_test)
       
        xgb_tfidf_pred_train_proba =xgb_tfidf.predict_proba(tfidf_train)
        xgb_tfidf_pred_test_proba =xgb_tfidf.predict_proba(tfidf_test)
        pickle.dump(xgb_tfidf, open('./models/xgb_tfidf.pkl', 'wb'))

        #print('The accuracy for the training data is {}'.format(xgb_tfidf.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(xgb_tfidf.score(tfidf_test, Y_test)))
        
        return xgb_pred_test,xgb_pred_test_tfidf
        
    def KNNCLassification(self,count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test):
        
        
        modelknn = KNeighborsClassifier(n_neighbors=5)
        modelknn.fit(count_train, Y_train)
        modelknn_train = modelknn.predict(count_train)
        modelknn_test = modelknn.predict(count_test)
        modelknn_train_proba = modelknn.predict_proba(count_train)
        modelknn_test_proba = modelknn.predict_proba(count_test)
        pickle.dump(modelknn, open('./models/modelknn.pkl', 'wb'))

        #print('The accuracy for the training data is {}'.format(modelknn.score(count_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(modelknn.score(count_test, Y_test)))
        
        
        modelknn_tfidf = KNeighborsClassifier(n_neighbors=5)
        modelknn_tfidf.fit(tfidf_train, Y_train)
        modelknn_tfidf_train = modelknn.predict(tfidf_train)
        modelknn_tfidf_test = modelknn.predict(tfidf_test)
        modelknn_tfidf_test_proba = modelknn.predict_proba(tfidf_test)
        pickle.dump(modelknn_tfidf, open('./models/modelknn_tfidf.pkl', 'wb'))

        #print('The accuracy for the training data is {}'.format(modelknn.score(tfidf_train, Y_train)))
        #print('The accuracy for the testing data is {}'.format(modelknn.score(tfidf_test, Y_test)))
        
        return modelknn_test, modelknn_tfidf_test 
                    
    
         
         
                  
        
  




