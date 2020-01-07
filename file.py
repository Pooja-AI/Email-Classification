##https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
from flask import Flask, request, jsonify,render_template,redirect,flash
import pandas as pd
import matplotlib.pyplot as plt
#from flask_cors import CORS
from data_Preprocessing import DataPreprocessing
from vectorization import Embedding
from models import model
import os
from predict_model import predict
from dataVisualization import DataVisualization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
app = Flask(__name__, static_url_path='')
#app = Flask(__name__, static_url_path = "/static", static_folder = "static")


@app.route('/')
def home():
   return render_template('index.html')
   

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():

    
    if request.method == 'POST':
        #print(request.files['file-7[]'])
       #import pdb;pdb.set_trace();
        f = request.files['file-7[]']
        #data_xls = pd.read_excel(f)
        resp = DataPreprocessing()
        data_df = resp.preprocessing(f)
        #print(data_df)
        ##Object for Vectorization
        target_names = ['Cancellation_Rescheduling','EDI_CustomerProgram','Escalation_Linedown',
                'Logistic_changes','MDL_Exclusion','NewPO_Forecast',       
             'OrderEnquiry','Other','POChanges','RMA' ]
        class_vector = Embedding()
        X_train, X_test, Y_train, Y_test=class_vector.input_data(data_df)
        count_train,count_test = class_vector.Countvectorization(X_train, X_test)
        tfidf_train,tfidf_test = class_vector.TfIdfVectorization(X_train, X_test)
        ##Created Objects for models
        models=model()
        vis = DataVisualization()
        ##multinomialNB
        nb_pred_test,nb_pred_test_tfidf=models.multinomialNB(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        
        ##confusion matrix and classification report using CoutVectorization
        print("----NaiveBayes model Using Count Vectors----")
        print(classification_report(Y_test, nb_pred_test))
        nbcm1 = confusion_matrix(Y_test, nb_pred_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm1, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/NB_CountVector.png')
        
        ##confusion matrix and classification report using Tfidf
        print("------NaiveBayes model Using Tfidf -----")
        nbcm2 = confusion_matrix(Y_test,nb_pred_test_tfidf)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/NB_TfIdf.png')
       
        ##supportVectorMachine
        svmc_pred_test,svmc_pred_test_tfidf = models.supportVectorMachine(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        ##confusion matrix and classification report using CoutVectorization
        print("----SVM Using Count Vectors----")
        print(classification_report(Y_test, svmc_pred_test))
        svmcm1 = confusion_matrix(Y_test,svmc_pred_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/svmCount.png')
        
        ##confusion matrix and classification report using Tfidf
        print("--------SVM Tfidf------")
        print(classification_report(Y_test,svmc_pred_test_tfidf))
        svmcm1 = confusion_matrix(Y_test,svmc_pred_test_tfidf)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/svmTfidf.png')
        
        ##decisionTreeClassifier
        dtc_pred_test,dtc_pred_test_tfidf=models.decisionTreeClassifier(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        
        ##confusion matrix and classification report using CoutVectorization
        print("--------Decision CountVector------")
        print(classification_report(Y_test,dtc_pred_test))
        dtc1 = confusion_matrix(Y_test,dtc_pred_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/dtc_Count.png')
        
        ##confusion matrix and classification report using Tfidf
        print("--------Decision tfidf------")
        print(classification_report(Y_test,dtc_pred_test_tfidf))
        dtc2 = confusion_matrix(Y_test,dtc_pred_test_tfidf)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/dtc_Tfidf.png')
        
        ##randomClassifier
        random_pred_test,random_pred_test_tfidf=models.randomClassifier(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        
        ##confusion matrix and classification report using CoutVectorization
        print("--------RandomForest CountVector------")
        print(classification_report(Y_test,random_pred_test))
        randomclassifier1 = confusion_matrix(Y_test,random_pred_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/RF_Count.png')
        
        ##confusion matrix and classification report using Tfidf
        print("--------RandomForest tfidf------")
        print(classification_report(Y_test,dtc_pred_test_tfidf))
        randomclassifier2 = confusion_matrix(Y_test,random_pred_test_tfidf)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/RF_Tfidf.png')
        
        
        
        ##LogisticRegression
        logeg_test,logreg_tfidf_test= models.LogisticRegression(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        ##confusion matrix and classification report using CoutVectorization
        print("--------LogisticRegression CountVector------")
        print(classification_report(Y_test,logeg_test))
        randomclassifier1 = confusion_matrix(Y_test,logeg_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/logreg_Count.png')
        
        ##confusion matrix and classification report using Tfidf
        print("--------LogisticRegression tfidf------")
        print(classification_report(Y_test,logreg_tfidf_test))
        randomclassifier2 = confusion_matrix(Y_test,logreg_tfidf_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/logreg_Tfidf.png')
        
        ##XGBootClassification
        xgb_pred_test,xgb_pred_test_tfidf=models.XGBootClassification(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        
        
        ##confusion matrix and classification report using CoutVectorization
        print("-------- XGBootClassification CountVector------")
        print(classification_report(Y_test,xgb_pred_test))
        randomclassifier1 = confusion_matrix(Y_test,xgb_pred_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/xgb_Count.png')
        
        ##confusion matrix and classification report using Tfidf
        print("--------XGBootClassification  tfidf------")
        print(classification_report(Y_test,xgb_pred_test_tfidf))
        randomclassifier2 = confusion_matrix(Y_test,xgb_pred_test_tfidf)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/xgb_Tfidf.png')
        
        ##KNNCLassification
        modelknn_test, modelknn_tfidf_test = models.KNNCLassification(count_train,count_test,tfidf_train,tfidf_test,Y_train,Y_test)
        ##confusion matrix and classification report using CoutVectorization
        print("-------- KNN Classification CountVector------")
        print(classification_report(Y_test,modelknn_test))
        randomclassifier1 = confusion_matrix(Y_test,modelknn_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/knn_Count.png')
        
        ##confusion matrix and classification report using Tfidf
        print("--------KNN Classification  tfidf------")
        print(classification_report(Y_test,modelknn_tfidf_test))
        randomclassifier2 = confusion_matrix(Y_test,modelknn_tfidf_test)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        vis.plot_confusion_matrix(nbcm2, classes=target_names)
        plt.savefig('/home/allu/Documents/TCSProjetcs/EmailClassification/static/images/knn_Tfidf.png')
        
        

        return render_template('home.html')
        #return 'File Uploaded successfully'  
        #print(data_xls)
        #return data_xls.to_html()
    return render_template('file.html')
        #return "File uploaded successfully"

@app.route("/predict", methods=['GET', 'POST'])
def predictor():
    p = predict()
    if request.method == 'POST':
        message = request.form['mail']
        data = [message]
        result = p.prediction(data)
        #result = str(result)
        #print(result)
        #print(type(result))
        return render_template('sample.html',  tables=[result.to_html(classes='data')], titles=result.columns.values)
        #return result
    return render_template('predict.html')


@app.route("/evalute")
def evalute():
   return render_template('dash.html')


@app.route("/export", methods=['GET'])
def export_records():
    return 

if __name__ == "__main__":
    app.run()
