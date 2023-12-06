from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from joblib import load
# from imblearn.over_sampling import SMOTE, RandomOverSampler

# var global

app = Flask(__name__, static_url_path='/static')
model = load('model_efektivitas_karyawan1.model')


# routing
# @app.post()
#beranda
@app.route("/")
def Slider():
    return render_template('slider.html')

@app.route("/Beranda")
def Beranda():
    return render_template('kelompok1.html')

@app.route("/About")
def About():
    return render_template('biodata.html')

if __name__ == '__main__': 
    #load model training
    # model = load('model_efektivitas_karyawan.model')
    
    #run flask lokal
        app.run(host="localhost", port=5000, debug=True)
    
 #run flask lokal

#routing untuk api
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():

   
    # nilai default variabel input
    input_age = 50
    input_job_level = 50
    input_marital_status = 50
    input_monthly_income = 50
    input_overtime = 50
    input_total_working_years = 50
    input_years_in_current_role = 50

    if request.method=='POST':
        #set nilai variabel input 
        input_age = int(request.form['age'])
        input_job_level = int(request.form['job_level'])
        input_marital_status = int(request.form['marital_status'])
        input_monthly_income = int(request.form['monthly_income'])
        input_overtime = int(request.form['overtime'])
        input_total_working_years = int(request.form['total_working_years'])
        input_years_in_current_role = int(request.form['years_in_current_role'])

        #prediksi faktor efektivitas
        df_test = pd.DataFrame(data={
            "age" : [input_age],
            "job_level" : [input_job_level],
            "marital_status" : [input_marital_status],
            "monthly_income" : [input_monthly_income],
            "overtime" : [input_overtime],
            "total_working_years" : [input_total_working_years],
            "years_in_current_role" : [input_years_in_current_role]
        })

        hasil_prediksi = model.predict(df_test)
        # hasil = model.predict([[50, 50, 50, 50, 50, 50, 50]])
        #set path

   
        #return hasil dgn format json
        return jsonify({
            "prediksi" : hasil_prediksi
        })




