import pickle
from flask import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler




##app = create_app()
##app.app_context().push()

##def create_app():
    ##application = Flask(__name__)
    ##db.init_application(application)
    ##return application
    
## pip install flask


##with application.app_context().push():

## get pickle file

application = Flask(__name__)

app=application

## app.config['SQLALCHEMY_DATABASE_URI']='sqlite:////mnt/c/Users/cj_ch/Desktop/database.db'

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=='__main__':
    application.run(host='0.0.0.0')



























