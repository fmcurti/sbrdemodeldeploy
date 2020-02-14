from fastapi import FastAPI, File, UploadFile
from joblib import load,dump
from pydantic import  BaseModel
from typing import List
import shutil
import os
import numpy as np
import sklearn

# load model
async def load_model(clientID,modelName):
    try:
        return load('models/'+str(clientID)+'/'+modelName)
    except:
        return False


async def get_prediction(name, params,clientID):
    clf = await load_model(clientID,name)
    if not clf:
        return {'error': 'unknown model'}

    x = np.array([params])
    y = clf.predict(x)[0]  # just get single value
    
    try:
        prob = clf.predict_proba(x)[0].tolist()  # send to list for return
    except:
        return {'prediction': float(y)}

    return {'prediction': float(y), 'probabilities': prob}


app = FastAPI(title="SBRDE Model Deployment",
    description="Simple Model Inference API for SKLearn and Tensorflow",
    version="0.1",)

class Model(BaseModel):
    name: str
    clientID : int
    params: list = []

@app.post("/uploadfile/{clientID}")
async def create_upload_file(clientID:int ,file: UploadFile = File(...)):
    file_object = file.file
    try:
        clf = load(file_object)
        if not isinstance(clf,sklearn.base.BaseEstimator) and not isinstance(clf,sklearn.pipeline.Pipeline):
            return {'error': 'File is not SkLearn Model'}
    except:
        return {'error': 'File is not SkLearn Model'}
    upload_folder = os.path.join('models',str(clientID))
    if not os.path.exists(upload_folder):
        os.mkdir(upload_folder)
    #create empty file to copy the file_object to
    dump(clf,os.path.join(upload_folder, file.filename))
    #upload_folder = open(os.path.join(upload_folder, file.filename), 'wb+')
    #shutil.copyfileobj(file_object, upload_folder)
    #upload_folder.close()
    return {"filename": file.filename}

@app.get("/listmodels/{clientID}")
async def list_models(clientID: int):
    mypath = os.path.join('models',str(clientID))
    if not os.path.exists(mypath):
        return []
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return '|'.join(onlyfiles)

@app.post("/predict")
async def predict(model: Model):
    pred = await get_prediction(model.name, model.params,model.clientID)
    return pred