from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import cv2 as cv
from sklearn.svm import SVC
import os
#1. Lunch the server
#2. Create the route path
#3. Make the prediction
#4. Plot the result
app = FastAPI()

@app.get('/')
async def root():
    return {'message':'Server says hello'}

@app.get('/result')
async def prediction(image:str):
    
    # Load the model
    with open('model_SVC.p', 'rb') as model_file:
        model : SVC= pickle.load(model_file)
    # Lunch the prediction
    image_preprocess = cv.imread(image)
    image_preprocess = cv.resize(image_preprocess, (15,15))
    image_array = np.asarray(image_preprocess).flatten()
    print(image_array.shape)
    print(model.classes_)
    image_prediction = model.predict([image_array]) [0]
    label = "Car" if image_prediction == 1 else "No Car"
    
    os.remove(image)
    return {label}
    

if __name__=="__main__":
    uvicorn.run("server:app", port=5000, log_level="info", reload=True)