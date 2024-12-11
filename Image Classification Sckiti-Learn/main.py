import os
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
#Load Data
cars_path = "clf-data"
#Load the two directories (images with empty cars and images with cars)
categories = os.listdir(cars_path)

print(categories)
data = []
labels =  []
for file in categories:
    folder_path = os.path.join(cars_path, file)
    for image in os.listdir(folder_path):
        data_image = cv.imread(os.path.join(folder_path, image))
        data_image = cv.resize(data_image, (15, 15))
        data.append(data_image.flatten())
        labels.append(file)
        
    
print(len(data), len(labels))

data = np.asarray(data)
labels = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, shuffle=True, test_size=0.2, train_size=0.8, stratify=labels)

print(f'X_train = {X_train.shape}\n, y_train = {y_train.shape}')
print(f'X_test = {X_test.shape}\n, y_test = {y_test.shape}')

# Model
classifier = SVC()
parameters = {'C':[1.0, 2.0, 100, 1000], 'gamma':[0.0001,0.001, 0.01]}

gird = GridSearchCV(classifier, parameters)

gird.fit(X_train, y_train)
best_estimator = gird.best_estimator_
print(f"Best estimator : {best_estimator}")
y_pred = best_estimator.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=y_pred))

score = accuracy_score(y_test, y_pred)
print('\n {}% of the data where classified correctly'.format(score * 100))
pickle.dump(best_estimator, open('./model_SVC.p', 'wb'))