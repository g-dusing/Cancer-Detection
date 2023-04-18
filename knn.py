import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import Functions

def createKNN(df, k):
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=41)
    print("Split data")
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_std, y_train)
    y_pred = knn.predict(x_test_std)
    acc = accuracy_score(y_pred, y_test)
    print(f"Model accuracy is {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

if __name__ == "__main__":
    df = Functions.preprocess("F:\MS\Spring_23_Semester\Cancer-Detection-main\Original Images")
    print("df created")
    createKNN(df, 2)
