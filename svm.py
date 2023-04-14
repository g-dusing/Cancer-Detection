import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import Functions


def createSVM(df):
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=41)
    print("Split data")
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    svm = SVC(kernel="linear", random_state=1, C=0.1)
    svm.fit(x_train_std, y_train)
    y_pred = svm.predict(x_test_std)
    acc = accuracy_score(y_pred, y_test)
    print("Model accuracy is {acc*100}%")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))


if __name__ == "__main__":
    df = Functions.preprocess("C:/Users/Glenn/PycharmProjects/ECE 515/Original Images")
    print("df created")
    createSVM(df)
