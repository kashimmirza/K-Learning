# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:02:17 2021

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv("data.csv")
print(data.head())
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)
data = data.rename(columns = {"diagnosis":"target"})
print(data.columns)
print("Data Shape:", data.shape) 
data.info() 
sns.countplot(data["target"])
print(data.target.value_counts())
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target] 
describe = data.describe()
print(describe)
corr_matrix = data.corr()

plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix,annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()
threshold = 0.75 
filtre = np.abs(corr_matrix["target"]) > threshold 
corr_features = corr_matrix.columns[filtre].tolist()
sns.heatmap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Theshold 0.75")
plt.show()
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure(figsize=(10,7))
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90) 
plt.show()

sns.pairplot(data[corr_features], diag_kind = "kde", markers = "+", hue = "target")
plt.show()
y = data.target
x = data.drop(["target"], axis = 1)
columns = x.columns.tolist()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)

X_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = X_score
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()
plt.figure(figsize=(10,7))
plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color = "blue", s = 50, label = "outliers")
plt.scatter(x.iloc[:,0],x.iloc[:,1],color = "k", s = 3, label = "Data Points")
radius = (X_score.max()- X_score) / (X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s = 1000*radius, edgecolors = "r", facecolors = "none", label = "Outlier Scores")
plt.legend() 
plt.show()

x = x.drop(outlier_index)
y = y.drop(outlier_index).values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA

test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("Y_train",len(Y_train))
print("Y_test",len(Y_test))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df_describe
X_train_df["target"] = Y_train
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")
plt.figure(figsize=(10,7))
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()
sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()
