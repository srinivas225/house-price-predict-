# house-price-predict-
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:15:56 2019

@author: srinivas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import os
from PIL import Image 
os.getcwd()                 # get working dir
os.chdir("C:/Users/Admin\\Desktop/my ml projects")       # Setting directory
p=Image.open("C:\\Users\\Admin\\Desktop\\final year project work\\images\\house.jpg")
housing=pd.read_csv("housedata123.csv")
housing.head(5) #TOP 5 ROWS SHOWS
#housing.info() #INFORMATION 
#housing.describe() #its show mean,std,min,25%,50%,75%,max
#housing.hist(bins=50, figsize=(20,15))
'''def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(f"RoWs in train set:{len(train_set)}\nRows in test set: {len(test_set)}\n")'''
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"RoWs in train set:{len(train_set)}\nRows in test set: {len(test_set)}\n")
split= StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    #see test and train data 
strat_test_set.describe()
strat_test_set.info()
strat_train_set['CHAS'].value_counts()
strat_test_set['CHAS'].value_counts()
housing = strat_train_set.copy()

#looking for correlations
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes =['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind='scatter', x='RM',y='MEDV',alpha=0.8)
#trying out attribute combinations
housing['TAXRM'] =housing['TAX']/housing['RM']
housing.head()
housing=strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()
#missing attributes
#take care  of missing data points
#1 get rid of the missing data points
#2 get rid of the whole attribute
#3 set the value to some value(0, mean or  median)
a=housing.dropna(subset=['RM']) #option 1
a.shape
housing.drop("RM", axis=1).shape #option 2
# note that there is no Rm column
median = housing['RM'].median()
housing['RM'].fillna(median)# note that the original housing datafram will remain unchange
housing.shape
housing.describe() # before we started  filling missing attributes
#import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
imputer.statistics_
x=imputer.transform(housing)
housing_tr =pd.DataFrame(x, columns=housing.columns)
housing_tr.describe()
#scikit-learn design
#primarily,three types of objects
#1 Estimators-It estimates some parameters based on a dataset . Eg imputer
#it has a fit method and transform method
#fit method - fits  and the dataset and calculates internal parameters
#2 Transformers -  transform method takes input and returns output based on the learning from fit() it also has a convenience function called fit_transform()
#which file and  then transforms
#3 predictors linearregression model is an example of predictor .fit() and predict() are two comom fuction it also gives score() fuction which will evaluat the predictions

#creating pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('std_scaler',StandardScaler()),])
housing_num_tr =my_pipeline.fit_transform(housing)
#selecting a desired model for dragon real  esstates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
some_data=housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data =  my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)
#evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)
#using better evaluation technique cross validation
from sklearn.model_selection import cross_val_score
scores =cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
def print_scores(scores):
    print("scores:",scores)
    print("mean: ", scores.mean())
    print("Standard deviation:", scores.std())
    

import pickle 
saved_model = pickle.dumps(model) 
# Save the trained model as a pickle string.  
  
# Load the pickled model 
knn_from_pickle = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions
x_test=strat_test_set.drop("MEDV", axis=1) 
y_test= strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions=knn_from_pickle.predict(x_test_prepared) 
final_mse=mean_squared_error(y_test, final_predictions)
final_rmse  =np.sqrt(final_mse)
print(final_predictions, list(y_test))
input =np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -07.144603, -1.31238772,  7.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
knn_from_pickle.predict(input)



