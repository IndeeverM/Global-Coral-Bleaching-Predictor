#!/usr/bin/env python
# coding: utf-8
# Indeever Madireddy 2021 
# <b>Research:</b><br>
# https://www.nature.com/articles/s41467-019-09238-2/ <br> <br>
# <b>Introduction:</b><br>
# <blockquote> Coral bleaching is a fatal process in which corals expel their symbiotic dinoflagellates. Conservation efforts such as targeted reef restoration, developing heat resistant coral strains, and building artificial reefs attempt to restore damaged reefs. However, there are over 360,000 square miles of coral reef worldwide making it challenging to perform targeted reef conservation. Herein, a machine learning model to predict locations at high risk for bleaching was developed. Data was obtained from BCO-DMO and consisted of various coral bleaching events and the parameters under which the bleaching occurred.
# </blockquote>
# 
# <b>Objective:</b><br>
# <ol>
#     <li> Predict Average Bleaching Percentage </li>
# </ol>
# 
# <b>Table of Content: </b> <br>
# <ol>
# <li>Fetch Dataset </li>
# <li>Install & Import Libraries </li>
# <li>Load Datasets </li>
# <li>Exploratory Data Analysis </li>
# <li>Feature Engineering </li>
# <li>Model Development </li>
# <li>Find Prediction </li>
# </ol>
# 
# <blockquote>
#     <ol>
# Datasets:
#         <li>Coral Bleaching https://www.bco-dmo.org/dataset/773466 </li>
#         <li>Temperatures https://climateknowledgeportal.worldbank.org/download-data </li>
#         <li>Co2 Emissions per Country https://ourworldindata.org/co2-and-other-greenhouse-gas-emissions </li>
#         <li>Population data: https://datahub.io/JohnSnowLabs/population-figures-by-country </li>
#     </ol>
# </blockquote>

# <blockquote><b>Install & Import Libraries</b></blockquote>

# In[ ]:


# loading packages
import numpy as np
import pandas as pd
from pandas import datetime
from string import ascii_letters

# data visualization and missing values
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import seaborn as sns

# machine learning
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures , MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, BayesianRidge 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage,dendrogram,cut_tree
from scipy.io import loadmat
from collections import defaultdict
from collections import Counter
from shapely.geometry import Point
#import geopandas as gpd
#from geopandas import GeoDataFrame

import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).
sns.set_style('whitegrid') # One of the five seaborn themes
import warnings
warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, RidgeCV

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


import tkinter as tk 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# <blockquote><b>Load Datasets</b></blockquote>

# In[ ]:


data=pd.read_csv("global_bleaching_environmental.csv")
data.columns


# <blockquote><b>Exploratory Data Analysis</b></blockquote>

# In[ ]:


#Changing date to year
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year

data.groupby(['Year']).size()


# ##### Not enough data before 2002. Hence removing it. Additionally removing negative SSTA values and non-numerical data. 

# In[ ]:


#Not enough data before 2003, so setting the dataset to start from there
data = data[data['Year'] > 2002]


# In[ ]:


# Taking only SSTA > 0
data = data[data['SSTA'] > 0]


# In[ ]:


#Renaming columns
data = data.rename(columns={"Country_Name": "Country", "Depth": "Depth_m", "Average_Bleaching": "Average_Bleaching_percent"})

#Transforming column datatypes
data['Depth_m'] = pd.to_numeric(data['Depth_m'],errors='coerce')
data['Average_Bleaching_percent'] = pd.to_numeric(data['Average_Bleaching_percent'],errors='coerce')

data.head(5)


# In[ ]:


#Check NAN Data and by what percentage
def nan_check(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent_1 = data.isnull().sum()/data.isnull().count()*100
    percent_2 = (np.round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    return missing_data

nan_check(data)


# In[ ]:


#Fill NA into the data object
data.fillna(method ='ffill', inplace = True)

#Count of Null Values In the Data
data.isnull().sum().sort_values(ascending = False).head(35)


# In[ ]:


# Drop the categorical variables existing in the Data
data.drop(["ID", "Realm", "Ecoregion", "Country" ,"State_Island_Province", "City_Town","City_Town_2", "City_Town_3", "Date", "Date2"],inplace=True,axis=1)


# In[ ]:


#Find highly correlated pairs
corr_matrix = data.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
#print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
print(to_drop)


# In[ ]:


# Delete Correlated and other non significant Variables
# I dropped Year column also which I created for data normilization purpose

data.drop([
"Ocean","Temperature_Kelvin","Temperature_Mean", "Temperature_Minimum", "Temperature_Maximum", "Temperature_Kelvin_Standard_Deviation",
"Windspeed", "SSTA_Standard_Deviation", "SSTA_Mean", "SSTA_Minimum", "SSTA_Maximum", "SSTA_Frequency",
"SSTA_Frequency_Standard_Deviation","SSTA_FrequencyMax", "SSTA_FrequencyMean", "SSTA_DHW",
"SSTA_DHW_Standard_Deviation", "SSTA_DHWMax", "SSTA_DHWMean", "TSA", "TSA_Standard_Deviation",
"TSA_Minimum", "TSA_Maximum", "TSA_Mean","TSA_Frequency","TSA_Frequency_Standard_Deviation",
"TSA_FrequencyMax", "TSA_FrequencyMean","TSA_DHW", "TSA_DHW_Standard_Deviation", "TSA_DHWMax",
"TSA_DHWMean","Year"],inplace=True,axis=1)

data.head()


# ##### Define X y

# In[ ]:


#Define X y
# Labels (y) are the values we want to predict


X=data.iloc[:,:-1].values # here we have 5 input variables for multiple
y=data.iloc[:,-1].values # output variable (what we are trying to predict)

print(X.shape,type(X))
print(y.shape,type(y))


# ##### Generate train/test sets

# In[ ]:


# Generate train/test sets
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# ###### MAIN CODE

# In[ ]:


# with best parameters for 300 estimators
forest = RandomForestRegressor(max_features = 'sqrt', max_depth = 16, criterion = 'mse', bootstrap = 'False',
                            n_estimators = 300, random_state = seed)

forest.fit(X_train, y_train)


# In[ ]:


# tkinter GUI
root= tk.Tk()
root.title('Coral Bleaching Predictor')

#Create the Canvas
canvas1 = tk.Canvas(root, width = 600, height = 300,  relief = 'raised')
canvas1.pack()


# In[ ]:


#Header Label
labelheader = tk.Label(root, text='Bleaching Predictor')
labelheader.config(font=('Arial', 20))
canvas1.create_window(270, 60, window=labelheader)

# Latitude In Degrees label and input box
label1 = tk.Label(root, text='Latitude (Degrees): ', justify = 'right')
label1.config(font=('helvetica', 12))
label1.pack(fill=tk.X,side=tk.RIGHT)

canvas1.create_window(40, 100, window=label1)
entry1 = tk.Entry (root) # create 1st entry box
entry1.pack(side=tk.RIGHT)
canvas1.create_window(270, 100, window=entry1)

# Longitude In Degrees label and input box
label2 = tk.Label(root, text='Longitude (Degrees): ', justify = 'right')
label2.config(font=('helvetica', 12))
canvas1.create_window(40, 120, window=label2)
entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)


# Depth
label3 = tk.Label(root, text='Depth (Meters): ', justify = 'right')
label3.config(font=('helvetica', 12))
canvas1.create_window(40, 140, window=label3)
entry3 = tk.Entry (root) # create 3nd entry box
canvas1.create_window(270, 140, window=entry3)

# ClimSST label and input box
label4 = tk.Label(root, text='ClimSST: ', justify = 'right')
label4.config(font=('helvetica', 12))
canvas1.create_window(40, 160, window=label4)
entry4 = tk.Entry (root) # create 3nd entry box
canvas1.create_window(270, 160, window=entry4)

# Recorded Date label and input box
label5 = tk.Label(root, text='Sea Surface Temperature Anomaly (Celsius): ', justify = 'right')
label5.config(font=('helvetica', 12))
canvas1.create_window(40, 180, window=label5)
entry5 = tk.Entry (root) # create 3nd entry box
canvas1.create_window(270, 180, window=entry5)


# In[ ]:


def derivePredictedValue():
    global New_Latitude_Degrees #our 1st input variable
    New_Latitude_Degrees = float(entry1.get())     
    global New_Longitude_Degrees #our 2nd input variable
    New_Longitude_Degrees = float(entry2.get()) 
    global New_Depth #our 3nd input variable
    New_Depth = float(entry3.get()) 
    global New_ClimSST #our 4nd input variable
    New_ClimSST = float(entry4.get())     
    global New_SSTA #our 5nd input variable 
    New_SSTA = float(entry5.get())
    Prediction_result  = ('Predicted Bleaching Percentage: ', forest.predict([[New_Latitude_Degrees ,New_Longitude_Degrees, New_Depth, New_ClimSST, New_SSTA]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    print(Prediction_result)
    canvas1.create_window(250, 280, window=label_Prediction)


# In[ ]:


#Show Button
button1 = tk.Button (root, text='Show',command=derivePredictedValue, bg='brown',  font=('helvetica', 10, 'bold')) # button to call the 'values' command above 
canvas1.create_window(260, 207, window=button1)

#Clear Button 
button2 = tk.Button (root, text='Clear',command=derivePredictedValue, bg='brown', font=('helvetica', 10, 'bold')) # button to call the 'values' command above 
canvas1.create_window(320, 207, window=button2)


# In[ ]:


root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




